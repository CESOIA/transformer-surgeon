"""Configuration objects of the converted (export-ready) stacks.

A converted stack is described by a Hugging-Face-style configuration holding:

- the model geometry (widths, number of heads and blocks), read from the source
  model's configuration through the attribute names listed in the model
  dictionary (``embed_dim_attr``, ``num_heads_attr``, ...);
- the atoms the model needs (``attn_type``, ``mlp_type``, ``norm_type``,
  ``rope_type``), also from the model dictionary;
- the implementation choices of the export (``attn_impl``, ``cache_impl``,
  ``rmsnorm_impl``, ...), see ``export/operations.py``;
- ``layer_specs``: the geometry of every already-compressed linear layer
  (``{"blocks.0.attn.q_proj": {"in_features", "out_features", "rank", "bias"}}``).
"""

from transformers import PretrainedConfig

from ..positions.rope import resolve_rope_type

# Source-configuration attribute names read through the model dictionary.
_GEOMETRY_KEYS = {
    "num_hidden_layers": "num_blocks_attr",
    "hidden_size": "embed_dim_attr",
    "num_attention_heads": "num_heads_attr",
    "intermediate_size": "mlp_hidden_dim_attr",
    "hidden_act": "mlp_activation_attr",
}


def rmsnorm_impl_from_flags(prescale=True, upcast=False):
    """Map the older boolean options ``rmsnorm_prescale``/``rmsnorm_upcast`` to an ``rmsnorm_impl`` name."""
    if upcast:
        return "rmsnorm_upcast"
    return "rmsnorm_prescaled" if prescale else "rmsnorm"


class _ConvertedStackConfig(PretrainedConfig):
    """Fields and source-configuration mapping shared by decoder and encoder stacks."""

    _defaults = {}

    def __init__(self, num_hidden_layers=0, hidden_size=0, num_attention_heads=1, intermediate_size=0,
                 hidden_act="silu", num_key_value_heads=None, layer_specs=None, attn_impl=None,
                 use_sdpa=False, rmsnorm_impl=None, **kwargs):
        options = {key: kwargs.pop(key, default) for key, default in self._defaults.items()}
        prescale = kwargs.pop("rmsnorm_prescale", True)
        upcast = kwargs.pop("rmsnorm_upcast", False)
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_key_value_heads = num_key_value_heads
        self.layer_specs = layer_specs or {}
        self.attn_impl = attn_impl
        self.use_sdpa = use_sdpa
        self.rmsnorm_impl = rmsnorm_impl or rmsnorm_impl_from_flags(prescale, upcast)
        for key, value in options.items():
            setattr(self, key, value)

    def block_layer_specs(self, block_index, submodule):
        """``{layer name: spec}`` of the linear layers of ``blocks.<i>.<submodule>``."""
        prefix = f"blocks.{block_index}.{submodule}."
        return {path[len(prefix):]: spec for path, spec in self.layer_specs.items() if path.startswith(prefix)}

    @classmethod
    def from_source_config(cls, source_config, indexing, layer_specs=None, **options):
        """Build the configuration from a source model configuration and its model dictionary.

        ``options`` are the export implementation choices (``attn_impl``,
        ``cache_impl``, ...). Fields of the source configuration that are not
        consumed here are passed through unchanged (for example ``rms_norm_eps``
        or the rotary-embedding parameters).
        """
        source = source_config.to_dict()

        def read(index_key):
            attr = indexing.get(index_key)
            if attr is None:
                raise KeyError(f"Model dictionary is missing '{index_key}', required to build the stack config.")
            if attr in source:
                return source[attr]
            if hasattr(source_config, attr):
                return getattr(source_config, attr)
            raise KeyError(f"Source config is missing attribute '{attr}' mapped by '{index_key}'.")

        values = {field: read(key) for field, key in _GEOMETRY_KEYS.items()}
        consumed = {indexing[key] for key in _GEOMETRY_KEYS.values()}
        if "kv_num_heads_attr" in indexing:
            values["num_key_value_heads"] = read("kv_num_heads_attr")
            consumed.add(indexing["kv_num_heads_attr"])
        for key, default in cls._model_keys.items():
            values[key] = indexing.get(key, default)
        reserved = consumed | set(values) | set(options) | {"compression_config", "layer_specs"}
        passthrough = {k: v for k, v in source.items() if k not in reserved}
        values = cls._adapt_model_values(values, source_config)
        return cls(layer_specs=layer_specs or {}, **values, **options, **passthrough)

    @classmethod
    def _adapt_model_values(cls, values, source_config):
        return values


class CustomDecoderConfigCompress(_ConvertedStackConfig):
    """Configuration of :class:`~transformersurgeon.blocks.stacks.decoder.TransformerDecoder`."""

    model_type = "transformersurgeon-custom-decoder"
    # Atoms named by the model dictionary, with their defaults.
    _model_keys = {"attn_type": "mha_causal", "mlp_type": "mlp", "norm_type": "rmsnorm", "rope_type": "default"}
    # Export implementation options, with their defaults.
    _defaults = {
        **_model_keys,
        "cache_impl": "mutable",
        "max_cache_len": 2048,
        "add_batch_dim": False,
        "masked_softmax": False,
    }

    def __init__(self, **kwargs):
        # transformers generates a constructor for config subclasses without their own.
        super().__init__(**kwargs)

    @classmethod
    def _adapt_model_values(cls, values, source_config):
        values["rope_type"] = resolve_rope_type(source_config, values["rope_type"])
        return values


class CustomEncoderConfigCompress(_ConvertedStackConfig):
    """Configuration of :class:`~transformersurgeon.blocks.stacks.encoder.TransformerEncoder`."""

    model_type = "transformersurgeon-custom-encoder"
    _model_keys = {"attn_type": "mha_encoder", "mlp_type": "mlp", "norm_type": "layernorm",
                   "norm_position": "pre", "use_final_norm": True}
    _defaults = dict(_model_keys)

    def __init__(self, **kwargs):
        # transformers generates a constructor for config subclasses without their own.
        super().__init__(**kwargs)


__all__ = ["CustomDecoderConfigCompress", "CustomEncoderConfigCompress", "rmsnorm_impl_from_flags"]
