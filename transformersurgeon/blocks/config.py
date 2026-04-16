"""HF-style configuration for converted custom decoder graph compression."""

from transformers import PretrainedConfig

from ..utils.configuration import init_compressed_config

class CustomDecoderConfigCompress(PretrainedConfig):
    """Minimal decoder config used by converted TransformerDecoder graphs."""

    model_type = "transformersurgeon-custom-decoder"

    def __init__(
        self,
        num_hidden_layers=0,
        hidden_size=0,
        num_attention_heads=1,
        intermediate_size=0,
        hidden_act="silu",
        attn_type="mha_causal",
        mlp_type="mlp_gated",
        norm_type="rmsnorm",
        num_key_value_heads=None,
        compression_config=None,
        bias_required=None,
        use_sdpa=False,
        max_cache_len=2048,
        indexing=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_key_value_heads = num_key_value_heads

        self.attn_type = attn_type
        self.mlp_type = mlp_type
        self.norm_type = norm_type
        self.use_sdpa = use_sdpa
        self.max_cache_len = max_cache_len
        self.bias_required = bias_required or {"attn": {}, "mlp": {}}

        if indexing is None:
            self.compression_config = compression_config or {}
        else:
            init_compressed_config(
                config_instance=self,
                indexing=indexing,
                compression_config=compression_config or {},
            )

    @classmethod
    def from_source_config(
        cls,
        source_config_obj,
        source_indexing,
        converted_indexing,
        compression_config=None,
        bias_required=None,
        use_sdpa=False,
        max_cache_len=2048,
    ):
        """Build converted decoder config from source HF config using indexing metadata."""

        source_cfg_dict = source_config_obj.to_dict()

        def _get_attr_from_source(index_key):
            source_attr = source_indexing.get(index_key)
            if source_attr is None:
                raise KeyError(
                    f"Indexing is missing '{index_key}', required to build decoder config."
                )

            if source_attr in source_cfg_dict:
                return source_cfg_dict[source_attr]
            if hasattr(source_config_obj, source_attr):
                return getattr(source_config_obj, source_attr)

            raise KeyError(
                f"Source config is missing attribute '{source_attr}' mapped by '{index_key}'."
            )

        consumed_attrs = {
            source_indexing["num_blocks_attr"],
            source_indexing["embed_dim_attr"],
            source_indexing["num_heads_attr"],
            source_indexing["mlp_hidden_dim_attr"],
            source_indexing["mlp_activation_attr"],
        }
        if "kv_num_heads_attr" in source_indexing:
            consumed_attrs.add(source_indexing["kv_num_heads_attr"])

        passthrough_kwargs = {
            k: v
            for k, v in source_cfg_dict.items()
            if k not in consumed_attrs
            and k not in {
                "compression_config",
                "bias_required",
                "attn_type",
                "mlp_type",
                "norm_type",
                "use_sdpa",
                "max_cache_len",
            }
        }

        return cls(
            num_hidden_layers=_get_attr_from_source("num_blocks_attr"),
            hidden_size=_get_attr_from_source("embed_dim_attr"),
            num_attention_heads=_get_attr_from_source("num_heads_attr"),
            intermediate_size=_get_attr_from_source("mlp_hidden_dim_attr"),
            hidden_act=_get_attr_from_source("mlp_activation_attr"),
            num_key_value_heads=(
                _get_attr_from_source("kv_num_heads_attr")
                if "kv_num_heads_attr" in source_indexing
                else None
            ),
            attn_type=source_indexing.get("attn_type", "mha_causal"),
            mlp_type=source_indexing.get("mlp_type", "mlp"),
            norm_type=source_indexing.get("norm_type", "rmsnorm"),
            compression_config=compression_config or {},
            bias_required=bias_required or {"attn": {}, "mlp": {}},
            use_sdpa=use_sdpa,
            max_cache_len=max_cache_len,
            indexing=converted_indexing,
            **passthrough_kwargs,
        )

__all__ = ["CustomDecoderConfigCompress"]
