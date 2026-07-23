from transformers.models.vit.modeling_vit import ViTForImageClassification
from transformers.models.vit.configuration_vit import ViTConfig
from . import VIT_C_INDEXING as INDEXING
from .legacy_conversion import build_legacy_vit_key_mapping, peek_checkpoint_keys
from ...utils import (
    init_compressed_config,
    replace_layers_upon_init,
    CompressionSchemesManager,
)

# Define configuration
class ViTConfigCompress(ViTConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        init_compressed_config(
            config_instance=self,
            indexing=INDEXING["vit"],
            compression_config=kwargs.get("compression_config", {})
        )

# Define model
class ViTForImageClassificationCompress(ViTForImageClassification):
    config_class = ViTConfigCompress
    indexing = INDEXING
    def __init__(self, config):
        super().__init__(config)
        replace_layers_upon_init(self, INDEXING["vit"], config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Legacy ViT checkpoints (e.g. google/vit-base-patch16-224[-in21k]) were
        # produced by the original convert_vit_timm_to_pytorch.py script and use
        # attention.attention.query/key/value + encoder.layer naming; some (bare
        # ViTModel backbone checkpoints) also lack the "vit." prefix entirely.
        # transformers' own runtime renaming only fires once that prefix is
        # already present, so we compute the exact old->new key mapping
        # ourselves here instead of depending on it -- see legacy_conversion.py.
        if kwargs.get("key_mapping") is None:
            try:
                raw_keys = peek_checkpoint_keys(
                    pretrained_model_name_or_path,
                    revision=kwargs.get("revision"),
                    subfolder=kwargs.get("subfolder"),
                    cache_dir=kwargs.get("cache_dir"),
                    token=kwargs.get("token"),
                )
            except Exception:
                raw_keys = None
            if raw_keys is not None:
                legacy_mapping = build_legacy_vit_key_mapping(raw_keys)
                if legacy_mapping:
                    kwargs["key_mapping"] = legacy_mapping
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

# Define compression manager
class ViTCompressionSchemesManager(CompressionSchemesManager):
    def __init__(self, model):
        super().__init__(model, INDEXING)

# Define __all__
__all__ = [
    "ViTConfigCompress",
    "ViTForImageClassificationCompress",
    "ViTCompressionSchemesManager",
]
