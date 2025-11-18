"""
Model definitions for ViT model compression."""

from transformers.models.vit.modeling_vit import ViTForImageClassification
from .configuration_vit_c import ViTConfigCompress
from .indexing_vit_c import VIT_C_INDEXING as INDEXING
from ...utils import replace_layers_upon_init

class ViTForImageClassificationCompress(ViTForImageClassification):
    config_class = ViTConfigCompress

    def __init__(self, config: ViTConfigCompress):
        super().__init__(config)
        print(self)

        replace_layers_upon_init(
            self,
            INDEXING["vit"],
            config,
        )

__all__ = ["ViTForImageClassificationCompress"]