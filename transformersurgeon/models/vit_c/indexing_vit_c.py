# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
Model-specific configuration for ViT model compression.
"""

# Configuration for ViT model compression schemes
VIT_C_INDEXING = {
    'vit':
    {
        'config_attr': '',
        'num_blocks_attr': 'num_hidden_layers',
        'path_list': ["attention.attention.query",
                      "attention.attention.key",
                      "attention.attention.value",
                      "attention.output.dense",
                      "intermediate.dense",
                      "output.dense"],
        'path_template': "vit.encoder.layer.{block_index}.{path}",
        'qkv_paths': [],
    },
}

__all__ = ["VIT_C_INDEXING"]