# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
Model-specific configuration for ViT model compression.
"""

# Configuration for ViT model compression schemes
VIT_C_INDEXING = {
    'vit':
    {
        # Attributes in the Hugging Face model config
        'config_attr': '',
        'num_blocks_attr': 'num_hidden_layers',
        'embed_dim_attr': 'hidden_size',
        'num_heads_attr': 'num_attention_heads',
        'mlp_hidden_dim_attr': 'intermediate_size',
        'mlp_activation_attr': 'hidden_act',

        # HF model structure specifics
        'path_list': {
            'layernorm_before': [],
            'attention.attention': ['query', 'key', 'value'],
            'attention.output': ['dense'],
            'layernorm_after': [],
            'intermediate': ['dense'],
            'output': ['dense'],
        },
        'calibration_groups': [
            ['attention.attention.query', 'attention.attention.key', 'attention.attention.value'],
        ],
        'path_template': "vit.encoder.layer.{block_index}.{path}",
        'qkv_paths': [],
        'preprocessing': "vit.embeddings",
        'final_layer': "classifier",

        # Transformersurgeon export topology specifics
        'structure': 'transformer_encoder',
        'attn_type': 'mha_encoder',
        'mlp_type': 'mlp',
        'norm_type': 'layernorm',
        'norm_position': 'pre',
        'layer_matching': {
            'layernorm_before': 'norm_in',
            'attention.attention': ['attn.q_proj', 'attn.k_proj', 'attn.v_proj'],
            'attention.output': ['attn.out_proj'],
            'layernorm_after': 'norm_out',
            'intermediate': ['mlp.up_proj'],
            'output': ['mlp.down_proj'],
        },
        'bias_required': {
            'attn.q_proj': True,
            'attn.k_proj': True,
            'attn.v_proj': True,
            'attn.out_proj': True,
            'norm_in': False,
            'mlp.up_proj': True,
            'mlp.down_proj': True,
            'norm_out': False,
        },
        'extra_layers': ["vit.layernorm"],
        'extra_layers_matching': ["norm"],
        'use_final_norm': True,
        'position_embedding_type': 'none',
    },
}

__all__ = ["VIT_C_INDEXING"]