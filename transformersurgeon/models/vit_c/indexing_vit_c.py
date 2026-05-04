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
        'path_list': ["attention.attention.query",
                      "attention.attention.key",
                      "attention.attention.value",
                      "attention.output.dense",
                      "layernorm_before",
                      "intermediate.dense",
                      "output.dense",
                      "layernorm_after"],
        'path_template': "vit.encoder.layer.{block_index}.{path}",
        'qkv_paths': [],

        # Transformersurgeon export topology specifics
        'structure': 'transformer_encoder',
        'attn_type': 'mha_encoder',
        'mlp_type': 'mlp',
        'norm_type': 'layernorm',
        'norm_position': 'pre',
        'layer_matching': ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", "norm_in", "mlp.up_proj", "mlp.down_proj", "norm_out"],
        'bias_required': [True, True, True, True, False, True, True, False],
        'extra_layers': ["vit.layernorm"],
        'extra_layers_matching': ["norm"],
        'use_final_norm': True,
        'position_embedding_type': 'none',
    },
}

__all__ = ["VIT_C_INDEXING"]