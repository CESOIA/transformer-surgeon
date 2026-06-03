# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
Model-specific configuration for Bert model compression.
"""

# Configuration for Bert model compression schemes
BERT_C_INDEXING = {
    'bert':
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
            'attention.self': ['query', 'key', 'value'],
            'attention.output': ['dense'],
            'attention.output.LayerNorm': [],
            'intermediate': ['dense'],
            'output': ['dense'],
            'output.LayerNorm': [],
        },
        'calibration_groups': [
            ['attention.self.query', 'attention.self.key', 'attention.self.value'],
        ],
        'no_cascade_calibration': True,
        'path_template': "bert.encoder.layer.{block_index}.{path}",
        'qkv_paths': [],
        'preprocessing': "bert.embeddings",
        'final_layer': "classifier",

        # Transformersurgeon export topology specifics
        'structure': 'transformer_encoder',
        'attn_type': 'mha_encoder',
        'mlp_type': 'mlp',
        'norm_type': 'layernorm',
        'norm_position': 'post',
        'layer_matching': {
            'attention.self': ['attn.q_proj', 'attn.k_proj', 'attn.v_proj'],
            'attention.output': ['attn.out_proj'],
            'attention.output.LayerNorm': 'norm_in',
            'intermediate': ['mlp.up_proj'],
            'output': ['mlp.down_proj'],
            'output.LayerNorm': 'norm_out',
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
        'use_final_norm': False,
        'position_embedding_type': 'none',
    },
}

__all__ = ["BERT_C_INDEXING"]
