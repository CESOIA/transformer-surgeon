"""
Model-specific configuration for Qwen2-VL-C model compression.
"""

# Configuration for Qwen2-VL-C model compression schemes
QWEN2_VL_C_INDEXING = {
    'vision':
    {
        'config_attr': 'vision_config',
        'num_blocks_attr': 'depth',
        'path_list': {
            'attn': ['qkv', 'proj'],
            'mlp': ['fc1', 'fc2'],
        },
        'calibration_groups': [],
        'path_template': "model.visual.blocks.{block_index}.{path}",
        'qkv_paths': ["attn.qkv"],  # Paths that represent QKV concatenated layers
        'pruning_supported': [],
    },
    'text':
    {
        'config_attr': 'text_config',
        'num_blocks_attr': 'num_hidden_layers',
        'path_list': {
            'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'mlp': ['gate_proj', 'up_proj', 'down_proj'],
        },
        'calibration_groups': [
            ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
            ['mlp.gate_proj', 'mlp.up_proj'],
        ],
        'path_template': "model.language_model.layers.{block_index}.{path}",
        'position_embeddings_source': "model.language_model.rotary_emb",
        'position_embeddings_targets': ["self_attn"],
        'position_ids_ndim': 2,
        'qkv_paths': [],  # No QKV concatenated layers in text blocks
        'lrd_supported': ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        'pruning_supported': [],
    }
}

__all__ = ["QWEN2_VL_C_INDEXING"]