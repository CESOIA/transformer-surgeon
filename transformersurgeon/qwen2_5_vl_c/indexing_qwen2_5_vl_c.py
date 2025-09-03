"""
Model-specific configuration for Qwen2-VL-C model compression.
"""

# Configuration for Qwen2-VL-C model compression schemes
QWEN2_5_VL_C_INDEXING = {
    'vision':
    {
        'name': 'vision',
        'num_blocks': 32,
        'key_list': ["sa_qkv", "sa_out", "mlp_gate", "mlp_up", "mlp_down"],
        'path_list': ["attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        'path_template': "model.visual.blocks.{block_index}.{path}",
        'config_attr': 'vision_config',
        'key_mappings': {
            "sa_all": ["sa_qkv", "sa_out"],
            "mlp_all": ["mlp_gate", "mlp_up", "mlp_down"],
            "all": ["sa_qkv", "sa_out", "mlp_gate", "mlp_up", "mlp_down"]
        },
        'qkv_keys': ["sa_qkv"],  # Keys that represent QKV concatenated layers
        'lrd_supported': [1, 1, 1, 1, 1],
        'pruning_supported': [0, 0, 0, 0, 0],
    },
    'text':
    {
        'name': 'text',
        'num_blocks': 28,
        'key_list': ["sa_q", "sa_k", "sa_v", "sa_out", "mlp_gate", "mlp_up", "mlp_down"],
        'path_list': ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        'path_template': "model.language_model.layers.{block_index}.{path}",
        'config_attr': 'text_config',
        'key_mappings': {
            "sa_qkv": ["sa_q", "sa_k", "sa_v"],
            "sa_all": ["sa_q", "sa_k", "sa_v", "sa_out"],
            "mlp_all": ["mlp_gate", "mlp_up", "mlp_down"],
            "all": ["sa_q", "sa_k", "sa_v", "sa_out", "mlp_gate", "mlp_up", "mlp_down"]
        },
        'qkv_keys': [],  # No QKV concatenated layers in text blocks
        'lrd_supported': [1, 1, 1, 1, 1],
        'pruning_supported': [0, 0, 0, 0, 0],
    }
}
