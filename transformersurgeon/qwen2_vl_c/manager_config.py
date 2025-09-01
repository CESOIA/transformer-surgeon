"""
Model-specific configuration for Qwen2-VL-C model compression.
"""

# Configuration for Qwen2-VL-C model compression schemes
QWEN2_VL_C_CONFIG = {
    'block_configs': [
        {
            'name': 'vision',
            'num_blocks': 32,
            'key_list': ["sa_qkv", "sa_out", "mlp_up", "mlp_down"],
            'path_list': ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            'path_template': "model.visual.blocks.{block_index}.{path}",
            'config_attr': 'vision_config',
            'qkv_keys': ["sa_qkv"]  # Keys that represent QKV concatenated layers
        },
        {
            'name': 'text',
            'num_blocks': 28,
            'key_list': ["sa_q", "sa_k", "sa_v", "sa_out", "mlp_gate", "mlp_up", "mlp_down"],
            'path_list': ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            'path_template': "model.language_model.layers.{block_index}.{path}",
            'config_attr': 'text_config',
            'qkv_keys': []  # No QKV concatenated layers in text blocks
        }
    ]
}

# Example configuration for variable head dimensions
# Use this when you want different head sizes for Q, K, V
QWEN2_VL_C_VARIABLE_HEAD_CONFIG = {
    'block_configs': [
        {
            'name': 'vision',
            'num_blocks': 32,
            'key_list': ["sa_q", "sa_k", "sa_v", "sa_out", "mlp_up", "mlp_down"],
            'path_list': ["attn.q", "attn.k", "attn.v", "attn.proj", "mlp.fc1", "mlp.fc2"],
            'path_template': "model.visual.blocks.{block_index}.{path}",
            'config_attr': 'vision_config',
            'qkv_keys': [],  # No concatenated QKV when using variable heads
            'variable_heads': True,  # Flag to enable variable head dimensions
            # Example: different dimensions for each head
            'head_dim_configs': {
                'sa_q': [64, 48, 64, 32, 64, 48, 64, 32] * 4,  # 32 heads with varying dims
                'sa_k': [64] * 32,  # K heads remain uniform
                'sa_v': [64] * 32,  # V heads remain uniform
            }
        },
        {
            'name': 'text', 
            'num_blocks': 28,
            'key_list': ["sa_q", "sa_k", "sa_v", "sa_out", "mlp_gate", "mlp_up", "mlp_down"],
            'path_list': ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            'path_template': "model.language_model.layers.{block_index}.{path}",
            'config_attr': 'text_config',
            'qkv_keys': [],
            'variable_heads': True,
            'head_dim_configs': {
                'sa_q': [80, 64, 80, 48, 80, 64, 80, 48] * 4,  # 32 heads with varying Q dims
                'sa_k': [64] * 32,  # K heads uniform
                'sa_v': [64] * 32,  # V heads uniform  
            }
        }
    ]
}
