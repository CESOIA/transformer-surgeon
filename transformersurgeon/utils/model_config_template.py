"""
Example template for creating model-specific configurations.

To add a new model:
1. Create a new folder under transformersurgeon/ (e.g., transformersurgeon/my_model/)
2. Create a model_config.py file similar to this template
3. Create a toolkit.py file that imports the generic CompressionSchemesManager
4. Update the model's __init__.py to export the configuration

Example usage:
    from transformersurgeon.utils import CompressionSchemesManager
    from transformersurgeon.my_model import MY_MODEL_CONFIG
    
    manager = CompressionSchemesManager(config, model, MY_MODEL_CONFIG)
"""

# Template configuration - replace with your model's specific values
MANAGER_CONFIG = {
    'block_configs': [
        {
            'name': 'encoder',  # Name for this block type
            'num_blocks': 12,   # Number of blocks of this type
            'key_list': ["attention_qkv", "attention_out", "mlp_intermediate", "mlp_output"],
            'path_list': ["attention.qkv", "attention.output", "mlp.intermediate", "mlp.output"],
            'path_template': "model.encoder.layers.{block_index}.{path}",
            'config_attr': 'encoder_config',  # Attribute in main config for this block
            'qkv_keys': ["attention_qkv"]     # Keys that are QKV concatenated (optional)
        },
        {
            'name': 'decoder',
            'num_blocks': 12,
            'key_list': ["self_attn_q", "self_attn_k", "self_attn_v", "self_attn_out", 
                        "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_out",
                        "mlp_gate", "mlp_up", "mlp_down"],
            'path_list': ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                         "cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.o_proj",
                         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            'path_template': "model.decoder.layers.{block_index}.{path}",
            'config_attr': 'decoder_config',
            'qkv_keys': []
        }
    ]
}

# Example of a simple model with just one block type
SIMPLE_MANAGER_CONFIG = {
    'block_configs': [
        {
            'name': 'transformer_layers',
            'num_blocks': 24,
            'key_list': ["attn_qkv", "attn_out", "mlp_fc1", "mlp_fc2"],
            'path_list': ["attn.qkv", "attn.out", "mlp.fc1", "mlp.fc2"],
            'path_template': "model.layers.{block_index}.{path}",
            'config_attr': 'model_config',
            'qkv_keys': ["attn_qkv"]
        }
    ]
}
