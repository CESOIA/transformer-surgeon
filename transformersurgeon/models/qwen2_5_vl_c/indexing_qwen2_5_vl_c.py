"""
Model-specific configuration for Qwen2.5-VL-C model compression.
"""

# Configuration for Qwen2.5-VL-C model compression schemes
QWEN2_5_VL_C_INDEXING = {
    'vision':
    {
        'config_attr': 'vision_config',
        'num_blocks_attr': 'depth',
        'path_list': ["attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        'path_template': "model.visual.blocks.{block_index}.{path}",
        'qkv_paths': ["sa_qkv"],  # Keys that represent QKV concatenated layers
        'pruning_supported': [],
    },
    'text':
    {
        'config_attr': 'text_config',
        'num_blocks_attr': 'num_hidden_layers',
        'path_list': ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm"],
        'path_template': "model.language_model.layers.{block_index}.{path}",
        'structure': 'transformer_decoder',
        'block_types': ["mha_causal", "mlp_gated", "rms_norm"],
        'layer_matching': ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "norm_in", "norm_out"],
        'qkv_paths': [],  # No QKV concatenated layers in text blocks
        'pruning_supported': [],
    }
}
