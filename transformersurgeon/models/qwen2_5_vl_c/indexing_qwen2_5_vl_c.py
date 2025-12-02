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
        # Attributes in the Hugging Face model config
        'config_attr': 'text_config',
        'num_blocks_attr': 'num_hidden_layers',
        'embed_dim_attr': 'hidden_size',
        'num_heads_attr': 'num_attention_heads',
        'mlp_hidden_dim_attr': 'intermediate_size',
        'mlp_activation_attr': 'hidden_act',
        'kv_num_heads_attr': 'num_key_value_heads',

        # HF model structure specifics
        'path_list': ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm"],
        'path_template': "model.language_model.layers.{block_index}.{path}",
        'qkv_paths': [],  # No QKV concatenated layers in text blocks

        # Transformersurgeon's topology export model specifics
        'structure': 'transformer_decoder',
        'attn_type': 'mha_causal',
        'mlp_type': 'mlp_gated',
        'norm_type': 'rmsnorm',
        'layer_matching': ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "norm_in", "norm_out"],
    }
}
