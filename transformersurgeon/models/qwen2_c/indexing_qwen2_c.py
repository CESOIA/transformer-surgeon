"""
Model-specific configuration for Qwen2-C model compression.
"""

# Configuration for Qwen2-C model compression schemes
QWEN2_C_INDEXING = {
    'text':
    {
        # Attributes in the Hugging Face model config
        'config_attr': "",
        'num_blocks_attr': 'num_hidden_layers',
        'embed_dim_attr': 'hidden_size',
        'num_heads_attr': 'num_attention_heads',
        'mlp_hidden_dim_attr': 'intermediate_size',
        'mlp_activation_attr': 'hidden_act',
        'kv_num_heads_attr': 'num_key_value_heads',

        # HF model structure specifics
        'path_list': {
            'input_layernorm': [],
            'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'post_attention_layernorm': [],
            'mlp': ['gate_proj', 'up_proj', 'down_proj'],
        },
        'calibration_groups': [
            ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
            ['mlp.gate_proj', 'mlp.up_proj'],
        ],
        'path_template': "model.layers.{block_index}.{path}",
        'extra_layers': ["model.norm"],
        'preprocessing': "model.embed_tokens",
        'position_embeddings_source': "model.rotary_emb",
        'position_embeddings_targets': ["self_attn"],
        'position_ids_ndim': 2,
        'final_layer': "lm_head",

        # Transformersurgeon's topology export model specifics
        'structure': 'transformer_decoder',
        'attn_type': 'mha_causal',
        'mlp_type': 'mlp_gated',
        'norm_type': 'rmsnorm',
        'layer_matching': {
            'input_layernorm': 'norm_in',
            'self_attn': ['attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.out_proj'],
            'post_attention_layernorm': 'norm_out',
            'mlp': ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'],
        },
        'bias_required': {
            'attn.q_proj': True,
            'attn.k_proj': True,
            'attn.v_proj': True,
            'attn.out_proj': False,
            'mlp.gate_proj': False,
            'mlp.up_proj': False,
            'mlp.down_proj': False,
            'norm_in': False,
            'norm_out': False,
        },
        'extra_layers_matching': ["norm"],
    }
}

__all__ = ["QWEN2_C_INDEXING"]