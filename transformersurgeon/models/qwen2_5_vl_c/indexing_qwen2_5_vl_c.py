"""
Model-specific configuration for Qwen2.5-VL-C model compression.
"""

# Configuration for Qwen2.5-VL-C model compression schemes
QWEN2_5_VL_C_INDEXING = {
    'vision':
    {
        'config_attr': 'vision_config',
        'num_blocks_attr': 'depth',

        # HF model structure specifics
        'path_list': {
            'norm1': [],
            'attn': ['qkv', 'proj'],
            'norm2': [],
            'mlp': ['gate_proj', 'up_proj', 'down_proj'],
        },
        'skip_connections': [
            ['norm1', 'attn'],
            ['norm2', 'mlp'],
        ],
        'calibration_groups': {
            'attn': [['qkv'], ['proj']],
            'mlp': [['gate_proj', 'up_proj'], ['down_proj']],
        },
        # Structured-pruning annotations (see llama_c). Vision attention uses a
        # fused qkv projection, so only the gated MLP intermediate is cleanly prunable.
        'pruning': {
            'output_dependence': {
                'mlp.gate_proj': ['mlp.down_proj'],
                'mlp.up_proj': ['mlp.down_proj'],
            },
            'coupled_masks': [
                ['mlp.gate_proj', 'mlp.up_proj'],
            ],
            'coupled_masks_all': [
                ['attn.proj', 'mlp.down_proj'],
            ],
            'per_head_uniform': [],
        },
        'path_template': "model.visual.blocks.{block_index}.{path}",
        'preprocessing': "model.visual.patch_embed",
        'final_layer': "model.visual.merger",
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
        'path_list': {
            'input_layernorm': [],
            'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'post_attention_layernorm': [],
            'mlp': ['gate_proj', 'up_proj', 'down_proj'],
        },
        'skip_connections': [
            ['input_layernorm', 'self_attn'],
            ['post_attention_layernorm', 'mlp'],
        ],
        'calibration_groups': {
            'self_attn': [['q_proj', 'k_proj', 'v_proj'], ['o_proj']],
            'mlp': [['gate_proj', 'up_proj'], ['down_proj']],
        },
        # Structured-pruning annotations (see llama_c for field semantics).
        'pruning': {
            'output_dependence': {
                'mlp.gate_proj': ['mlp.down_proj'],
                'mlp.up_proj': ['mlp.down_proj'],
                'self_attn.v_proj': ['self_attn.o_proj'],
            },
            'coupled_masks': [
                ['self_attn.q_proj', 'self_attn.k_proj'],
                ['mlp.gate_proj', 'mlp.up_proj'],
            ],
            # coupled_masks_all: share ONE output mask across ALL blocks (the
            # residual/hidden-dim writers).
            'coupled_masks_all': [
                ['self_attn.o_proj', 'mlp.down_proj'],
            ],
            'per_head_uniform': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
        },
        'path_template': "model.language_model.layers.{block_index}.{path}",
        'extra_layers': ["model.language_model.norm"],
        'preprocessing': "model.language_model.embed_tokens",
        'position_embeddings_source': "model.language_model.rotary_emb",
        'position_embeddings_targets': ["self_attn"],
        'position_ids_ndim': 3,
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

__all__ = ["QWEN2_5_VL_C_INDEXING"]
