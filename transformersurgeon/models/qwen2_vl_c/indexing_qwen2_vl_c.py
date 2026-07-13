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
            'norm1': [],
            'attn': ['qkv', 'proj'],
            'norm2': [],
            'mlp': ['fc1', 'fc2'],
        },
        'skip_connections': [
            ['norm1', 'attn'],
            ['norm2', 'mlp'],
        ],
        'calibration_groups': {
            'attn': [['qkv'], ['proj']],
            'mlp': [['fc1'], ['fc2']],
        },
        # Structured-pruning annotations (see llama_c). Vision attention uses a
        # fused qkv projection, so only the MLP intermediate is cleanly prunable.
        'pruning': {
            'output_dependence': {
                'mlp.fc1': ['mlp.fc2'],
                'mlp.fc2': ['attn.qkv'],
                'attn.proj': ['mlp.fc1'],
            },
            'coupled_masks': [],
            'coupled_masks_all': [
                ['attn.proj', 'mlp.fc2'],
            ],
            'per_head_uniform': [],
        },
        'path_template': "model.visual.blocks.{block_index}.{path}",
        'qkv_paths': ["attn.qkv"],  # Paths that represent QKV concatenated layers
        'pruning_supported': [],
    },
    'text':
    {
        'config_attr': 'text_config',
        'num_blocks_attr': 'num_hidden_layers',
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
                'mlp.down_proj': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
                'self_attn.v_proj': ['self_attn.o_proj'],
                'self_attn.o_proj': ['mlp.up_proj', 'mlp.gate_proj'],
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
        'position_embeddings_source': "model.language_model.rotary_emb",
        'position_embeddings_targets': ["self_attn"],
        'position_ids_ndim': 2,
        'qkv_paths': [],  # No QKV concatenated layers in text blocks
        'lrd_supported': ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        'pruning_supported': [],
    }
}

__all__ = ["QWEN2_VL_C_INDEXING"]