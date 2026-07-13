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
                # 'preprocessing' is a sentinel: the embedding's output feeds
                # block 0's residual stream via input_layernorm. Resolved
                # directly (no block_index) by the manager/structured pruner,
                # not via path_template.
                'preprocessing': ['input_layernorm'],
                'input_layernorm': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
                'mlp.gate_proj': ['mlp.down_proj'],
                'mlp.up_proj': ['mlp.down_proj'],
                # Wraps to the next block's input_layernorm (or, on the last
                # block, to 'final_norm' -- see the 'final_norm'/'final_layer'
                # gating in structured_pruning.py).
                'mlp.down_proj': ['input_layernorm', 'final_norm'],
                'self_attn.v_proj': ['self_attn.o_proj'],
                'self_attn.o_proj': ['post_attention_layernorm', 'final_norm'],
                'post_attention_layernorm': ['mlp.up_proj', 'mlp.gate_proj'],
                # 'final_norm' (model.language_model.norm) is transparent to
                # the hidden dim and forwards straight to 'final_layer' (lm_head).
                'final_norm': ['final_layer'],
            },
            'coupled_masks': [
                ['self_attn.q_proj', 'self_attn.k_proj'],
                ['mlp.gate_proj', 'mlp.up_proj'],
            ],
            # coupled_masks_all: share ONE output mask across ALL blocks (the
            # residual/hidden-dim writers). 'preprocessing' joins this set as
            # the residual stream's initial writer.
            'coupled_masks_all': [
                ['self_attn.o_proj', 'mlp.down_proj', 'preprocessing'],
            ],
            'per_head_uniform': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
            # Normalization layers: transparent to the embedding/hidden size,
            # never user-compressible, and only ever pruned by forwarding a
            # mask through them (see CoupledPruner.apply_chain).
            'norm_layers': ['input_layernorm', 'post_attention_layernorm'],
        },
        'path_template': "model.language_model.layers.{block_index}.{path}",
        'preprocessing': "model.language_model.embed_tokens",
        'final_layer': "lm_head",
        'final_norm': "model.language_model.norm",
        'position_embeddings_source': "model.language_model.rotary_emb",
        'position_embeddings_targets': ["self_attn"],
        'position_ids_ndim': 2,
        'qkv_paths': [],  # No QKV concatenated layers in text blocks
        'lrd_supported': ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        'pruning_supported': [],
    }
}

__all__ = ["QWEN2_VL_C_INDEXING"]