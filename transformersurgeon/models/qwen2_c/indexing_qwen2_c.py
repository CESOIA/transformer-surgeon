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
        # - keys correspond to the model's block (e.g., mha, mlp)
        # - list entries correspond to single layers/projections within those blocks
        # - N.B. non-projection layers (e.g., normalization layers) point to empty lists
        'path_list': {
            'input_layernorm': [],
            'self_attn': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'post_attention_layernorm': [],
            'mlp': ['gate_proj', 'up_proj', 'down_proj'],
        },
        # Skip connections:
        # - they are defined as INPUT of layer A to OUTPUT of layer B
        # - they are defined at the block level
        # - if hf's implementation defines a skip connection internally within a block, it must not be defined here.
        'skip_connections': [
            ['input_layernorm', 'self_attn'],
            ['post_attention_layernorm', 'mlp'],
        ],
        'calibration_groups': {
            'self_attn': [['q_proj', 'k_proj', 'v_proj'], ['o_proj']],
            'mlp': [['gate_proj', 'up_proj'], ['down_proj']],
        },
        # Structured-pruning annotations (model truth; consumed generically):
        # - output_dependence: layer whose OUTPUT rows are pruned -> layers whose
        #   INPUT columns must shrink to match (drives coupled_pruning).
        # - coupled_masks: layers that must share ONE output pruning mask.
        # - per_head_uniform: layers needing uniform ratio per head (recorded only,
        #   not yet consumed).
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
                # 'final_norm' (model.norm) is transparent to the hidden dim
                # and forwards straight to 'final_layer' (lm_head).
                'final_norm': ['final_layer'],
            },
            'coupled_masks': [
                ['self_attn.q_proj', 'self_attn.k_proj'],
                ['mlp.gate_proj', 'mlp.up_proj'],
            ],
            # coupled_masks_all: share ONE output mask across ALL blocks (the
            # residual/hidden-dim writers), so the hidden dim is pruned identically
            # everywhere. 'preprocessing' joins this set as the residual stream's
            # initial writer.
            'coupled_masks_all': [
                ['self_attn.o_proj', 'mlp.down_proj', 'preprocessing'],
            ],
            'per_head_uniform': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
            # position_linked: layers whose output rows are indexed by RoPE
            # position (q/k head slices are paired 1:1 with rotary sin/cos
            # bands). Hard structured pruning removes rows by importance score,
            # not by contiguous rotary band, which desyncs the surviving rows
            # from the positional embedding they were paired with -- see the
            # warning raised in StructuredPruner.apply (structured_pruning.py).
            'position_linked': ['self_attn.q_proj', 'self_attn.k_proj'],
            # Normalization layers: transparent to the embedding/hidden size,
            # never user-compressible, and only ever pruned by forwarding a
            # mask through them (see CoupledPruner.apply_chain).
            'norm_layers': ['input_layernorm', 'post_attention_layernorm'],
        },
        'path_template': "model.layers.{block_index}.{path}",
        'extra_layers': ["model.norm"],
        'final_norm': "model.norm",
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