# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
Model-specific configuration for Bert model compression.
"""

# Configuration for Bert model compression schemes
BERT_C_INDEXING = {
    'bert':
    {
        # Attributes in the Hugging Face model config
        'config_attr': '',
        'num_blocks_attr': 'num_hidden_layers',
        'embed_dim_attr': 'hidden_size',
        'num_heads_attr': 'num_attention_heads',
        'mlp_hidden_dim_attr': 'intermediate_size',
        'mlp_activation_attr': 'hidden_act',

        # HF model structure specifics
        'path_list': {
            'attention.self': ['query', 'key', 'value'],
            'attention.output': ['dense'],
            'attention.output.LayerNorm': [],
            'intermediate': ['dense'],
            'output': ['dense'],
            'output.LayerNorm': [],
        },
        'calibration_groups': [
            ['attention.self.query', 'attention.self.key', 'attention.self.value'],
        ],
        # Structured-pruning annotations (see llama_c for field semantics).
        'pruning': {
            'output_dependence': {
                # 'bert.embeddings' is a composite module (word + position +
                # token_type embeddings + LayerNorm), not a plain nn.Embedding,
                # so it isn't given a CompressionScheme -- no 'preprocessing'
                # sentinel here.
                #
                # BERT is post-norm: attention.output.LayerNorm sits between
                # attention.output.dense's (residual-summed) output and
                # intermediate.dense's input; output.LayerNorm sits between
                # output.dense's output and the *next* block's q/k/v (or, on
                # the last block, 'final_layer' -- there's no separate final
                # norm in a post-norm architecture, the last block's own
                # output.LayerNorm already plays that role).
                'attention.self.value': ['attention.output.dense'],
                'attention.output.dense': ['attention.output.LayerNorm'],
                'attention.output.LayerNorm': ['intermediate.dense'],
                'intermediate.dense': ['output.dense'],
                'output.dense': ['output.LayerNorm'],
                'output.LayerNorm': ['attention.self.query', 'attention.self.key', 'attention.self.value', 'final_layer'],
            },
            'coupled_masks': [
                ['attention.self.query', 'attention.self.key'],
            ],
            # coupled_masks_all: share ONE output mask across ALL blocks (the
            # residual/hidden-dim writers).
            'coupled_masks_all': [
                ['attention.output.dense', 'output.dense'],
            ],
            'per_head_uniform': ['attention.self.query', 'attention.self.key', 'attention.self.value'],
            # Normalization layers: transparent to the embedding/hidden size,
            # never user-compressible, and only ever pruned by forwarding a
            # mask through them (see CoupledPruner.apply_chain).
            'norm_layers': ['attention.output.LayerNorm', 'output.LayerNorm'],
        },
        'no_cascade_calibration': True,
        'path_template': "bert.encoder.layer.{block_index}.{path}",
        'qkv_paths': [],
        'preprocessing': "bert.embeddings",
        'final_layer': "classifier",

        # Transformersurgeon export topology specifics
        'structure': 'transformer_encoder',
        'attn_type': 'mha_encoder',
        'mlp_type': 'mlp',
        'norm_type': 'layernorm',
        'norm_position': 'post',
        'layer_matching': {
            'attention.self': ['attn.q_proj', 'attn.k_proj', 'attn.v_proj'],
            'attention.output': ['attn.out_proj'],
            'attention.output.LayerNorm': 'norm_in',
            'intermediate': ['mlp.up_proj'],
            'output': ['mlp.down_proj'],
            'output.LayerNorm': 'norm_out',
        },
        'bias_required': {
            'attn.q_proj': True,
            'attn.k_proj': True,
            'attn.v_proj': True,
            'attn.out_proj': True,
            'norm_in': False,
            'mlp.up_proj': True,
            'mlp.down_proj': True,
            'norm_out': False,
        },
        'use_final_norm': False,
        'position_embedding_type': 'none',
    },
}

__all__ = ["BERT_C_INDEXING"]
