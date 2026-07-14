# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
Model-specific configuration for ViT model compression.
"""

# Configuration for ViT model compression schemes
VIT_C_INDEXING = {
    'vit':
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
            'layernorm_before': [],
            'attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'layernorm_after': [],
            'mlp': ['fc1', 'fc2'],
        },
        'skip_connections': [
            ['layernorm_before', 'attention'],
            ['layernorm_after', 'mlp'],
        ],
        'calibration_groups': {
            'attention': [['q_proj', 'k_proj', 'v_proj'], ['o_proj']],
            'mlp': [['fc1'], ['fc2']],
        },
        # Structured-pruning annotations (see llama_c for field semantics).
        'pruning': {
            'output_dependence': {
                # 'vit.embeddings' as a whole is a composite module
                # (patch-embed conv + cls token + position embeddings), not a
                # plain nn.Embedding, so it isn't given a CompressionScheme --
                # no 'preprocessing' sentinel here. Its patch-embed *conv*
                # leaf (vit.embeddings.patch_embeddings.projection) is
                # separately compressible though -- see 'preprocessing_conv'.
                #
                # ViT is pre-norm: layernorm_after sits between
                # attention.o_proj's (residual-summed) output and mlp.fc1's
                # input; mlp.fc2's (residual-summed) output feeds the *next*
                # block's layernorm_before, or, on the last block,
                # 'final_norm' (vit.layernorm, the pre-classifier norm kept
                # in extra_layers) -> 'final_layer' (classifier).
                # 'preprocessing_conv' is a sentinel: the conv's output feeds
                # block 0's residual stream via layernorm_before. Resolved
                # directly (no block_index) by the manager/structured pruner,
                # not via path_template.
                'preprocessing_conv': ['layernorm_before'],
                'mlp.fc1': ['mlp.fc2'],
                'mlp.fc2': ['layernorm_before', 'final_norm'],
                'layernorm_before': ['attention.q_proj', 'attention.k_proj', 'attention.v_proj'],
                'attention.v_proj': ['attention.o_proj'],
                'attention.o_proj': ['layernorm_after'],
                'layernorm_after': ['mlp.fc1'],
                # 'final_norm' (vit.layernorm) is transparent to the hidden
                # dim and forwards straight to 'final_layer' (classifier).
                'final_norm': ['final_layer'],
            },
            'coupled_masks': [
                ['attention.q_proj', 'attention.k_proj'],
            ],
            # coupled_masks_all: share ONE output mask across ALL blocks (the
            # residual/hidden-dim writers). 'preprocessing_conv' joins this
            # set as the residual stream's initial writer.
            'coupled_masks_all': [
                ['attention.o_proj', 'mlp.fc2', 'preprocessing_conv'],
            ],
            'per_head_uniform': ['attention.q_proj', 'attention.k_proj', 'attention.v_proj'],
            # Normalization layers: transparent to the embedding/hidden size,
            # never user-compressible, and only ever pruned by forwarding a
            # mask through them (see CoupledPruner.apply_chain).
            'norm_layers': ['layernorm_before', 'layernorm_after'],
        },
        'path_template': "vit.layers.{block_index}.{path}",
        'qkv_paths': [],
        'preprocessing': "vit.embeddings",
        'preprocessing_conv': "vit.embeddings.patch_embeddings.projection",
        # cls_token/position_embeddings are concatenated/added directly to
        # the conv's output along the hidden dim inside ViTEmbeddings (a
        # composite module, hence no 'preprocessing' scheme) -- hard-pruning
        # 'preprocessing_conv' also index-selects these along their last dim
        # with the same mask, see StructuredPruner._prune_extra_params.
        'preprocessing_conv_extra_params': [
            "vit.embeddings.cls_token",
            "vit.embeddings.position_embeddings",
        ],
        'final_layer': "classifier",
        'final_norm': "vit.layernorm",

        # Transformersurgeon export topology specifics
        'structure': 'transformer_encoder',
        'attn_type': 'mha_encoder',
        'mlp_type': 'mlp',
        'norm_type': 'layernorm',
        'norm_position': 'pre',
        'layer_matching': {
            'layernorm_before': 'norm_in',
            'attention': ['attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.out_proj'],
            'layernorm_after': 'norm_out',
            'mlp': ['mlp.up_proj', 'mlp.down_proj'],
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
        'extra_layers': ["vit.layernorm"],
        'extra_layers_matching': ["norm"],
        'use_final_norm': True,
        'position_embedding_type': 'none',
    },
}

__all__ = ["VIT_C_INDEXING"]