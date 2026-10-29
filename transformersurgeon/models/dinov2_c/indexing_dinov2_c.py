"""
Model-specific configuration for DINOv2-C model compression.
"""

# Configuration for DINOv2-C model compression schemes
DINOV2_C_INDEXING = {
    'dinov2':
    {
        # Attributes in the Hugging Face model config. There is no
        # 'mlp_hidden_dim_attr': DINOv2 gives the MLP width as `mlp_ratio`, and the
        # converted encoder reads it from the fc1/fc2 weights instead.
        'config_attr': '',
        'num_blocks_attr': 'num_hidden_layers',
        'embed_dim_attr': 'hidden_size',
        'num_heads_attr': 'num_attention_heads',
        'mlp_activation_attr': 'hidden_act',

        # HF model structure specifics. layer_scale1/2 multiply each branch output
        # per channel before the residual addition; they are listed (as
        # normalization-like layers, see 'norm_layers') so that conversion copies them.
        'path_list': {
            'norm1': [],
            'attention': ['attention.query', 'attention.key', 'attention.value', 'output.dense'],
            'layer_scale1': [],
            'norm2': [],
            'mlp': ['fc1', 'fc2'],
            'layer_scale2': [],
        },
        'skip_connections': [
            ['norm1', 'layer_scale1'],
            ['norm2', 'layer_scale2'],
        ],
        'calibration_groups': {
            'attention': [['attention.query', 'attention.key', 'attention.value'], ['output.dense']],
            'mlp': [['fc1'], ['fc2']],
        },
        # Structured-pruning annotations (see llama_c for field semantics). Only the
        # dimensions internal to a block are prunable (MLP hidden, value -> output).
        # The residual (hidden) dimension is not: the classifier reads the
        # concatenation [CLS token, mean of the patch tokens], so its input columns
        # are not one-to-one with the hidden channels.
        'pruning': {
            'output_dependence': {
                'mlp.fc1': ['mlp.fc2'],
                'attention.attention.value': ['attention.output.dense'],
            },
            'coupled_masks': [
                ['attention.attention.query', 'attention.attention.key'],
            ],
            'coupled_masks_all': [],
            # Normalization and layer-scale layers: transparent to the hidden size,
            # never user-compressible.
            'norm_layers': ['norm1', 'norm2', 'layer_scale1', 'layer_scale2'],
        },
        'path_template': "dinov2.encoder.layer.{block_index}.{path}",
        'preprocessing': "dinov2.embeddings",
        'preprocessing_conv': "dinov2.embeddings.patch_embeddings.projection",
        'final_layer': "classifier",
        'final_norm': "dinov2.layernorm",

        # Transformersurgeon export topology specifics
        'structure': 'transformer_encoder',
        'attn_type': 'mha_encoder',
        'mlp_type': 'mlp',
        'norm_type': 'layernorm',
        'norm_position': 'pre',
        'layer_scale': True,
        'layer_matching': {
            'norm1': 'norm_in',
            'attention': ['attn.q_proj', 'attn.k_proj', 'attn.v_proj', 'attn.out_proj'],
            'layer_scale1': 'scale_attn',
            'norm2': 'norm_out',
            'mlp': ['mlp.up_proj', 'mlp.down_proj'],
            'layer_scale2': 'scale_mlp',
        },
        'extra_layers': ["dinov2.layernorm"],
        'extra_layers_matching': ["norm"],
        'use_final_norm': True,
        'position_embedding_type': 'none',
    },
}

__all__ = ["DINOV2_C_INDEXING"]
