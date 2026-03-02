"""
Model-specific configuration for Qwen2.5-VL-C model compression.
"""

# Configuration for Qwen2.5-VL-C model compression schemes
QWEN2_5_VL_IMP_INDEXING = {
    'vision':
    {
        'config_attr': 'vision_config',
        'num_blocks_attr': 'depth',
        'path_list': ["", "norm2"],
        'path_template': "model.visual.blocks.{block_index}.{path}"
    },
    'text':
    {
        'config_attr': 'text_config',
        'num_blocks_attr': 'num_hidden_layers',
        'path_list': ["", "post_attention_layernorm"],
        'path_template': "model.language_model.layers.{block_index}.{path}"
    }
}
