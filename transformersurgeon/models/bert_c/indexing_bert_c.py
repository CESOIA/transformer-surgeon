# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
Model-specific configuration for Bert model compression.
"""

# Configuration for Bert model compression schemes
BERT_C_INDEXING = {
    'bert':
    {
        'config_attr': '',
        'num_blocks_attr': 'num_hidden_layers',
        'path_list': ["attention.self.query",
                      "attention.self.key",
                      "attention.self.value",
                      "attention.output.dense",
                      "intermediate.dense",
                      "output.dense"],
        'path_template': "bert.encoder.layer.{block_index}.{path}",
        'qkv_paths': [],
    },
}

