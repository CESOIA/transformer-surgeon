# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
Model-specific configuration for DistilBERT model compression.
"""

# Configuration for DistilBERT model compression schemes
DISTILBERT_C_INDEXING = {
    "distilbert": {
        "config_attr": "",
        "num_blocks_attr": "n_layers",
        "path_list": [
            "attention.q_lin",
            "attention.k_lin",
            "attention.v_lin",
            "attention.out_lin",
            "ffn.lin1",
            "ffn.lin2",
        ],
        "path_template": "distilbert.transformer.layer.{block_index}.{path}",
        "qkv_paths": [
            "attention.q_lin",
            "attention.k_lin",
            "attention.v_lin",
        ],
    }
}

__all__ = ["DISTILBERT_C_INDEXING"]
