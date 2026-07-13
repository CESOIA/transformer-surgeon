# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
Model-specific configuration for DistilBERT model compression.
"""

# Configuration for DistilBERT model compression schemes
DISTILBERT_C_INDEXING = {
    "distilbert": {
        # Attributes in the Hugging Face model config
        "config_attr": "",
        "num_blocks_attr": "n_layers",
        "embed_dim_attr": "dim",
        "num_heads_attr": "n_heads",
        "mlp_hidden_dim_attr": "hidden_dim",
        "mlp_activation_attr": "activation",

        # HF model structure specifics
        "path_list": {
            "attention": ["q_lin", "k_lin", "v_lin", "out_lin"],
            "sa_layer_norm": [],
            "ffn": ["lin1", "lin2"],
            "output_layer_norm": [],
        },
        "skip_connections": [
            ["attention", "attention"],
            ["ffn", "ffn"],
        ],
        "calibration_groups": {
            "attention": [["q_lin", "k_lin", "v_lin"], ["out_lin"]],
            "ffn": [["lin1"], ["lin2"]],
        },
        # Structured-pruning annotations (see llama_c for field semantics).
        "pruning": {
            "output_dependence": {
                "ffn.lin1": ["ffn.lin2"],
                "ffn.lin2": ["attention.q_lin", "attention.k_lin", "attention.v_lin"],
                "attention.v_lin": ["attention.out_lin"],
                "attention.out_lin": ["ffn.lin1"],
            },
            "coupled_masks": [
                ["attention.q_lin", "attention.k_lin"],
            ],
            # coupled_masks_all: share ONE output mask across ALL blocks (the
            # residual/hidden-dim writers).
            "coupled_masks_all": [
                ["attention.out_lin", "ffn.lin2"],
            ],
            "per_head_uniform": ["attention.q_lin", "attention.k_lin", "attention.v_lin"],
        },
        "path_template": "distilbert.transformer.layer.{block_index}.{path}",
        "qkv_paths": [
            "attention.q_lin",
            "attention.k_lin",
            "attention.v_lin",
        ],
        "preprocessing": "distilbert.embeddings",

        # Transformersurgeon export topology specifics
        "structure": "transformer_encoder",
        "attn_type": "mha_encoder",
        "mlp_type": "mlp",
        "norm_type": "layernorm",
        "norm_position": "post",
        "layer_matching": {
            "attention": ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.out_proj"],
            "sa_layer_norm": "norm_in",
            "ffn": ["mlp.up_proj", "mlp.down_proj"],
            "output_layer_norm": "norm_out",
        },
        "bias_required": {
            "attn.q_proj": True,
            "attn.k_proj": True,
            "attn.v_proj": True,
            "attn.out_proj": True,
            "norm_in": False,
            "mlp.up_proj": True,
            "mlp.down_proj": True,
            "norm_out": False,
        },
        "use_final_norm": False,
        "position_embedding_type": "none",
    }
}

__all__ = ["DISTILBERT_C_INDEXING"]
