"""Model-style indexing for converted custom decoder graph compression."""

# Configuration for converted decoder graph compression schemes
CUSTOM_DECODER_INDEXING = {
    "decoder": {
        "config_attr": "",
        "num_blocks_attr": "num_hidden_layers",
        "path_list": [
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.out_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        "path_template": "blocks.{block_index}.{path}",
    }
}

__all__ = ["CUSTOM_DECODER_INDEXING"]
