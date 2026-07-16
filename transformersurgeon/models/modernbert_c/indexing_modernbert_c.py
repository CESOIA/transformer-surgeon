# Copyright 2025 The CESOIA project team, Politecnico di Torino and King Abdullah University of Science and Technology. All rights reserved.
"""
Model-specific configuration for ModernBERT model compression.
"""

# Configuration for ModernBERT model compression schemes
MODERNBERT_C_INDEXING = {
    'modernbert':
    {
        # Attributes in the Hugging Face model config
        'config_attr': '',
        'num_blocks_attr': 'num_hidden_layers',
        'embed_dim_attr': 'hidden_size',
        'num_heads_attr': 'num_attention_heads',
        'mlp_hidden_dim_attr': 'intermediate_size',
        'mlp_activation_attr': 'hidden_activation',

        # HF model structure specifics.
        # ModernBERT fuses attention Q/K/V into one projection (`Wqkv`) and the
        # MLP gate/value halves into one projection (`Wi`) instead of BERT's
        # separate query/key/value or Llama's separate gate_proj/up_proj. Layer 0
        # additionally replaces `attn_norm` with `nn.Identity()` (pre-norm
        # architecture, no norm before the first attention), which is harmless
        # here since only `nn.Linear` leaves are ever swapped for
        # `LinearCompressed` (see `utils/modeling.py::replace_layers_upon_init`).
        'path_list': {
            'attn_norm': [],
            'attn': ['Wqkv', 'Wo'],
            'mlp_norm': [],
            'mlp': ['Wi', 'Wo'],
        },
        'calibration_groups': {
            'attn': [['Wqkv'], ['Wo']],
            'mlp': [['Wi'], ['Wo']],
        },
        # Structured-pruning annotations (see llama_c for field semantics).
        # `Wqkv` and `Wi` are single fused matrices, not a pair of separate
        # Linear modules -- there is nothing to declare "coupled", the fusion
        # already lives inside one matrix, and the framework's coupled/shared
        # -mask pruning operates on whole Linear rows across modules. Coupled
        # structured pruning is therefore not wired for this family, mirroring
        # the fused `attn.qkv` vision tower in qwen2_vl_c (FRAMEWORK_PROBLEMS.md #6).
        'pruning': {
            'output_dependence': {},
            'coupled_masks': [],
            'coupled_masks_all': [],
            'per_head_uniform': [],
        },
        'pruning_supported': [],
        # Alternating global/local attention layers each need their own rotary
        # embedding call keyed by layer type
        # (`rotary_emb(hidden_states, position_ids, layer_type)`), which the
        # single block-wise cascade calibration flow in `utils/cascade.py`
        # (`_attach_position_embeddings_if_configured`) does not model -- it
        # calls the configured rotary module the same way for every block.
        # Cascade calibration is therefore unsupported here, same as bert_c
        # (see FRAMEWORK_PROBLEMS.md #1 and `no_cascade_calibration` there).
        'no_cascade_calibration': True,
        'path_template': "model.layers.{block_index}.{path}",
        'qkv_paths': ['attn.Wqkv'],
        'preprocessing': "model.embeddings",
        'final_layer': "classifier",

        # NOTE: no 'structure' key on purpose -- convert_for_export() /
        # export_to_backend() assume split q_proj/k_proj/v_proj projections for
        # the transformer_encoder converter, which ModernBERT's fused `Wqkv`
        # doesn't provide. Backend export is not wired for this family;
        # `convert_for_export` degrades gracefully (warns and skips) when
        # 'structure' is absent.
    },
}

__all__ = ["MODERNBERT_C_INDEXING"]
