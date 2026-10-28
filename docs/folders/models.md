# `models/` — model families as dictionaries

Everything that distinguishes one model from another lives here, as two dictionaries per family. Code
outside `models/` never branches on a model name.

| Family | Hugging Face class | Generated classes |
|---|---|---|
| `qwen2_c` | `Qwen2ForCausalLM` | `Qwen2ConfigCompress`, `Qwen2ForCausalLMCompress`, `Qwen2CompressionSchemesManager` |
| `llama_c` | `LlamaForCausalLM` | `Llama...Compress`, `LlamaCompressionSchemesManager` |
| `qwen2_vl_c`, `qwen2_5_vl_c` | `Qwen2(.5)VLForConditionalGeneration` | vision + text configurations, model, manager |
| `bert_c`, `distilbert_c`, `modernbert_c` | `...ForSequenceClassification` | configuration, model, manager |
| `vit_c` | `ViTForImageClassification` | configuration, model, manager |

## The family dictionary (`family_<m>_c.py`)

Read by `models/factory.py::build_family`, which generates the classes:

```python
LLAMA_C_FAMILY = {
    "indexing": LLAMA_C_INDEXING,                       # the model dictionary
    "hf_config": "transformers.models.llama.configuration_llama.LlamaConfig",
    "hf_model": "transformers.models.llama.modeling_llama.LlamaForCausalLM",
    "class_names": {"config": "LlamaConfigCompress", "model": "LlamaForCausalLMCompress",
                    "manager": "LlamaCompressionSchemesManager"},
    "auto_class": "AutoModelForCausalLM",               # recorded by export_to_hf
}
_classes = build_family(LLAMA_C_FAMILY, __name__)
```

Optional keys: `sub_configs` (components with their own configuration, e.g. vision and text towers) and
`checkpoint_key_mapping` (load checkpoints written with an older layer naming).

## The model dictionary (`indexing_<m>_c.py`)

One entry per component (e.g. `"text"`, `"vision"`). The main keys:

| Key | Meaning |
|---|---|
| `config_attr` | where the component's configuration lives in the HF config (`""` = root) |
| `num_blocks_attr`, `embed_dim_attr`, `num_heads_attr`, ... | names of the HF configuration attributes |
| `path_list` | per-block layers, grouped by sub-block (`{"self_attn": ["q_proj", ...], ...}`) |
| `path_template` | format string of a layer's path: `"model.layers.{block_index}.{path}"` |
| `preprocessing`, `final_layer`, `final_norm`, `extra_layers` | paths of the embedding, output head and layers outside the blocks |
| `pruning` | couplings for structured pruning: `output_dependence`, `coupled_masks`, `coupled_masks_all`, `norm_layers`, `position_linked`, `block_pruning` |
| `smoothing` | SmoothQuant groups (producer → consumers) |
| `calibration_groups`, `no_cascade_calibration` | scheduling of cascade calibration |
| `structure`, `attn_type`, `mlp_type`, `norm_type`, `rope_type` | **export**: which atoms of `BLOCK_CATALOG` build the export graph |
| `layer_matching`, `extra_layers_matching` | how HF layers map onto the atoms' layers |

Excerpt (Llama):

```python
'pruning': {
    'output_dependence': {
        'preprocessing': ['input_layernorm'],          # sentinel: the embedding
        'input_layernorm': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
        'mlp.gate_proj': ['mlp.down_proj'],
        'mlp.down_proj': ['input_layernorm', 'final_norm'],   # next block, or the final norm
        ...
    },
    'coupled_masks': [['self_attn.q_proj', 'self_attn.k_proj'], ['mlp.gate_proj', 'mlp.up_proj']],
    'norm_layers': ['input_layernorm', 'post_attention_layernorm'],
},
'structure': 'transformer_decoder',
'attn_type': 'mha_causal',
'mlp_type': 'mlp_gated',
'norm_type': 'rmsnorm',
'rope_type': 'auto',          # take the HF configuration's RoPE type
```

Adding a family: see [Recipes](../recipes.md#a-model-family).

API: [models](../api/models.md).
