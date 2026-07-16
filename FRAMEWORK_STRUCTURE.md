# TransformerSurgeon Framework Structure

This document describes the current structure of the `transformersurgeon` package and how compression flows through it.

## Package Layout

```
transformersurgeon/
├── __init__.py
├── blocks/
│   ├── linear_compressed.py
│   ├── pruning_dims.py          # effective_out_features: single source for ratio->kept dim
│   ├── vcon_block.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── mha.py
│   ├── mlp.py
│   ├── norm.py
│   ├── rope.py
│   └── config.py
├── compression/
│   ├── abstract.py
│   ├── registry.py
│   ├── lrd.py
│   ├── structured_pruning.py
│   ├── coupled_pruning.py       # prunes next-layer inputs (cascaded by structured_pruning)
│   ├── unstructured_pruning.py
│   ├── quantization.py
│   ├── lrd_methods/
│   ├── structured_pruning_methods/
│   ├── unstructured_pruning_methods/
│   └── quantization_methods/
├── calibration/
│   ├── backbone.py
│   ├── raw_data/
│   └── summaries/
├── utils/
│   ├── scheme.py
│   ├── grouping.py              # SchemeGroup: shared-mask groups for pruning
│   ├── manager.py
│   ├── configuration.py
│   ├── modeling.py
│   ├── convert.py
│   └── utils.py
├── models/
│   ├── __init__.py
│   ├── qwen2_c/
│   ├── llama_c/
│   ├── qwen2_vl_c/
│   ├── qwen2_5_vl_c/
│   ├── bert_c/
│   ├── modernbert_c/
│   ├── distilbert_c/
│   └── vit_c/
├── hf/
│   └── hf_export.py
└── export/
		├── export.py                 # export_to_backend() dispatcher
		├── registry.py                # EXPORT_ROUTINES: xnnpack, qnn, tensorrt
		├── config.py                  # BackendExportConfig (shared base)
		├── common.py                  # shared quant-metadata / PT2E / calibration machinery
		├── executorch_exporters/
		│   ├── common.py
		│   ├── xnnpack/
		│   └── qnn/
		└── tensorrt/
```

## Core Concepts

### 1) Compression-ready layers

- `blocks/linear_compressed.py`
	- Defines `LinearCompressed`, a drop-in replacement for `torch.nn.Linear`.
	- Supports low-rank mode with `weight` + `weight_2` factorization.
- `blocks/vcon_block.py`
	- Defines `VCONBlock` that blends two modules with a scalar `beta`.

These are the primitive modules modified by compression schemes.

### 2) Compression algorithms

- `compression/registry.py`
	- Central registry of compressors and configurable properties.
	- Maps names to classes: `lrd`, `structured_pruning`, `unstructured_pruning`, `quantization`.
- `compression/*.py`
	- Each compressor class validates config, decides calibration needs, applies updates, and defines restore behavior.
- `compression/*_methods/`
	- Method-specific implementations (for example SVD variants, magnitude/random pruning).

### 3) Per-layer scheme + model-level manager

- `utils/scheme.py` (`CompressionScheme`)
	- Binds one model path (for example one projection layer) to one compression config.
	- Handles module lookup, optional VCON wrapping, apply/restore.
- `utils/manager.py` (`CompressionSchemesManager`)
	- Builds all schemes from model indexing metadata.
	- Provides filtering by criteria and batch operations:
		- `set(...)`
		- `apply(...)`
		- `restore(...)`
		- VCON helpers (`init_vcon`, `set_vcon_beta`, `cancel_vcon`)
	- Integrates calibration through:
		- `set_calibration_data(...)`
		- `set_calibration_mode(mode="standard"|"cascade")`
		- `run_calibration(...)`
	- In cascade mode, consumes indexing-provided `calibration_groups`.

### 4) Configuration and model patching

- `utils/configuration.py`
	- `init_compressed_config(...)` injects and validates `compression_config` defaults.
- `utils/modeling.py`
	- `replace_layers_upon_init(...)` walks indexed paths and replaces `nn.Linear` with `LinearCompressed`.

### 5) Model adapters

`models/*_c/` contains architecture-specific glue code:

- `indexing_*.py`: path templates, model metadata, and optional cascade calibration scheduling metadata (`calibration_groups`)
- `define_*.py`:
	- compressed config class (HF-compatible)
	- compressed model class
	- model-specific manager subclass

The manager itself stays generic; model-specific behavior lives in indexing.

### 5.1) Cascade calibration groups in indexing

Each indexing block can define explicit parallel calibration groups:

```python
"calibration_groups": [
	["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
	["mlp.gate_proj", "mlp.up_proj"],
]
```

Groups are interpreted with the same matching semantics used by manager filtering (substring criteria over layer paths). Ungrouped layers are calibrated as singleton stages.

`calibration_groups` may also be given as a dict keyed by subblock name (each value a list of layer groups, with subblock-relative names auto-qualified) — see `models/qwen2_c/indexing_qwen2_c.py`. Both forms are parsed by `CompressionSchemesManager._get_calibration_groups_from_indexing`.

An indexing block can opt entirely out of cascade calibration with `'no_cascade_calibration': True` — `apply_cascade` (`utils/cascade.py`) raises a `ValueError` up front if cascade mode is requested with any scheme selected from that block. `bert_c` and `modernbert_c` set this: BERT's grouped/bidirectional QKV layout and ModernBERT's per-layer-type rotary embeddings (alternating local/global attention) aren't modeled by the single-flow block-wise cascade algorithm. Use `"standard"` calibration mode for those families.

### 6) Export pipeline

- `utils/convert.py`
	- `convert_for_export(...)` remaps supported models into compact custom encoder/decoder graph blocks.
	- Carries over weights and remapped compression config.
- `hf/hf_export.py`
	- Saves/publishes compressed models to Hugging Face Hub (`export_to_hf`).
- `export/export.py`
	- `export_to_backend(model_or_graph, config)` — the generic backend-export entry point. Resolves the model into `{embedding, decoder, final_layer, config}`, then delegates to the backend named in `config.backend` via `export/registry.py`'s `EXPORT_ROUTINES`. `export_to_executorch(...)` is a deprecated alias.
- `export/common.py`
	- Backend-agnostic machinery shared by every backend exporter: model wrapper construction, per-layer compression-metadata extraction (`extract_layer_quant_info`), PT2E scale injection/calibration, weight-mismatch checks, and result assembly (`finalize_export_result`).
- `export/executorch_exporters/{xnnpack,qnn}/`
	- ExecuTorch backend exporters (`XNNPACKExportConfig`/`export_with_xnnpack`, `QNNExportConfig`/`export_with_qnn`), each producing a `.pte` file. Mixed INT8/INT4 + float export is driven entirely by per-layer compression metadata already on the model.
- `export/tensorrt/`
	- TensorRT backend exporter (`TensorRTExportConfig`/`export_with_tensorrt`), lowering via `torch-tensorrt`'s Dynamo path to a TensorRT engine/exported program. Reuses `export/common.py` for everything except the quantizer and compile/save step. Requires the `tensorrt` extra (`torch-tensorrt`) and a CUDA device; tests under `test/e2e/test_export_pipelines.py`, CLI runner at `scripts/tensorrt/run_export.sh`.

## End-to-End Data Flow

1. Load compressed model class from `models/*_c`.
2. During init, `replace_layers_upon_init(...)` swaps target linear layers to `LinearCompressed`.
3. Create manager (`CompressionSchemesManager` subclass).
4. Manager builds `CompressionScheme` objects from indexing + `compression_config`.
5. Use `manager.set(...)` with criteria to configure selected layers.
6. Optional: provide calibration dataloader.
7. Optional: choose calibration scheduling with `set_calibration_mode(...)`.
8. Run `manager.apply(...)` (or `run_calibration(...)`) for calibration-required methods.
9. During shifted-summary calibration, backbone can report shifted-activation sanity metrics (`pairs`, `mean_rel_l2_diff`, `max_rel_l2_diff`) in verbose mode.
10. Optional: use VCON utilities for smooth transition training.
11. Optional: convert and export using `convert_for_export(...)` and export modules.

## Extending to a New Architecture

To add a new model family:

1. Add `models/newmodel_c/indexing_newmodel_c.py` with:
	 - block count attribute key
	 - layer path list
	 - path template
	 - optional export structure metadata
2. Add `models/newmodel_c/define_newmodel_c.py` with:
	 - compressed config class using `init_compressed_config`
	 - compressed model class replacing layers at init
	 - manager subclass calling `super().__init__(model, INDEXING)`
3. Export symbols in `models/newmodel_c/__init__.py` and `models/__init__.py`.

No new manager logic is required unless the architecture needs custom behavior beyond indexing.
