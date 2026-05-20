# TransformerSurgeon Framework Structure

This document describes the current structure of the `transformersurgeon` package and how compression flows through it.

## Package Layout

```
transformersurgeon/
├── __init__.py
├── blocks/
│   ├── linear_compressed.py
│   ├── vcon_block.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── mha.py
│   ├── mlp.py
│   ├── norm.py
│   ├── rope.py
│   ├── config.py
│   └── indexing.py
├── compression/
│   ├── abstract.py
│   ├── registry.py
│   ├── lrd.py
│   ├── structured_pruning.py
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
│   ├── manager.py
│   ├── configuration.py
│   ├── modeling.py
│   ├── convert.py
│   └── utils.py
├── models/
│   ├── __init__.py
│   ├── qwen2_c/
│   ├── qwen2_vl_c/
│   ├── qwen2_5_vl_c/
│   ├── bert_c/
│   ├── distilbert_c/
│   └── vit_c/
└── export/
		├── hf_export.py
		└── executorch_export.py
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
	- Integrates calibration through `set_calibration_data` and `run_calibration`.

### 4) Configuration and model patching

- `utils/configuration.py`
	- `init_compressed_config(...)` injects and validates `compression_config` defaults.
- `utils/modeling.py`
	- `replace_layers_upon_init(...)` walks indexed paths and replaces `nn.Linear` with `LinearCompressed`.

### 5) Model adapters

`models/*_c/` contains architecture-specific glue code:

- `indexing_*.py`: path templates and model metadata
- `define_*.py`:
	- compressed config class (HF-compatible)
	- compressed model class
	- model-specific manager subclass

The manager itself stays generic; model-specific behavior lives in indexing.

### 6) Export pipeline

- `utils/convert.py`
	- `convert_for_export(...)` remaps supported models into compact custom encoder/decoder graph blocks.
	- Carries over weights and remapped compression config.
- `export/hf_export.py`
	- Saves/publishes compressed models to Hugging Face Hub.
- `export/executorch_export.py`
	- Export path for ExecuTorch, with adapter and quantization-plan scaffolding.

## End-to-End Data Flow

1. Load compressed model class from `models/*_c`.
2. During init, `replace_layers_upon_init(...)` swaps target linear layers to `LinearCompressed`.
3. Create manager (`CompressionSchemesManager` subclass).
4. Manager builds `CompressionScheme` objects from indexing + `compression_config`.
5. Use `manager.set(...)` with criteria to configure selected layers.
6. Optional: provide calibration dataloader and run/apply calibration-required methods.
7. Call `manager.apply(...)` to update weights/modules.
8. Optional: use VCON utilities for smooth transition training.
9. Optional: convert and export using `convert_for_export(...)` and export modules.

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
