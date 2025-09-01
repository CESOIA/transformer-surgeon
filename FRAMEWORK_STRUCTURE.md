# TransformerSurgeon Framework Structure

This document provides an overview of the main modules, their purpose, and how they interact in the TransformerSurgeon framework.

---

## Top-Level Layout

```
transformersurgeon/
├── __init__.py
├── utils/
│   ├── __init__.py
│   ├── checks.py
│   ├── compression.py
│   ├── configuration.py
│   ├── linearLRD.py
│   ├── manager.py
│   └── model_config_template.py
├── qwen2_vl_c/
│   ├── __init__.py
│   ├── configuration_qwen2_vl_c.py
│   ├── manager_config.py
│   ├── manager_qwen2_vl_c.py
│   ├── modeling_qwen2_vl_c.py
│   └── NOTICE
└── ...
```

---

## Module Overview

### `utils/`
- **compression.py**: Defines `CompressionScheme`, the core class for layer-wise pruning and low-rank decomposition (LRD). Used to configure and apply compression to individual layers.
- **manager.py**: Implements `CompressionSchemesManager`, a generic manager that orchestrates multiple `CompressionScheme` instances across a model. Handles applying/restoring all compression schemes.
- **linearLRD.py**: Provides `LinearLRD`, a drop-in replacement for `nn.Linear` supporting low-rank decomposition and pruning masks. Used in model definitions (modeling_*.py).
- **configuration.py**: Utilities for general configuration of the compression options in the transformer. Used by each model in the configuration_*.py files.
- **checks.py**: Helper functions to check validity of the options.
- **model_config_template.py**: Template for model-specific configuration of the compression manager and model configuration.

### `qwen2_vl_c/`
- **configuration_qwen2_vl_c.py**: Defines compression-aware config classes for Qwen2-VL (vision and text blocks). Extends HuggingFace configs to support pruning and LRD.
- **manager_config.py**: Contains block-level configuration for Qwen2-VL compression and compression manager (e.g., which layers/blocks can be compressed, key mappings, etc.).
- **manager_qwen2_vl_c.py**: Provides a Qwen2-VL specific manager, wrapping the generic `CompressionSchemesManager` with model-specific config.
- **modeling_qwen2_vl_c.py**: Main Qwen2-VL model implementation, patched to use `LinearLRD` and compression-aware configs. Integrates vision/text blocks, compression logic, and custom attention/MLP layers.
- **NOTICE**: Licensing and attribution.

---

## How Components Are Used

- **CompressionScheme** (`utils/compression.py`):
  - Used by `CompressionSchemesManager` to represent and apply compression to individual layers.
  - Stores pruning ratio, LRD rank, and module reference.

- **CompressionSchemesManager** (`utils/manager.py`):
  - Instantiated in model-specific managers (e.g., `manager_qwen2_vl_c.py`).
  - Applies/restores all compression schemes to the model.

- **LinearLRD** (`utils/linearLRD.py`):
  - Used in model definitions (e.g., `modeling_qwen2_vl_c.py`) to replace standard linear layers with compression-aware layers.
  - Supports both pruning and low-rank decomposition.

- **Qwen2VL Model Classes** (`qwen2_vl_c/modeling_qwen2_vl_c.py`):
  - Use `LinearLRD` for all linear layers in attention and MLP blocks.
  - Accept compression configs and apply them via the manager.

- **Configuration Classes** (`qwen2_vl_c/configuration_qwen2_vl_c.py`):
  - Extend HuggingFace configs to support compression parameters.
  - Used to initialize models with compression-aware settings.

- **Manager Configs** (`qwen2_vl_c/manager_config.py`):
  - Define which blocks/layers are compressible and how to map config keys to model paths.
  - Used by managers to generate and apply compression schemes.

---

## Extending to Other Models

- The framework is designed to be generic: by providing a model-specific config (see `model_config_template.py`), you can use the same compression logic for other transformer architectures.
- Only the block/layer mapping and config need to be adapted for new models.
