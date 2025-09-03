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
├── {model}_c/
│   ├── __init__.py
│   ├── configuration_{model}_c.py
│   ├── indexing_{model}_c.py
│   ├── manager_{model}_c.py
│   ├── modeling_{model}_c.py
│   └── NOTICE
└── ...
```

---

## Module Overview

### `utils/`
- **compression.py**: Defines `CompressionScheme`, the core class for compression algorithms. Used to configure and apply compression to individual layers.
- **manager.py**: Implements `CompressionSchemesManager`, a generic manager that orchestrates multiple `CompressionScheme` instances across a model. Handles applying/restoring all compression schemes.
- **linearLRD.py**: Provides `LinearLRD`, a drop-in replacement for `nn.Linear` supporting low-rank decomposition and pruning masks. Used in model definitions (modeling_*.py).
- **configuration.py**: Utilities for general configuration of the compression options in the transformer. Used by each model in the configuration_*.py files.
- **checks.py**: Helper functions to check validity of the options.

### `qwen2_vl_c/`
- **configuration_{model}_c_c.py**: Defines compression-aware config classes for the specific model. Extends HuggingFace configs to support compression.
- **indexing_{model}_c.py**: Contains indexing information (modules paths and keys) and information if each block supports compression.
- **manager_{model}_c.py**: Provides a compression manager specialized with the specific model's indexing.
- **modeling_{model}_c.py**: Specialization of the `modeling_{model}.py` model description with compressed blocks.


## Extending to Other Models

- The framework is designed to be generic: by providing a model-specific indexing (`indexing_{model}_c.py`), you can use the same compression logic for other transformer architectures.
- Additionally, `modeling_{model}.py` should be carefully modified to accomodate compressed structures.
