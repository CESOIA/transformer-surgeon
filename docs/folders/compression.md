# `compression/` — deciding and applying compression

Three layers, one direction:

| Layer | Class | Role |
|---|---|---|
| what | `Compressor` subclasses | how to compress one layer (`LRDer`, `StructuredPruner`, `BlockPruner`, `UnstructuredPruner`, `Quantizer`) |
| where | `CompressionScheme` | one layer of the model: its compressors, options, calibration data and **topology** |
| how many | `CompressionSchemesManager` | all schemes: filtering by criteria, groups, calibration, apply/restore |

| Module | Contents |
|---|---|
| `manager.py` | `CompressionSchemesManager`; `smooth_activations` (SmoothQuant) |
| `scheme.py` | `CompressionScheme`; `coupled_target_chains()` and `resolve_in_block()` resolve the model dictionary's names into modules |
| `grouping.py` | `SchemeGroup`: schemes sharing one pruning mask |
| `cascade.py` | cascade calibration: blocks calibrated one at a time on the partially compressed model |
| `config.py`, `compressible_layers.py` | default `compression_config`; replacement of listed layers by compressible ones at model init |
| `registry.py`, `validators.py` | options, defaults and validators of every compression type |
| `lrd.py`, `structured_pruning.py`, `coupled_pruning.py`, `block_pruning.py`, `unstructured_pruning.py`, `quantization.py`, `smoothquant.py` | the compressors |
| `*_methods/` | algorithm variants, one file each, listed in `METHOD_FUNCTIONS` |

```python
manager = LlamaCompressionSchemesManager(model)
for group in manager.auto_groups():                        # gate/up, q/k share masks
    manager.set("structured_pruning", "share_mask", True, group=group)
manager.set("structured_pruning", "ratio", 0.25, criteria=["mlp.gate_proj", "mlp.up_proj"])
manager.set("quantization", "precision", 8, criteria="mlp")
manager.set("quantization", "granularity", "per_channel", criteria="mlp")
manager.apply(hard=True)
```

Two rules keep this folder generic:

- compressors receive **resolved module chains** from the scheme and never read paths or sentinel names;
- only `quantization.py` reads the private state it leaves on quantised layers; the export asks for it through
  `export_quant_info(module)` and `copy_quant_state(old, new)`.

Every option and its valid values: [Compression Methods](../compression_methods.md).
API: [compression](../api/compression.md).
