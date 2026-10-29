# CLAUDE.md — transformer-surgeon

Adds compression (LRD, pruning, quantization) to Hugging Face transformer models while keeping
`from_pretrained` / `generate` / `save_pretrained` intact, and exports the result to device backends
(XNNPACK, QNN, ONNX, TensorRT) through one generic pipeline.

**Read [AGENTS.md](AGENTS.md) before changing code**: folder rules, file map, test commands, parameter
reference, invariants, anti-patterns and extension recipes.

## Core loop

```python
manager.set("lrd", "rank", 128, criteria="mlp")   # configure (declarative)
manager.apply(hard=False)                          # compress (reversible)
model.generate(...)                                # use normally
manager.restore()                                  # undo
manager.apply(hard=True)                           # irreversible, before export
export_to_backend(model, XNNPACKExportConfig(output_path="model.pte"))
```

## Folder rules (binding)

| Folder | Holds | Never |
|---|---|---|
| `utils/` | generic module/path/batch helpers | model names, compression types, export |
| `models/` | per family: model dictionary `indexing_*.py` + family dictionary `family_*.py` | algorithms |
| `compression/` | manager, schemes (own the topology), compressors, registry, validators | model names, export details |
| `calibration/` | statistics collection and calibration data | compression decisions |
| `blocks/` | export-ready atoms, one per file, listed in `catalog.py::BLOCK_CATALOG` | model specifics, compression decisions |
| `export/` | conversion, generic `Exporter`, `backends/`, `operations.py::BACKEND_OPERATIONS` | compression decisions |

Adding a model touches only `models/`; a compression method only `compression/`; a calibration statistic
only `calibration/`; a backend only `export/` (plus new `blocks/` atoms when an operation is missing).

## Critical invariants

- `blocks/` is a frozen, already-compressed graph: `convert_for_export` copies shapes/ranks/biases from the
  source layers; weight-dependent geometry is resolved once at conversion, never in `forward()`.
- Compressors receive resolved module chains (`CompressionScheme.coupled_target_chains`), never paths.
- Only `compression/quantization.py` reads the private quantisation state (`export_quant_info`, `copy_quant_state`).
- Export implementation choices live in `BACKEND_OPERATIONS`, not in exporter code.
- `manager.set()` is declarative; `hard=True` is irreversible.
- `manager.prepare_for_save()` before `export_to_hf()` when applying manually; never before a backend export.
- After LRD, `weight` is `[out, rank]` and `linear_V.weight` is `[rank, in]`.

## Tests

```bash
pytest                                                   # test/ mirrors the package; tiny models, no downloads
```

Use the conda environments `py312_tensorrt`, `py310-executorch-qnn`, `py312_executorch_xnnpack`; backends are
capability-gated.
