# CLAUDE.md — transformer-surgeon

Adds compression (LRD, pruning, quantization) to HuggingFace transformer models while keeping the standard `from_pretrained` / `generate` / `save_pretrained` workflow intact.

**Read [AGENTS.md](AGENTS.md) before making any code changes.** It has the file map, test commands, parameter reference, coding invariants, anti-patterns, and extension recipes.

## Core loop

```python
manager.set("lrd", "rank", 128, criteria="mlp")   # configure
manager.apply(hard=False)                           # compress (reversible)
model.generate(...)                                 # use normally
manager.restore()                                   # undo
```

`hard=True` is irreversible — only for final export.

## Structured pruning & grouping

```python
manager.auto_groups()                                    # build coupled-mask groups from indexing
manager.set("structured_pruning", "share_mask", True, group="group1")  # group-only option
manager.set("structured_pruning", "ratio", 0.1, criteria="mlp")
manager.apply(hard=True)                                 # removes neurons, resizes + cascades to next layer
```

- Hard pruning removes output rows and cascades the input pruning to coupled next
  layers (`compression/coupled_pruning.py`), driven by `pruning.output_dependence` in `indexing_*.py`.
- `share_mask`/`reduce_op`/`granularity`/`repeated_pattern` support grouped and per-head pruning; grouping + coupling logic lives in `compression/structured_pruning.py`, not the manager.
- Only MLP pruning is wired end-to-end (prune → convert → export); attention (GQA head_dim) is deferred.

## Critical invariants

- `Compressor.apply()` always receives a `LinearCompressed`, never a plain `nn.Linear`
- `manager.set()` is purely declarative; the model is only touched during `manager.apply()`
- `manager.prepare_for_save()` must be called before `export_to_hf()` when applying manually
- After LRD, `module.weight` shape changes to `[out, rank]` and `module.linear_V.weight` is `[rank, in]`
- Don't instantiate `CompressionScheme` directly — the manager builds schemes from indexing metadata
- Don't put model-specific logic in `CompressionSchemesManager` — it belongs in `indexing_*.py`

## Tests

```bash
pytest                                    # default: test/unit + test/e2e, no downloads
pytest test/unit                          # fast pure-logic + bug regressions
pytest test/e2e/test_model_families.py    # all 7 families, tiny models, no downloads
pytest test/e2e/test_export_pipelines.py  # HF/convert/XNNPACK/TensorRT/QNN, capability-gated
```

Legacy per-model scripts live under `test/test_deprecated/` (see its README) — kept
for reference only, not collected by default.

## Backend export

`export_to_backend(model, config)` lowers a model to `xnnpack`/`qnn` (ExecuTorch) or `tensorrt`. See [Export Backends in AGENTS.md](AGENTS.md#export-backends).
