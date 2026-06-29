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

## Critical invariants

- `Compressor.apply()` always receives a `LinearCompressed`, never a plain `nn.Linear`
- `manager.set()` is purely declarative; the model is only touched during `manager.apply()`
- `manager.prepare_for_save()` must be called before `export_to_hf()` when applying manually
- After LRD, `module.weight` shape changes to `[out, rank]` and `module.linear_V.weight` is `[rank, in]`
- Don't instantiate `CompressionScheme` directly — the manager builds schemes from indexing metadata
- Don't put model-specific logic in `CompressionSchemesManager` — it belongs in `indexing_*.py`

## Tests

```bash
pytest test/compression_tests/ -v          # core logic, no large models
pytest test/bert_tests/ -v                 # small encoder models
pytest test/distilbert_tests/ -v
pytest test/qwen_tests/inference_test.py -v   # requires model download
pytest test/hf_export_tests/ -v
```
