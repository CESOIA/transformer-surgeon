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
- MLP pruning and GQA-aware attention pruning (q/k head_dim via RoPE, v→o cascade) are both wired end-to-end (prune → convert → export).

## `blocks/` is a frozen model — never a compression target

`blocks/` (`mha.py`, `mlp.py`, `decoder.py`, `encoder.py`, `linear_compressed.py`, `pruning_dims.py`, `config.py`, `rope.py`, ...) defines the **converted, already-compressed** graph — the step right before `torch.export`/ONNX/backend lowering. It must never itself decide *what* to prune or compress:

- Compression/pruning is always decided beforehand, on the original HF `*_c` model, via `CompressionSchemesManager.apply()` (`utils/manager.py` + `compression/*`). `utils/convert.py::convert_for_export()` then builds the `blocks/` module tree and copies over the already-decided (possibly hard-pruned) weights/shapes/masks — it never re-derives a pruning decision.
- `blocks/pruning_dims.py`, `blocks/linear_compressed.py`, `blocks/config.py` are passive: they turn an already-decided ratio/rank/mask into static geometry (kept-dim sizing, `LinearCompressed` shapes, HF-style config plumbing), never choosing ratios or building masks themselves.
- Data-dependent pruning geometry that can only be known once weights are loaded (e.g. attention's RoPE frequency projection, `blocks/mha.py::MHABase.finalize_rope_pruning()`) is resolved **once, at conversion time** (`convert_for_export`, right after the relevant weights/buffers are copied) — never lazily inside `forward()`. `forward()`/`_project_rope` only ever do the static runtime application of an already-frozen decision.
- There used to be a `blocks/indexing.py` (`CUSTOM_DECODER_INDEXING`/`CUSTOM_ENCODER_INDEXING`) whose sole purpose was letting a `CompressionSchemesManager` be pointed at an already-converted `blocks/` model (`new_model.indexing = ...`) to compress it a second time, post-conversion. That workflow was removed — it's exactly the "compress a frozen model" anti-pattern this section forbids. `convert_for_export` still builds an equivalent indexing dict internally (inlined in `utils/convert.py`), but only to default-fill each layer's `compression_config` (`utils/configuration.py::init_compressed_config`, itself passive); it is not attached to the converted model.
- Do not add a path that lets a `blocks/` model be handed to `CompressionSchemesManager` or otherwise mutated after construction (besides the one-time `finalize_*` conversion-time hooks described above). If a new block type needs pruning-aware sizing, follow the `pruning_dims.py` pattern (or, for weight-dependent geometry, a `finalize_*` method called once from `convert_for_export`) — not a runtime check inside `forward()`.

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
