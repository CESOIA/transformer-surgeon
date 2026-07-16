# AGENTS.md — transformer-surgeon

Agent-oriented guide for working on this codebase. For user-facing docs read [README.md](README.md). For architecture depth read [docs/concepts.md](docs/concepts.md).

---

## Project Summary

transformer-surgeon adds compression (LRD, pruning, quantization) to HuggingFace transformer models without breaking the standard `from_pretrained` / `generate` / `save_pretrained` workflow. The entry points are model-specific subclasses (e.g. `Qwen2ForCausalLMCompress`) and a manager (`Qwen2CompressionSchemesManager`). The core loop is always:

```python
manager.set("lrd", "rank", 128, criteria="mlp")   # configure
manager.apply(hard=False)                           # compress (reversible)
model.generate(...)                                 # use normally
manager.restore()                                   # undo
```

`hard=True` is irreversible and used only for final export.

---

## Dev Setup and Test Commands

```bash
git clone https://github.com/CESOIA/transformer-surgeon.git
cd transformer-surgeon
pip install -e ".[dev]"
```

Run tests:

```bash
# Default: test/unit + test/e2e/test_model_families.py — no downloads, no GPU.
# All 7 families (qwen2, llama, bert, modernbert, distilbert, vit, qwen2_vl,
# qwen2_5_vl) via tiny random-weight models from test/_helpers/model_factory.py.
pytest

# Bug regressions (pinned assertions for previously-broken framework behavior)
pytest test/unit/test_known_bugs.py -v

# Real-checkpoint export pipelines — HF roundtrip, convert, XNNPACK, TensorRT, QNN.
# Each backend is gated by test/_helpers/capabilities.py (skips if unavailable).
pytest test/e2e/test_export_pipelines.py -v

# Single file / test
pytest test/e2e/test_model_families.py::test_lrd_soft_apply_and_restore -v
```

Legacy per-model CLI scripts (pre-dating this hardened suite) live under
`test/test_deprecated/` — kept for reference only, not collected by default
(`pytest.ini` sets `testpaths = test/unit test/e2e`). See its README for why
they were retired.

---

## Architecture in 30 Seconds

Three layers, one direction:

| Layer | Class | Role |
|---|---|---|
| **What** | `Compressor` subclass | Algorithm: how to compress one `LinearCompressed` layer |
| **Where** | `CompressionScheme` | Binds a model path to a compressor config (built by manager, never instantiated directly) |
| **How many** | `CompressionSchemesManager` | Iterates all schemes, filters by criteria, runs calibration, calls apply/restore |

See [docs/concepts.md](docs/concepts.md) for the full lifecycle walkthrough.

---

## File Map

Where to look when you need to change something:

| Task | File |
|---|---|
| Change a compression algorithm | `transformersurgeon/compression/lrd.py`, `structured_pruning.py`, `unstructured_pruning.py`, `quantization.py` |
| Add a new compression method | Subclass `compression/abstract.py` → new file in `compression/` → register in `compression/registry.py` |
| Add an LRD variant (e.g. new SVD method) | New file in `compression/lrd_methods/`, import in `compression/lrd.py` |
| Add a pruning variant | New file in `compression/structured_pruning_methods/` or `unstructured_pruning_methods/` |
| Change structured-pruning masks / scoring / effective dims | `compression/structured_pruning.py`, `compression/structured_pruning_methods/`, `blocks/pruning_dims.py` |
| Change cross-layer input pruning (coupling) | `compression/coupled_pruning.py` (invoked in cascade by the structured pruner) |
| Change scheme grouping (shared masks) | `utils/grouping.py` (`SchemeGroup`) + `utils/manager.py` (`create_group`/`delete_group`/`auto_groups`) |
| Add a new model family | `models/newmodel_c/indexing_newmodel_c.py` + `models/newmodel_c/define_newmodel_c.py` |
| Change what layers get compressed / pruning coupling in a model | `models/qwen2_c/indexing_qwen2_c.py` (or the relevant model's `indexing_*.py`) — `path_list` and the `pruning` block |
| Change layer filtering / criteria logic | `utils/manager.py` → `iter_filtered()` |
| Change calibration data collection hooks | `calibration/raw_data/activation.py` or `weight_grad.py` |
| Add a new calibration summary (statistic) | Subclass `calibration/summaries/base.py` → new file → register in `calibration/summaries/registry.py` |
| Change how covariance is computed | `calibration/summaries/covariance.py` |
| Change the `LinearCompressed` forward pass | `blocks/linear_compressed.py` |
| Change VCON blending | `blocks/vcon_block.py` |
| Change HuggingFace export | `hf/hf_export.py` |
| Change export graph conversion | `utils/convert.py` |
| Change what parameters are valid for a compression type | `compression/registry.py` → `COMPRESSION_REGISTRY` |
| Change backend-export machinery shared by all backends (quant-metadata extraction, PT2E calibration, weight-mismatch checks) | `export/common.py` |
| Add/change a backend exporter | `export/registry.py` → `EXPORT_ROUTINES`, plus the backend's own subpackage (`export/executorch_exporters/xnnpack/`, `export/executorch_exporters/qnn/`, `export/tensorrt/`) |

---

## Export Backends

`transformersurgeon/export/` lowers a (possibly compressed) model to a deployment backend via `export_to_backend(model_or_graph, config)` (`export/export.py`). It dispatches through `EXPORT_ROUTINES` (`export/registry.py`) to one of:

| Backend | Config class | Module | Output |
|---|---|---|---|
| `xnnpack` | `XNNPACKExportConfig` | `export/executorch_exporters/xnnpack/` | ExecuTorch `.pte` |
| `qnn` | `QNNExportConfig` | `export/executorch_exporters/qnn/` | ExecuTorch `.pte` (Qualcomm NPU) |
| `tensorrt` | `TensorRTExportConfig` | `export/tensorrt/` | TensorRT engine / exported program (`.engine_path`) |

All three share the backend-agnostic machinery in `export/common.py`: `resolve_components_and_wrapper()` builds the model wrapper and example inputs, `extract_layer_quant_info()` reads per-layer compression metadata straight off the model (no separate quant config needed), and `inject_scales_into_pt2e_observers()` overrides PT2E-calibrated observers with the exact surgeon scales before `convert_pt2e()`. This is what makes **mixed-precision export** work: a model with some `LinearCompressed` layers hard-quantized to INT8/INT4 and others left float exports to a single engine/program with only the quantized layers getting Q/DQ ops.

```python
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.tensorrt import TensorRTExportConfig

config = TensorRTExportConfig(output_path="model.pt2", backend="tensorrt", device="cuda:0")
result = export_to_backend(model, config=config)   # model can be a full HF model or {embedding, decoder, final_layer}
print(result.engine_path)
```

Device placement is normalized internally (`resolve_components_and_wrapper` traces on CPU regardless of the input model's device; TensorRT then compiles the traced graph onto `config.device`), so callers don't need to manage component devices themselves.

`export_to_executorch(...)` is a deprecated alias for `export_to_backend(...)` — use `export_to_backend`.

TensorRT requires the `tensorrt` extra (`pip install -e ".[tensorrt]"`) plus a CUDA device. Tests live in `test/e2e/test_export_pipelines.py` (capability-gated, skips without torch-tensorrt/CUDA); the CLI runner is `scripts/tensorrt/run_export.sh` (mirroring `scripts/executorch/{xnnpack,qnn}/`).

---

## Compression Parameter Reference

All parameters are set via `manager.set(compression_type, param, value, criteria=...)`.
This is the terse lookup table; for a plain-language explanation of what each
method/parameter actually does, see [docs/compression_methods.md](docs/compression_methods.md).

### `"lrd"` — Low-Rank Decomposition

| param | default | valid values |
|---|---|---|
| `rank` | `"full"` | `int` (1 to `in_features`), or `"full"` |
| `method` | `"svd"` | `"svd"`, `"svd-llm-v2"`, `"aa-svd"` |
| `eps` | `1e-6` | `float > 0` |

Calibration requirements by method:

| method | calibration needed | mode |
|---|---|---|
| `"svd"` | none | any |
| `"svd-llm-v2"` | `"covariance"` | `"standard"` |
| `"aa-svd"` | `"cross_covariance"` + `"shifted_covariance"` | `"cascade"` only |

### `"structured_pruning"` — Output Neuron Removal

| param | default | valid values |
|---|---|---|
| `ratio` | `0.0` | `float` in `[0, 1)` |
| `method` | `"magnitude"` | `"magnitude"`, `"gradient"`, `"random"` |
| `granularity` | `"layer"` | `"layer"`, or a positive `int` (chunk/head size) |
| `repeated_pattern` | `False` | `bool` — one mask per chunk, tiled across chunks |
| `coupled_repeated_pattern` | `False` | `False`, or a positive `int` N — repeat each length-`granularity` chunk of the mask N times when cascading onto coupled next layers |
| `reduce_op` | `None` | `None`, `"add"`, `"multiply"` |
| `share_mask` | `False` | `bool` — **group-only** (set via `group=`) |

`"gradient"` requires the `"weight_grad"` calibration summary (needs a loss
callback: `manager.set_calibration_loss(...)`).

- **Soft** (`hard=False`) zeroes pruned output rows in place (reversible). **Hard**
  (`hard=True`) actually removes the rows, resizes `weight`/`bias`/`out_features`,
  and **cascades** the removal into the input columns of the coupled next layers
  (from the model's `pruning.output_dependence` indexing) via `CoupledPruner`.
- `granularity=g` prunes the same count within each consecutive chunk of `g`
  neurons (e.g. per attention head). `repeated_pattern=True` reduces scores across
  those chunks (`reduce_op`) into one length-`g` mask that is tiled back — this is
  what lets GQA `q_proj`/`k_proj` (different head counts) share one mask.
- `coupled_repeated_pattern=N` changes only the mask *cascaded onto coupled next
  layers* (hard apply): each length-`g` chunk of this layer's own keep-mask is
  repeated `N` times in place (`chunk chunk ... | next_chunk next_chunk ...`)
  before being used to prune the coupled layer's input columns, for a coupled
  layer whose input is `N`x this layer's own (pruned) output width. E.g. mask
  `[0,1,1,0,0,1,0,1]` with `granularity=4`, `coupled_repeated_pattern=2` cascades
  as `[0,1,1,0, 0,1,1,0, 0,1,0,1, 0,1,0,1]`. This layer's own output rows are
  still pruned by the unexpanded mask; only the downstream cascade changes.
- Effective kept dim is single-sourced in `blocks/pruning_dims.py`
  (`effective_out_features`), reused by the pruner, coupled pruning, and the
  converted MLP blocks so a hard-pruned model converts/exports with matching shapes.

#### Grouped structured pruning (shared masks)

Layers that must be pruned identically (same output mask) are expressed in model
indexing and turned into groups by the manager:

```python
manager = Qwen2CompressionSchemesManager(model)
groups = manager.auto_groups()                 # reads pruning.coupled_masks[_all]
for g in groups:                               # e.g. per-block gate/up, q/k
    manager.set("structured_pruning", "share_mask", True, group=g)   # group-only
    manager.set("structured_pruning", "reduce_op", "add",  group=g)
manager.set("structured_pruning", "method", "random", criteria=None)
manager.set("structured_pruning", "ratio", 0.1, criteria="mlp.gate_proj")
manager.set("structured_pruning", "ratio", 0.1, criteria="mlp.up_proj")
manager.apply(hard=True)
```

Rules: `share_mask` (and any `GROUP_OPTIONS`) can only be set through `group=`
(not `criteria`), and enabling one resets that scheme's non-group config. Grouping
and the coupled cascade live entirely in `compression/structured_pruning.py` (the
first compressor in a group computes the shared mask; siblings reuse it) — the
manager only builds groups and iterates `scheme.apply`.

Indexing annotations (`models/*/indexing_*.py`, under a `pruning` key):
`output_dependence` (coupling targets), `coupled_masks` (share a mask within a
block), `coupled_masks_all` (share a mask across all blocks — the residual/hidden
writers), `per_head_uniform` (recorded only).

Scope: both MLP and attention (q/k/v) structured pruning are wired end-to-end
(prune → cascade → convert → export). Attention hard pruning changes `head_dim`
(GQA); RoPE projection geometry and the pruned KV-cache are resolved per-kv-group
at conversion time (`blocks/mha.py::MHABase.finalize_rope_pruning()`), and a
hard-pruned `v_proj` cascades into `o_proj` via `coupled_repeated_pattern`. See
`test/e2e/test_gqa_attention_pruning.py`.

### `"unstructured_pruning"` — Weight-Level Sparsity

| param | default | valid values |
|---|---|---|
| `ratio` | `0.0` | `float` in `[0, 1)` |
| `method` | `"magnitude"` | `"magnitude"`, `"gradient"`, `"random"` |
| `granularity` | `"layer"` | `"layer"` |

Pruning masks survive `restore()` for STE fine-tuning. Call `manager.remove_masks()` to drop them.

### `"quantization"` — Fixed-Point and Binary Weights

| param | default | valid values |
|---|---|---|
| `method` | `"vanilla"` | `"vanilla"`, `"gptq"` |
| `precision` | `"full"` | `"full"`, `"binary"`, or `int` in `[2, 16]` (e.g. `8`, `4`, `2` — NOT the strings `"int8"`/`"int4"`/`"int2"`) |
| `granularity` | `"per_tensor"` | `"per_tensor"`, `"per_channel"` |
| `sparsity` | `0.0` | `float` in `[0, 1)` |
| `sparse_method` | `"magnitude"` | `"magnitude"`, `"random"` |
| `precision_activation` | `"full"` | same as `precision` |
| `method_activation` | `"maxmin"` | `"maxmin"` |
| `scheme_activation` | `"asymmetric"` | `"symmetric"`, `"asymmetric"` |
| `eps` | `1e-6` | `float > 0` |

---

## Criteria Language

Passed to `manager.set()`, `manager.apply()`, `manager.restore()`, etc.:

| criteria | matches | example |
|---|---|---|
| `None` or `"all"` | every scheme | `manager.set("lrd", "rank", 64)` |
| `int` | all layers in that block index | `criteria=2` |
| `str` | layers whose path contains the substring | `criteria="mlp"` |
| `[[str, int, ...]]` | AND of all items in the inner list | `criteria=[["mlp", 5]]` → block 5 AND "mlp" |
| `[str, int, ...]` | OR across items | `criteria=["q_proj", 3]` → "q_proj" OR block 3 |

---

## Coding Invariants

Things the codebase silently relies on. Breaking these causes silent wrong behavior or hard-to-trace errors:

1. **`Compressor.apply()` always receives a `LinearCompressed`**, never a plain `nn.Linear`. Target layers are replaced at model init time by `replace_layers_upon_init()` in `utils/modeling.py`.

2. **`manager.set()` is purely declarative** — it writes config but never touches the model. The model is only modified during `manager.apply()`.

3. **Soft apply (`hard=False`) is always reversible**; hard apply is permanent. Once `hard=True` is used, `manager.restore()` cannot undo the compression.

4. **`init_vcon()` must be called before `apply()`** when using VCON. Reversing the order raises an error because the scheme wrapping needs to happen before compression is applied to the secondary block.

5. **`manager.prepare_for_save()` must be called before `export_to_hf()` when applying manually** — it strips runtime quantization artifacts. Passing `manager=manager` to `export_to_hf()` triggers this automatically.

6. **`calibration_store` is a plain `dict`** owned by each `CompressionScheme`. `CalibrationSummary` implementations write to it; `Compressor` implementations read from it. Don't write to it from anywhere else.

7. **`INDEXING["path_template"]` must be a Python format string** with `{block_index}` and `{path}` — the manager calls `.format(block_index=..., path=...)` to resolve full module paths. Missing either placeholder breaks all scheme lookups.

---

## Anti-Patterns

- **Don't instantiate `CompressionScheme` directly.** The manager builds schemes from indexing metadata. Direct instantiation bypasses path resolution and calibration group assignment.

- **Don't assume `module.weight.shape` is `[out_features, in_features]` after LRD.** After `init_lrd(rank)`, `weight` becomes `[out_features, rank]` and `module.linear_V.weight` is `[rank, in_features]`. Check `module.rank != "full"` or `module.linear_V is not None` before accessing weight shape.

- **Don't call `hard=True` then `restore()`.** Hard apply is irreversible. `restore()` will appear to succeed but produce wrong shapes or no-op silently.

- **Don't put model-specific logic in `CompressionSchemesManager`.** The manager is generic. All model-specific information belongs in the model's `indexing_*.py` file.

- **Don't use `"standard"` calibration mode with `"aa-svd"`.** AA-SVD requires cross-layer shifted activations, which need staged (cascade) passes. Set `manager.set_calibration_mode("cascade")` before calling `apply()`.

- **Don't use `"cascade"` calibration mode with a family indexed `'no_cascade_calibration': True`** (currently `bert_c`, `modernbert_c`). `apply()` raises `ValueError` immediately — these families' layer layout (grouped/bidirectional QKV, or ModernBERT's per-layer-type rotary embeddings) isn't modeled by the single-flow block-wise cascade in `utils/cascade.py`. Use `"standard"` mode (`"svd"`/`"svd-llm-v2"` LRD, or non-AA-SVD compressors) instead.

- **Don't call `reapply_masks()` before `manager.apply()`.** Masks are created during `apply()`. `reapply_masks()` is for re-applying existing masks after an optimizer step during STE fine-tuning.

---

## Extension Recipes

### Adding a New Compression Method

1. Subclass `Compressor` from `transformersurgeon/compression/abstract.py`. Implement all six abstract methods:
   - `set_calibration_store(calibration_data: dict)` — store the per-scheme result dict reference
   - `needs_calibration() -> tuple[str, ...]` — return required summary names (empty tuple if none)
   - `apply(module: LinearCompressed, hard: bool, soft_applied: bool)` — compress the module
   - `restore(module: LinearCompressed)` — undo compression
   - `_to_compress() -> bool` — guard against no-op (e.g., rank == "full")
   - `__repr__() -> str` — human-readable config string for printing

2. Place the class in a new file, e.g. `transformersurgeon/compression/mymethod.py`.

3. Register it in `transformersurgeon/compression/registry.py`:
   ```python
   # In COMPRESSOR_DICT:
   "mymethod": MyMethodCompressor,

   # In COMPRESSION_REGISTRY:
   "mymethod": {
       "param_name": {"default": default_val, "validator": lambda v: isinstance(v, int)},
       ...
   }
   ```

4. (Optional) If the method needs a new calibration summary, follow the **Adding a Calibration Summary** steps below and reference its name in `needs_calibration()`.

### Adding a New Model Family

1. Create `transformersurgeon/models/newmodel_c/indexing_newmodel_c.py` with an `INDEXING` dict. Use `models/qwen2_c/indexing_qwen2_c.py` as a template. Required keys: `num_blocks_attr`, `path_list`, `path_template`. Optional but recommended: `calibration_groups`, `skip_connections`, export metadata (`structure`, `attn_type`, `mlp_type`).

2. Create `transformersurgeon/models/newmodel_c/define_newmodel_c.py` with three classes (exact pattern — do not deviate):
   ```python
   from transformersurgeon.utils.configuration import init_compressed_config
   from transformersurgeon.utils.modeling import replace_layers_upon_init
   from transformersurgeon.utils.manager import CompressionSchemesManager
   from .indexing_newmodel_c import INDEXING

   class NewModelConfigCompress(NewModelConfig):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           init_compressed_config(self)

   class NewModelForTaskCompress(NewModelForTask):
       config_class = NewModelConfigCompress
       indexing = INDEXING
       def __init__(self, config):
           super().__init__(config)
           replace_layers_upon_init(self, INDEXING)

   class NewModelCompressionSchemesManager(CompressionSchemesManager):
       def __init__(self, model):
           super().__init__(model, INDEXING)
   ```

3. Create `transformersurgeon/models/newmodel_c/__init__.py` exporting the three classes and add them to `transformersurgeon/models/__init__.py`.

### Adding a Calibration Summary

1. Subclass `CalibrationSummary` from `transformersurgeon/calibration/summaries/base.py`. Set the `name` class attribute and `required_raw_data` (list of raw data stream names the summary consumes). Implement `update_from_raw()` (single-batch update) and optionally `update_runtime()` for running-statistic accumulation across batches.

2. Register a singleton instance in `transformersurgeon/calibration/summaries/registry.py`:
   ```python
   SUMMARY_REGISTRY["my_summary_name"] = MySummary()
   ```

3. Reference `"my_summary_name"` in your `Compressor.needs_calibration()` return value.

### Adding a Raw Data Collector

1. Subclass `RawDataCollector` from `transformersurgeon/calibration/raw_data/base.py`. Set the `name` class attribute and implement `build_forward_hook(emit_raw)`. Set `requires_backward = True` and implement `collect_after_backward` if gradient data is needed.

2. Register in `transformersurgeon/calibration/raw_data/registry.py`:
   ```python
   RAW_DATA_REGISTRY["my_collector"] = MyCollector
   ```
