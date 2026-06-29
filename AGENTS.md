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
pip install -e .
```

Run tests:

```bash
# Core logic — no large models required
pytest test/compression_tests/ -v

# Encoder model tests (BERT, DistilBERT) — small models, fast
pytest test/bert_tests/ -v
pytest test/distilbert_tests/ -v

# Causal LM tests — require model downloads
pytest test/qwen_tests/inference_test.py -v
pytest test/qwen_tests/svd_llm_v2_test.py -v

# Vision-language
pytest test/qwen_vl_tests/ -v

# HuggingFace export roundtrip
pytest test/hf_export_tests/ -v

# Single file
pytest test/compression_tests/compression_test.py::test_name -v
```

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
| Add a new model family | `models/newmodel_c/indexing_newmodel_c.py` + `models/newmodel_c/define_newmodel_c.py` |
| Change what layers get compressed in a model | `models/qwen2_c/indexing_qwen2_c.py` (or the relevant model's `indexing_*.py`) |
| Change layer filtering / criteria logic | `utils/manager.py` → `iter_filtered()` |
| Change calibration data collection hooks | `calibration/raw_data/activation.py` or `weight_grad.py` |
| Add a new calibration summary (statistic) | Subclass `calibration/summaries/base.py` → new file → register in `calibration/summaries/registry.py` |
| Change how covariance is computed | `calibration/summaries/covariance.py` |
| Change the `LinearCompressed` forward pass | `blocks/linear_compressed.py` |
| Change VCON blending | `blocks/vcon_block.py` |
| Change HuggingFace export | `hf/hf_export.py` |
| Change export graph conversion | `utils/convert.py` |
| Change what parameters are valid for a compression type | `compression/registry.py` → `COMPRESSION_REGISTRY` |

---

## Compression Parameter Reference

All parameters are set via `manager.set(compression_type, param, value, criteria=...)`.

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

`"gradient"` requires `"weight_grad"` calibration summary.

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
| `precision` | `"full"` | `"full"`, `"int8"`, `"int4"`, `"int2"`, `"binary"` |
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
