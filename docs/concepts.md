# Core Concepts

This page explains the mental model behind TransformerSurgeon. Reading it once
will make the API feel intuitive rather than arbitrary.

---

## The Three-Layer Model

TransformerSurgeon separates three concerns:

| Layer | Class | Answers the question |
|---|---|---|
| *What* | `Compressor` | Which algorithm? What parameters? |
| *Where* | `CompressionScheme` | Which projection layer in which block? |
| *How many* | `CompressionSchemesManager` | All target layers in the whole model |

A **`Compressor`** is a stateless algorithm object (e.g., `LRDer`, `StructuredPruner`).
It knows how to compress one `LinearCompressed` module and whether it needs
calibration data to do so.

A **`CompressionScheme`** binds one model path (e.g., `model.layers.3.self_attn.q_proj`)
to one set of compressor configs. You never create schemes directly — the manager
builds them from the model's indexing metadata at init time.

A **`CompressionSchemesManager`** is the user-facing entry point. It holds all
schemes for a model, lets you filter them with `criteria`, and coordinates
calibration and application in the right order.

You never subclass `CompressionScheme` or the manager to add a new compression
method — you only subclass the `Compressor` ABC.

---

## `LinearCompressed` — The Compression Target

When you load a supported model class (e.g., `Qwen2ForCausalLMCompress`), every
target projection layer is **replaced at init time** with `LinearCompressed` — a
drop-in `nn.Linear` subclass.

In its default **full-rank mode** `LinearCompressed` behaves identically to
`nn.Linear`: `weight` has shape `[out_features, in_features]` and `weight_2` is
`None`. There is no overhead in forward pass, parameter count, or memory.

After `module.init_lrd(rank)`:

- `weight` is reshaped to `[out_features, rank]`
- `weight_2` is added with shape `[rank, in_features]`
- The forward pass computes `x @ weight_2.T @ weight.T` (two smaller matmuls)

Call `module.cancel_lrd()` to merge the factors back into a single full-rank
`weight` and remove `weight_2`. This is what `manager.restore()` triggers
internally.

A `Compressor.apply()` always receives a `LinearCompressed` instance — never a
plain `nn.Linear`.

---

## The `Compressor` Lifecycle

Here is exactly what happens under the hood when you call the three-line manager
idiom for a calibration-aware method (SVD-LLM-v2 in this example):

```python
manager.set("lrd", "method", "svd-llm-v2")
manager.set("lrd", "rank", 128, criteria="mlp")
manager.set_calibration_data(calibration_loader)
manager.apply(device=device)
```

**Step 1 — `set(...)`**
Writes `rank=128` and `method="svd-llm-v2"` into the `compression_config` dict
of every scheme whose path contains `"mlp"`. No module is touched yet.

**Step 2 — `set_calibration_data(loader)`**
Stores the DataLoader on the manager. Still no computation.

**Step 3 — `manager.apply()` orchestration**

For each scheme the manager:

1. Calls `compressor.set_calibration_store(scheme.calibration_data)` — hands
   the scheme's private result dict to the compressor so it can read summaries
   during `apply`.
2. Calls `compressor.needs_calibration()` — returns `("covariance",)` for
   `svd-llm-v2`.
3. Since calibration is needed, calls `run_compression_calibration(...)` which:
   - Registers `ActivationCollector` hooks on the target module
   - Runs the model over the calibration DataLoader
   - Routes each captured activation tensor to `CovarianceSummary`
   - Writes the result into `scheme.calibration_data["covariance"]`
4. Calls `compressor.apply(module, hard=False)`:
   - Reads `self.calibration_store["covariance"]`
   - Factorizes `module.weight` using the whitened SVD variant
   - Calls `module.init_lrd(rank)` to set up the two-matrix structure
   - Copies the factored matrices into `module.weight` and `module.weight_2`

**Step 4 — `manager.restore()` (optional)**

Calls `compressor.restore(module)` for each scheme:

- Reconstructs `full_weight = module.weight @ module.weight_2`
- Calls `module.cancel_lrd()` to remove `weight_2` and restore the original shape
- Copies `full_weight` back into `module.weight`

---

## Calibration Pipeline — Raw Data and Summaries

The calibration system has two levels:

### Level 1: `RawDataCollector` — hook factory

A `RawDataCollector` is **stateless**. The backbone instantiates it once per
required stream and asks it for a forward hook via `build_forward_hook(emit_raw=...)`.
The hook fires once per module per calibration batch and calls `emit_raw(name, tensor)`.

The `emit_raw` callback fans the tensor out to all `SummaryRuntime` objects that
declared this stream in their `required_raw_data`.

### Level 2: `CalibrationSummary` — statistic accumulator

A `CalibrationSummary` is a **singleton** registered in `SUMMARY_REGISTRY`. Its
per-scheme accumulation state lives in a companion `SummaryRuntime` object.
When all required raw streams have arrived (matched in FIFO order), the runtime
calls `summary.update_runtime(runtime, calibration_store, payload)`, which
writes the result into the per-scheme `calibration_store` dict.

### Tensor flow

```
DataLoader batch
  → model forward (hooks registered on target LinearCompressed modules)
  → ActivationCollector hook fires
      → emit_raw("activation", tensor)
          → SummaryRuntime buffers tensor
          → all required streams present → fire update
          → CovarianceSummary.update_runtime writes calibration_store["covariance"]
  → LRDer.apply reads calibration_store["covariance"]
```

The `calibration_store` is a plain `dict` owned by one `CompressionScheme`.
Summaries write to it; compressors read from it.

---

## Standard vs. Cascade Calibration

Select the mode with `manager.set_calibration_mode(mode)` before calling
`apply`.

### `"standard"` (default)

All target schemes are calibrated in a **single model pass**. The backbone
registers hooks on every target module at once and runs the DataLoader once.
Works for all methods that only need local activations (e.g., SVD-LLM-v2,
magnitude pruning).

### `"cascade"`

Required for **AA-SVD** and any other method that needs a *shifted model* — a
copy of the model with a small positional offset. In cascade mode:

- Schemes are grouped according to `calibration_groups` defined in the model's
  `indexing_*.py` file.
- Each group is calibrated in a **separate pass**. Only one group's hooks are
  active per pass, so the shifted model only needs to mirror the current group's
  modules.
- Ungrouped layers are calibrated as singleton stages.

The `calibration_groups` structure in an indexing file looks like:

```python
"calibration_groups": {
    "self_attn": [["q_proj", "k_proj", "v_proj"], ["o_proj"]],
    "mlp":       [["gate_proj", "up_proj"], ["down_proj"]],
}
```

Groups within a subblock are calibrated together (parallel hooks); subblocks
are processed sequentially across transformer layers.

```python
manager.set("lrd", "method", "aa-svd")
manager.set("lrd", "rank", 128, criteria="all")
manager.set_calibration_mode(mode="cascade")
manager.set_calibration_data(calibration_loader)
manager.apply(device=device, verbose=True)
```

With `verbose=True`, cascade calibration prints per-stage diagnostics:
`pairs`, `mean_rel_l2_diff`, and `max_rel_l2_diff` for shifted-summary stages.

---

## VCON — Smooth Compression for Fine-Tuning

`VCONBlock` wraps any module in an original / compressed pair and blends their
outputs with a scalar `beta`:

```
output = beta * block_a(x) + (1 - beta) * block_b(x)
```

- `beta = 1.0` → original output only
- `beta = 0.0` → compressed output only
- Intermediate values → smooth interpolation

The typical VCON workflow for knowledge-distillation-style fine-tuning:

```python
manager.init_vcon(criteria="mlp")          # wrap target layers in VCONBlock
manager.apply(hard=False, criteria="mlp")  # compress only block_b
manager.set_vcon_beta(1.0, criteria="mlp") # start at original output

# --- fine-tune the model, gradually lowering beta toward 0.0 ---

manager.cancel_vcon(keep_block_b=True)     # collapse to compressed module
```

`init_vcon` must be called before `apply`. `cancel_vcon(keep_block_b=True)`
discards `block_a` (original) and keeps `block_b` (compressed), leaving a
standard (non-VCON) module.

---

## Extending TransformerSurgeon

### New compression method

1. Subclass `Compressor` (`compression/abstract.py`) and implement all six
   abstract methods.
2. Add the class to `COMPRESSOR_DICT` in `compression/registry.py`.
3. Add a parameter schema entry to `COMPRESSION_REGISTRY` (same file) with
   defaults and validator functions.

### New raw data collector

1. Subclass `RawDataCollector` (`calibration/raw_data/base.py`), set the
   `name` class attribute, and implement `build_forward_hook` (and/or
   `collect_after_backward` if `requires_backward = True`).
2. Add a factory entry to `RAW_DATA_REGISTRY` in
   `calibration/raw_data/registry.py`.

### New calibration summary

1. Subclass `CalibrationSummary` (`calibration/summaries/base.py`), set `name`
   and `required_raw_data`, and implement `update_from_raw`. Override
   `update_runtime` for multi-batch running statistics.
2. Add a singleton instance to `SUMMARY_REGISTRY` in
   `calibration/summaries/registry.py`.
3. Reference `self.name` in your `Compressor.needs_calibration()` return value.

### New model family

Create two files in `transformersurgeon/models/newmodel_c/`:

1. **`indexing_newmodel_c.py`** — define the `INDEXING` dict with
   `num_blocks_attr`, `path_list`, `path_template`, and optionally
   `calibration_groups` and export metadata.
2. **`define_newmodel_c.py`** — define three classes following the standard
   pattern:

```python
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

Export the three classes from `models/newmodel_c/__init__.py` and add them to
`models/__init__.py`. No changes to the manager or compression logic are needed.
