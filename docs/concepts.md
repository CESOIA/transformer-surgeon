# Core Concepts

This page explains the mental model behind TransformerSurgeon. Reading it once
will make the API feel intuitive rather than arbitrary.

---

## Folders and the Two Dictionaries

The package is one pipeline, compression → export, and every folder has one job:
`models/` describes models (as dictionaries), `compression/` decides and applies
compression, `calibration/` only collects statistics, `blocks/` is a simplified
export-ready transformer made of selectable *atoms*, `export/` turns the result
into device artifacts, and `utils/` holds generic helpers. Code outside `models/`
never mentions a specific model.

Two dictionaries connect the folders:

- the **model dictionary** (`models/<m>_c/indexing_<m>_c.py`) describes a model:
  where its layers are, which ones can be compressed, how pruning couples them,
  how to calibrate them, and which operations its export graph needs
  (`structure`, `attn_type`, `mlp_type`, `norm_type`, `rope_type`), i.e. which
  atoms of `blocks/catalog.py::BLOCK_CATALOG` build it;
- **`BACKEND_OPERATIONS`** (`export/operations.py`) says, for every block type and
  every backend, which implementation runs on the device by default and which
  alternatives are supported (for example the KV cache: an internal mutable
  buffer on XNNPACK/QNN, an in-place graph input/output on ONNX/TensorRT).

So adding a model touches only `models/`, a compression method only
`compression/`, a calibration statistic only `calibration/`, and a backend only
`export/`.

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
`nn.Linear`: `weight` has shape `[out_features, in_features]` and `linear_V` is
`None`. There is no overhead in forward pass, parameter count, or memory.

After `module.init_lrd(rank)`:

- `weight` is reshaped to `[out_features, rank]` (the U factor)
- `linear_V` is added, a linear layer with weight `[rank, in_features]` (the V factor)
- The forward pass computes `linear_V` then the U projection (two smaller matmuls)

Call `module.cancel_lrd()` to merge the factors back into a single full-rank
`weight` and remove `linear_V`. The two methods come from
`blocks/layers/low_rank.py::LowRankMixin`, shared with `EmbeddingCompressed`. This is what `manager.restore()` triggers
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
   - Copies the factored matrices into `module.weight` and `module.linear_V.weight`

**Step 4 — `manager.restore()` (optional)**

Calls `compressor.restore(module)` for each scheme:

- Reconstructs `full_weight = module.weight @ module.linear_V.weight`
- Calls `module.cancel_lrd()` to remove `linear_V` and restore the original shape
- Copies `full_weight` back into `module.weight`

---

## Structured Pruning — Coupling and Grouping

Structured pruning removes whole output neurons (rows) of a layer. Two mechanisms
keep a pruned model consistent, both driven by generic annotations under the
`pruning` key of each model's `indexing_*.py` (so `compression/` stays
model-agnostic). The scheme turns those names into module chains
(`CompressionScheme.coupled_target_chains`); the compressors only receive chains:

- **Coupling (`output_dependence`).** When a layer's output rows are removed,
  the input columns of the layers it feeds must shrink to match. Hard
  `StructuredPruner.apply` builds the keep-mask, removes the rows (resizing
  `weight`/`bias`/`out_features`), then cascades into each coupled next layer via
  `CoupledPruner` (input-column removal, hard-only). Soft apply just zeroes rows,
  so no cascade is needed. The kept-dimension formula lives once in
  `compression/structured_pruning_methods/pruning_dims.py` (`effective_out_features`);
  `convert_for_export` reads the pruned shapes from the source layers, so a
  hard-pruned model converts and exports with matching shapes.

- **Grouping (shared masks).** Some layers must be pruned *identically* — e.g.
  `gate_proj`/`up_proj` (multiplied together) or `q_proj`/`k_proj` (matched
  head dim). `manager.auto_groups()` reads `coupled_masks` (within a block) and
  `coupled_masks_all` (across all blocks, for the residual/hidden writers) and
  builds `SchemeGroup`s holding **pointers** to the schemes plus a shared
  `properties` dict. With `share_mask` enabled, the first compressor in a group
  computes one reduced mask (`reduce_op` over the members' scores) and stores it
  in the group; the others reuse it. All of this lives in
  `compression/structured_pruning.py` — the manager only builds groups and
  iterates `scheme.apply`.

`granularity=<int>` prunes uniformly within each chunk (e.g. per attention head);
`repeated_pattern=True` reduces scores across chunks into one length-`granularity`
mask that is tiled back, letting layers with different chunk counts (GQA
`q_proj`/`k_proj`) share a single mask. Group-only options (`share_mask`, listed in
`GROUP_OPTIONS`) can only be set through `manager.set(..., group=...)`.

Scope: both MLP and attention (q/k/v) structured pruning are wired end-to-end
through convert/export. Attention hard pruning changes `head_dim` (GQA); RoPE
projection geometry and the pruned KV-cache are resolved per-kv-group at
conversion time, and the GQA `v→o` coupling is handled via
`coupled_repeated_pattern`.

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

### Opting a family out of cascade calibration

An indexing block can set `'no_cascade_calibration': True` when its layer layout
isn't compatible with the block-wise cascade algorithm above. `manager.apply()`
then raises a `ValueError` immediately if cascade mode is requested with any
scheme from that block selected, instead of silently running an unsupported flow.
`bert_c` and `modernbert_c` set this flag today:

- **BERT** — cascade's `_collect_preprocessing_outputs`/`_collect_loader_inputs`
  discard `attention_mask` once the embeddings step runs, threading only the raw
  hidden-state tensor between blocks. Bidirectional encoders lean on
  `attention_mask` much more than causal decoders (padded batches, no causal mask
  to fall back on), so block-wise calibration without it isn't a safe stand-in for
  real inference.
- **ModernBERT** — alternating global/local attention layers each need their own
  rotary embedding call keyed by layer type
  (`rotary_emb(hidden_states, position_ids, layer_type)`), but cascade's
  position-embedding injection calls the configured rotary module the same way
  for every block.

Use `"standard"` calibration mode for these families (`"svd"` or `"svd-llm-v2"`
LRD instead of `"aa-svd"`).

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

## Export Backends

`transformersurgeon.export.export_to_backend(model, config)` picks the exporter
registered for `config.backend` (`export/exporter.py`) and runs the steps every
backend shares (`Exporter.export`), then the backend's own `lower()`:

| Backend | Config class | Output |
|---|---|---|
| `xnnpack` | `XNNPACKExportConfig` | ExecuTorch `.pte` |
| `qnn` | `QNNExportConfig` | ExecuTorch `.pte` (Qualcomm NPU) |
| `onnx` | `ONNXExportConfig` | Portable ONNX + manifest (I/O contract) |
| `tensorrt` | `TensorRTExportConfig` | ONNX + TensorRT engine built from it |

1. **Options.** `convert_options` are completed with the backend defaults of
   `BACKEND_OPERATIONS`; unsupported choices are rejected before tracing.
2. **Conversion.** `convert_for_export` (`export/convert.py`) builds the decoder
   stack from `blocks/` atoms and copies shapes, ranks, biases and quantisation
   state from the source layers.
3. **Wrapper.** `export/wrapper.py::LLMWrapper` presents embedding, decoder and
   output head with a fixed signature that depends only on the KV-cache atom.
4. **Quantisation metadata.** `extract_layer_quant_info` (`export/quant_info.py`)
   asks the compression step for each layer's description
   (`compression/quantization.py::export_quant_info`). A layer is quantised in the
   artifact if and only if it was quantised on the model.
5. **Lowering.** PT2E backends (XNNPACK, QNN) annotate the quantised linears with
   `export/pt2e_linear_quantizer.py::LinearOnlyQuantizer`, calibrate the observers
   and inject the exact scales (`export/pt2e.py`); ONNX rewrites quantised weights
   as `DequantizeLinear` (`export/backends/onnx/quantization.py`).
6. **Result.** `export/result.py::finalize_export_result` runs the optional weight
   check and writes the manifest (`export/manifest.py`).

This is what makes **mixed-precision export** work: only the quantised layers
get quantise/dequantise operations.

```python
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.backends import TensorRTExportConfig

# Portable ONNX + manifest, and a TensorRT engine for the local GPU
# (for a Jetson, set build_engine=False and build the engine on the device).
config = TensorRTExportConfig(
    output_path="out/model.onnx", max_input_len=512,
    convert_options={"max_cache_len": 1024, "rmsnorm_impl": "rmsnorm_upcast"},
)
result = export_to_backend(model, config)
print(result.onnx_path, result.engine_path)
```

For TensorRT, the ONNX file + manifest are the portable artifact and the engine is
built where it runs (`python -m transformersurgeon.export.backends.tensorrt.exporter
model.manifest.json` on a Jetson). Tests: `test/export/test_exporter.py`
(capability-gated).

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

Create `transformersurgeon/models/newmodel_c/` with two dictionaries:

1. **`indexing_newmodel_c.py`** — the model dictionary (`num_blocks_attr`,
   `path_list`, `path_template`, and optionally `pruning`, `calibration_groups`
   and the export keys `structure`, `attn_type`, `mlp_type`, `norm_type`,
   `rope_type`). Start from the closest existing family.
2. **`family_newmodel_c.py`** — the family dictionary, from which
   `models/factory.py::build_family` generates the classes:

```python
from ..factory import build_family
from .indexing_newmodel_c import NEWMODEL_C_INDEXING

NEWMODEL_C_FAMILY = {
    "indexing": NEWMODEL_C_INDEXING,
    "hf_config": "transformers.models.newmodel.configuration_newmodel.NewModelConfig",
    "hf_model": "transformers.models.newmodel.modeling_newmodel.NewModelForCausalLM",
    "class_names": {"config": "NewModelConfigCompress", "model": "NewModelForCausalLMCompress",
                    "manager": "NewModelCompressionSchemesManager"},
    "auto_class": "AutoModelForCausalLM",
}
_classes = build_family(NEWMODEL_C_FAMILY, __name__)
```

Add an `__init__.py` re-exporting both modules and one line to
`models/__init__.py::FAMILIES`. Nothing else changes (`mistral_c` was added this way,
as a copy of `llama_c`).

### New export backend

1. In `transformersurgeon/export/backends/`, subclass `Exporter`
   (`export/exporter.py`) with `backend`, `config_class` (a `BackendExportConfig`
   subclass) and `lower(components, config)`; decorate it with
   `@register_exporter` and import it in `export/backends/__init__.py`.
   ExecuTorch backends can start from `executorch_base.py::ExecuTorchExporter`
   (`write_program`, `finalize`).
2. Add the backend to every entry of `export/operations.py::BACKEND_OPERATIONS`
   (default and supported implementations).
3. Add a capability-gated test in `test/export/test_exporter.py`.
