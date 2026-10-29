# AGENTS.md — transformer-surgeon

Agent-oriented guide for working on this codebase. For user-facing docs read [README.md](README.md); for the
architecture in depth read [docs/concepts.md](docs/concepts.md).

---

## Project Summary

transformer-surgeon adds compression (low-rank decomposition, pruning, quantization) to Hugging Face transformer
models without breaking `from_pretrained` / `generate` / `save_pretrained`, and exports the compressed model to
device backends (ExecuTorch XNNPACK and Qualcomm QNN, ONNX, TensorRT). The entry points are the generated
per-family classes (e.g. `Qwen2ForCausalLMCompress`) and their manager (`Qwen2CompressionSchemesManager`):

```python
manager.set("lrd", "rank", 128, criteria="mlp")   # configure (declarative)
manager.apply(hard=False)                          # compress (reversible)
model.generate(...)                                # use normally
manager.restore()                                  # undo
manager.apply(hard=True)                           # permanent, before export
export_to_backend(model, QNNExportConfig(output_path="model.pte"))
```

---

## Dev Setup and Test Commands

```bash
pip install -e ".[dev]"
pytest                     # the suite in test/ (one folder per package folder), tiny random models, no downloads
```

Backends are capability-gated (`test/_helpers/capabilities.py`): tests skip when ExecuTorch, the QNN SDK, ONNX,
TensorRT or a GPU are missing. On the development machine use the conda environments `py312_tensorrt`,
`py310-executorch-qnn`, `py312_executorch_xnnpack`.

---

## Architecture

One pipeline, compression → export, driven by dictionaries. Each package folder has one job:

| Folder | Job | Must not contain |
|---|---|---|
| `utils/` | Generic helpers on modules, dotted paths, batches, devices | anything about a model, a compression type or the export |
| `models/` | One folder per family: the **model dictionary** (`indexing_<m>.py`) and the **family dictionary** (`family_<m>.py`); `factory.py::build_family` generates config/model/manager classes | algorithms |
| `compression/` | Manager, schemes, groups, compressors (LRD, pruning, quantization, SmoothQuant), option registry and validators | model names, export details |
| `calibration/` | Collection of calibration statistics (raw-data collectors, summaries, input absmax, calibration datasets) | compression decisions |
| `blocks/` | Simplified export-ready transformer split into selectable **atoms** (one class per file), listed in `blocks/catalog.py::BLOCK_CATALOG` | model-specific information, compression decisions |
| `export/` | Conversion to `blocks/` stacks, generic exporter, per-backend exporters, and `export/operations.py::BACKEND_OPERATIONS` (block type → device operation per backend) | compression decisions |
| `hf/` | Save to a Hugging Face directory / Hub with remote code | — |

Inside `compression/` there are three layers: **what** (`Compressor` subclasses: how to compress one layer),
**where** (`CompressionScheme`: one model layer, its compressors and its topology, built by the manager),
**how many** (`CompressionSchemesManager`: filtering by criteria, calibration, apply/restore).

Two dictionaries join the folders:

- the model dictionary says which operations the model needs (`structure`, `attn_type`, `mlp_type`,
  `norm_type`, `rope_type`), i.e. which atoms of `BLOCK_CATALOG` build its export graph;
- `BACKEND_OPERATIONS` says, per backend, which implementation of each operation is used by default and which
  alternatives are supported (`cache_impl`, `attn_impl`, `rmsnorm_impl`, `masked_softmax`, `add_batch_dim`,
  and the configuration fields `quant_path`, `quant_scope`, `int4_backend`, `convert_linear_to_conv2d`).
  `resolve_convert_options(backend, options)` fills the defaults and rejects unsupported choices.

---

## File Map

| Task | Where |
|---|---|
| Change a compression algorithm | `compression/lrd.py`, `structured_pruning.py`, `block_pruning.py`, `unstructured_pruning.py`, `quantization.py`, `smoothquant.py` |
| Add a method variant (LRD, quantization, unstructured pruning) | new file in `compression/<type>_methods/` + one entry in its `METHOD_FUNCTIONS` |
| Change option defaults / validation | `compression/registry.py` (`COMPRESSION_REGISTRY`), `compression/validators.py` |
| Change cross-layer pruning (coupling) | `compression/coupled_pruning.py`; chain resolution in `compression/scheme.py::coupled_target_chains` |
| Change grouping (shared masks) | `compression/grouping.py` (`SchemeGroup`) + `compression/manager.py` (`create_group`, `auto_groups`) |
| Change criteria filtering | `compression/manager.py::iter_filtered` |
| Change cascade (block-wise) calibration | `compression/cascade.py` |
| Change which layers become compressible at model init | `compression/compressible_layers.py::replace_layers_upon_init` |
| Add / change a model family | `models/<m>_c/` (two dictionaries) + one line in `models/__init__.py::FAMILIES` |
| Change calibration hooks / statistics | `calibration/raw_data/`, `calibration/summaries/` (+ their `registry.py`) |
| Calibration data (token windows, WikiText-2) | `calibration/datasets.py` |
| Change the compressible layers | `blocks/layers/` (`LinearCompressed`, `EmbeddingCompressed`, `Conv2d/3dCompressed`, `LowRankMixin`, `VCONBlock, AtomicSum, LayerScale`) |
| Change an export atom | `blocks/attention/{kernels,kv_cache}/`, `blocks/norm/`, `blocks/mlp/`, `blocks/positions/`, `blocks/stacks/` |
| Add an atom | new file in the right `blocks/` subfolder + entry in `blocks/catalog.py` (+ `BACKEND_OPERATIONS` if the export chooses it) |
| Change HF model → `blocks/` conversion | `export/convert.py::convert_for_export` |
| Change the traced wrapper / example inputs | `export/wrapper.py` |
| Change quantisation metadata read at export | `compression/quantization.py::export_quant_info` (the only reader of the private layer state), `export/quant_info.py` |
| Change PT2E calibration / scale injection | `export/pt2e.py`, `export/pt2e_linear_quantizer.py` |
| Change the runtime manifest | `export/manifest.py` (graph I/O names: `graph_io_names`) |
| Change the generic export steps | `export/exporter.py` (`Exporter.export`, `resolve_components`) |
| Change / add a backend | `export/backends/` (`xnnpack.py`, `qnn/`, `onnx/`, `tensorrt/`, shared ExecuTorch steps in `executorch_base.py`) |
| Change the HF export | `hf/hf_export.py` |

---

## Export Backends

`export_to_backend(model_or_components, config)` picks the exporter registered for `config.backend`
(`export/exporter.py::EXPORTERS`) and runs `Exporter.export`: complete `convert_options` from
`BACKEND_OPERATIONS`, convert the model with `convert_for_export` if needed, move the components to CPU and the
requested dtype, then call the backend's `lower()`.

| Backend | Config class | Module | Output |
|---|---|---|---|
| `xnnpack` | `XNNPACKExportConfig` | `export/backends/xnnpack.py` | ExecuTorch `.pte` |
| `qnn` | `QNNExportConfig` | `export/backends/qnn/` | ExecuTorch `.pte` (Qualcomm NPU) |
| `onnx` | `ONNXExportConfig` | `export/backends/onnx/` | Portable `model.onnx` + manifest |
| `tensorrt` | `TensorRTExportConfig` | `export/backends/tensorrt/` | the ONNX output + a TensorRT engine built from it |

Every backend reads the per-layer quantisation decided by the compression step (`extract_layer_quant_info`), so
**mixed precision** works: only the quantised layers get quantise/dequantise operations.

```python
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.backends import TensorRTExportConfig

config = TensorRTExportConfig(output_path="out/model.onnx", max_input_len=512,
                              convert_options={"max_cache_len": 1024, "rmsnorm_impl": "rmsnorm_upcast"})
result = export_to_backend(model, config)   # full HF model, or {embedding, decoder, final_layer[, config]}
print(result.onnx_path, result.manifest_path, result.engine_path)
```

Every backend writes `<artifact stem>.manifest.json` next to its artifact (`export/manifest.py::build_llm_manifest`):
cache implementation, layout and dtype, per-layer key/value cache shapes (they differ per layer after pruning) and
graph I/O names. Runners use `load_llm_manifest` and `zero_caches_from_manifest` instead of re-deriving geometry.

## Backend Details

### ONNX → plain TensorRT (`onnx`, `tensorrt`) — the Jetson path

A TensorRT engine only runs on the GPU architecture + TensorRT version it was built with, so for edge targets (Jetson Orin) the artifact is the ONNX file; the engine is built on the device (`python -m transformersurgeon.export.backends.tensorrt.exporter model.manifest.json`). Needs `pip install -e ".[onnx]"` plus the `tensorrt` Python package matching the CUDA version (e.g. `tensorrt-cu13`). CLI: `scripts/tensorrt/export_and_generate.py`. Measured at parity with NVIDIA's TensorRT Edge-LLM for decode.

- **Cache:** convert with `cache_impl="io_inplace"`: BHSD `(1, kv_heads, max_cache_len, head_dim)` caches as graph I/O, written by `tsurgeon::kv_cache_update` (`blocks/attention/kv_cache/io_inplace.py`) which exports as opset-24 `TensorScatter`. TensorRT aliases each `present_*` output to its `past_*` input and updates it in place — bind both to the same buffer. `mutable` is rejected (ONNX has no state).
- **Fused decode attention (`cache_impl="io_paged"`, opt-in):** one paged K+V pool per layer `(2, ceil(max_cache_len/128), 128, kv_heads, head_dim)` (`past_kv_i`/`present_kv_i`, bind both to one buffer: the plugin writes in place without declaring the alias) and one `tsurgeon::paged_attention` op per layer (RoPE + cache write + attention over the filled slots; eager implementation in `blocks/attention/kernels/paged.py`). The TensorRT backend maps it to TensorRT Edge-LLM's `AttentionPlugin` (XQA decode kernels, NVRTC-compiled at engine build): decode cost follows the context, not `max_cache_len`, unlike the static-cache MatMul/Softmax path which reads every slot. Needs `EDGELLM_PLUGIN_PATH`, equal q/k/v head dims (head_dim 32/64/128, q:kv 1–8) and unpruned RoPE (within-head pruning keeps per-kv-group frequencies: use `io_inplace`); whole-head pruning works. Decode only (`max_input_len=1`; prompts run token by token). The plain `onnx` backend rejects it (no standard ONNX form).
- **Prefill:** `max_input_len > 1` makes `input_ids` `(seq,)` dynamic; only `io_inplace` accepts `in_seq_len > 1`. The engine gets two optimization profiles: 0 = decode (seq 1), 1 = prefill.
- **Numerics:** use `rmsnorm_upcast=True` (float32 RMSNorm, HF semantics) for fp16 GPU exports; `rmsnorm_prescale=False` alone overflows in fp16.
- **Quantization:** weight-only. INT4 → blocked `DequantizeLinear` (per-channel scales tiled over 128/64/32 blocks, exact). Plain TensorRT 10.16 has no fast single-token INT4 GEMV, so INT4 decode runs at fp16 speed unless `int4_backend="edgellm_plugin"` (opt-in): eligible layers become TensorRT Edge-LLM's `Int4GroupwiseGemmPluginV2`, and the engine then needs `libNvInfer_edgellm_plugin.so` (`EDGELLM_PLUGIN_PATH`; recorded in the manifest under `plugins`). INT8 weight-only gives no speedup on TensorRT (it folds to fp16). **W8A8 (native INT8 GEMMs):** quantize the layer with `precision_activation=8, method_activation="maxmin", scheme_activation="symmetric", quantize_output_activation=False` and calibration data; the exporter (`apply_activation_quantization`) puts a per-tensor INT8 QuantizeLinear/DequantizeLinear on the GEMM's activation input (an LRD layer: on its V factor's input), recorded as `act_precision` in the manifest's `quantized_layers`. Only symmetric INT8 per-tensor input scales are exportable; output-activation quantization raises.
  Models with massive activations (Qwen2.5 down_proj input amax ~5e3) need **SmoothQuant** first: `manager.set_calibration_data(...)`, then `manager.smooth_activations(alpha=0.5, criteria=...)` *before* `manager.apply` of the quantization (groups from `indexing['smoothing']`: norm→q/k/v, norm→gate/up, up→down; float result unchanged; refuses already-quantized weights). Qwen2.5-1.5B W8A8 all linears: KL vs fp16 1.71 → 0.18 (alpha 0.5).
- **Runtime:** `export/backends/tensorrt/session.py::TensorRTLLMSession` (prefill + CUDA-graph greedy decode, manifest-driven); `export/backends/tensorrt/engine.py` (engine build/run helpers).
- **Extending to a new target:** `ONNXExportConfig.custom_translations` (torch op → onnxscript) and `graph_passes` (post-export `(model, manifest)` rewrites) — `edgellm_int4.py` is the worked example.

Tests: `test/export/test_exporter.py` (ONNX, TensorRT, graph passes), `test/blocks/test_attention.py` (cache and kernel atoms), gated by `requires_onnx` / `requires_tensorrt` / `requires_edgellm_plugin`. Benchmark workspace with the Edge-LLM vendor baseline: `../trt-models/`.

### XNNPACK: `attn_impl` — optimized causal attention

`MHACausal` (`blocks/attention/causal.py`) supports three attention-kernel atoms (`blocks/attention/kernels/`) via `attn_impl` in `convert_options`, orthogonal to `cache_impl`:

| `attn_impl` | Kernel | Notes |
|---|---|---|
| `"manual"` (default) | Explicit GQA-aware softmax/matmul (`kernels/manual.py::attention`) | Always available, any backend |
| `"sdpa"` | `torch.nn.functional.scaled_dot_product_attention` | Equivalent to the legacy `use_sdpa=True` |
| `"custom_sdpa"` | ExecuTorch's fused CPU custom ops (`torch.ops.llama.custom_sdpa` / `update_cache`) | XNNPACK-only. Requires `cache_impl="mutable"` and `q_head_dim == value_head_dim` — both validated at construction/forward time rather than silently producing wrong output |

`custom_sdpa` needs ExecuTorch's `executorch.extension.llm.custom_ops` extension importable both at **export** time (`kernels/custom_sdpa.py` imports it lazily, guarded with a clear `RuntimeError` if unavailable) and at **inference** time in whatever process later loads the `.pte` — the AOT registration from the export process does not carry over to a separate process. `scripts/executorch/xnnpack/inference_exported_test.py` handles this by conditionally re-importing the extension, keyed off an `attn_impl` field in the `.cache_meta.json` sidecar written by `exporter_function_test.py`.

CLI: `scripts/executorch/xnnpack/exporter_function_test.py --attn-impl {manual,sdpa,custom_sdpa}`. Tests: `test/blocks/test_attention.py`, gated by `test/_helpers/capabilities.py::requires_custom_sdpa` (narrower than `requires_executorch`: a minimal ExecuTorch install may lack the LLM custom-ops extension).

### XNNPACK: `add_batch_dim` — experimental leading batch dimension

`add_batch_dim` (`convert_options`, default `False`) makes `LLMWrapper`'s `input_ids`
and every component's hidden states carry an explicit leading batch dim
(`(1, in_seq_len, ...)`) instead of the framework's usual bare `(in_seq_len,
...)` contract — matching ExecuTorch's own llama exporter's convention, on the
hypothesis that XNNPACK may select faster GEMM/FC microkernels or weight
packing for batched shapes. Threaded through `blocks/stacks/config.py`,
`export/convert.py`, `blocks/stacks/decoder.py` exactly like `attn_impl`/`cache_impl`.

Scoped narrowly to minimize risk: `MHACausal` absorbs the batch dim
immediately after q/k/v projection (`.view(in_seq_len, ...)` on a
leading-1-dim tensor is a no-op reshape) and re-attaches it only after
`out_proj`, so **attention math, KV-cache read/write, and `custom_sdpa` are
completely unaffected either way** — only the surrounding Linear/RMSNorm/MLP
"glue" ops actually see a batch dim. `MHAEncoder` (encoder-only models) is not
touched by this option at all.

This is opt-in and off by default specifically so QNN/TensorRT and every
existing XNNPACK export are 100% unaffected unless a caller explicitly passes
`"add_batch_dim": True` in `convert_options`. It has not been validated against
QNN's or TensorRT's partitioners/converters — the extra leading dim may not be
supported by every op converter on those backends, so treat it as
**XNNPACK-only until proven otherwise elsewhere**.

### XNNPACK: `quant_path` — how quantized linears are lowered

`XNNPACKExportConfig.quant_path`: `"pt2e"` (legacy; per-tensor/per-channel
only; the linear-only PT2E quantizer always used dynamic activations) or
`"torchao"` (ExecuTorch's own LLM recipe): each quantized linear becomes an
`IntxUnpackedToInt8Tensor` from tsurgeon's exact integers/scales, lowered with
`XnnpackDynamicallyQuantizedPartitioner` then `XnnpackPartitioner` →
qd8-f32-qc4w/qb4w/qc8w kernels (= vendor 8da4w/8da8w). Quantize with
`method_activation="dynamic_per_token"`; weight-only layers warn (XNNPACK has no
fp32×intN GEMM). Keep XNNPACK exports **fp32**: fp16 XNNPACK kernels drift
(KL ≈ 4e-3 vs HF on Qwen2.5-1.5B; ARM fp16 GEMMs accumulate in fp16), and the
fp16 qd8×int4 kernel does not exist on x86 hosts (`xnn_status_unsupported_hardware`).

### QNN: `quant_scope`

`QNNExportConfig.quant_scope="linear"` (legacy) annotates only compressed
linears → every quantized/float seam splits the HTP graph (measured on SM8850,
Qwen2.5-0.5B: 169 partitions + CPU mul/add/sigmoid, ~20 tok/s vs 41 tok/s fp16,
incoherent output). `"all"` is the vendor recipe: every op 16-bit MinMax
activations, W4 → 16a4w, W8 → 16a8w (tsurgeon's exact per-channel scales),
float linears 16a16w → one partition. `keep_reference_module=True` returns the
PT2E fake-quant model the .pte was lowered from (`result.reference_module`) —
the host parity reference, because the x86 HTP simulator costs ~8 min/token at 1.5B.

**16a16w and biases (`split_float_linear_bias`, default True).** A 16a16w linear computes its bias
in int32 with scale `act_scale × weight_scale`; with 16-bit weights that scale is tiny (largest bias
≈ input range × max|w| of the channel, ~1–2 on Qwen2.5-1.5B), and Qwen2.5's q/k biases (up to ~316)
overflow it by up to ~65× and are clipped in every layer: MLP-only quantized exports (attention left
float → 16a16w) were garbage on device *and* in the fake-quant reference. With the option on, every
biased linear that is not compression-quantized becomes `add(linear(x, W), b)` before PT2E: the
weights stay 16-bit and the bias gets its own 16-bit activation scale, so only the targeted linears
change precision. Compression-quantized (W8/W4) linears keep the fused derived int32 bias (8-bit
weights have ~250× more headroom). Still 1 delegate / 0 CPU ops. Test: `test/export/test_exporter.py::test_qnn_split_float_linear_bias_moves_only_float_biases_and_keeps_outputs`.

**Activation-only reference (`quantize_float_model`, default False, needs `quant_scope="all"`).** Lowers a
model with *no* compression-quantized linear as the same fixed-point graph (16a16w everywhere, bias split
as above) instead of fp16. Its distance from fp16 is the cost of HTP's 16-bit activations alone, so a
compressed config measured against it isolates the weight compression. Tests: `test/export/test_exporter.py::test_qnn_quantize_float_model_*`.

### QNN: activation calibration and masked softmax

- PT2E calibration (`export/pt2e.py::calibrate_pt2e_observers`) decodes each sequence from position 0
  with the KV cache filling up (reset between sequences). It used to feed every token at position 1 with
  an empty cache, which left the attention/cache observers uncalibrated. Pass real text of the target
  distribution with `QNNExportConfig.calibration_token_ids`; the built-in sentences are only a fallback.
- `convert_options={"masked_softmax": True}` (manual attention only): masked scores become `row min - 20`
  (the vendor's `enable_masked_softmax`) instead of `score + mask_penalty`. It is the same function in float,
  but a 16-bit quantized score tensor no longer spans [-65504, +1e4] (a ~1.15-logit step). Measured on Qwen2.5-1.5B 16a8w
  all-op vs the same weights without activation quantization: KL 0.96 (old) -> 0.57 (decode calibration)
  -> 0.36 (+ masked softmax), top-1 61 % -> 90 %.
- RMSNorm must reach HTP as one fused `RmsNorm` op. ExecuTorch's `RecomposeRmsNorm` matches
  `w * (x * rsqrt(mean(x*x) + eps))` (or with `pow.Tensor_Scalar`), but this exporter rewrites `pow(x, 2)`
  to `pow.Tensor_Tensor`, so the RMS normalisation atoms (`blocks/norm/`) compute `x * x` (bit-identical). Decomposed and quantized,
  `x*x` of Qwen2.5's ~5e3 residual activations spans ~2.5e7 in one 16-bit scale. With the fusion:
  16a8w all-op KL 0.25 -> 0.062, top-1 91 % (`rmsnorm_impl="rmsnorm_prescaled"`; 0.070 with `"rmsnorm"`, the vendor form).
- Attention scores always use the original `1/sqrt(head_dim)` (`MHABase.score_scale`), including after
  q/k head_dim pruning. Pruning channels must not change the softmax temperature, and the converted
  hard-pruned graph must equal the soft-pruned HF model. Before this, it used `1/sqrt(pruned dim)`.

### QNN: block-wise W4 (LPBQ)

HTP holds block-quantized weights only as LPBQ: 4-bit weights, one float scale per output channel and
a 4-bit integer multiplier per block. Free per-group scales (XNNPACK/TensorRT style) cannot reach the
device exactly, so the QNN exporter raises for them. Quantize with `precision=4, granularity="per_group",
group_size=G, group_scale_bits=4`: those layers are annotated `use_16a4w_block` with block `(1, G)`, their
exact scales are injected into the per-block observers, and QNN's encoder
(`builders/node_visitor.py::make_qnn_per_block_config`) re-derives the same channel scale and integers, so the
torch model and the HTP weights are identical (2-layer Qwen2.5-1.5B, G=128, x86 simulator step 0 vs fake-quant
reference: cos 0.999999, KL 7e-6; 1 partition, 0 CPU ops). HTP runs LPBQ only on convolutions (Qualcomm's LLM
recipes lower every linear as a 1×1 conv2d), so an LPBQ export forces ExecuTorch's `ConvertLinearToConv2d`
and unties the embedding/lm_head weight first (that pass mishandles tied weights). The lm_head then gets its own
copy of the matrix: a bigger file, and the same decode reads.

### Numerics caveats found on Qwen2.5-1.5B (large attention logits)

- **TensorRT fp16 GEMV accumulates in fp16.** At M=1 TensorRT 10.16 picks `gemvx_f16f16_f16_f16` kernels (fp16 compute) for every
  fp16 GEMM, including under `STRONGLY_TYPED` with tactic sources restricted (they come from Myelin). Dense layers tolerate it; plain-SVD LRD
  layers do not (Qwen2.5-1.5B lrd0.5-mlp: KL 0.10 vs torch fp16, whose own fp32 floor is 3e-3). `TensorRTExportConfig.fp32_accumulate_lrd`
  (default True, `export/backends/tensorrt/precision_passes.py`) casts the operands of LRD-factor GEMMs to fp32 and the result back. TensorRT fuses
  the casts, so the weights stay fp16 (same engine size) and the result is KL 3.5e-4 at +4.7 % decode time (8.94 vs 8.54 ms, L4). Casting
  only the gate/up V factors recovers most of it (KL 5e-3, +2 %).

- Causal-mask penalty was `-10000`; Qwen2.5 logits exceed ±1e4 (k biases ≈300),
  so masked slots got weight (KL 2.6e-3 vs HF in fp32). Now `-65504`
  (`blocks/stacks/decoder.py`). Any mask constant must exceed the model's logit range.
- q·kᵀ partial sums overflow fp16 (≈1.9e5) → any backend kernel that
  accumulates attention scores in fp16 produces inf/NaN (TensorRT 10.16's fused
  attention did: token 659 @ pos 0). TensorRT export needs the scores in fp32
  (see compression-benchmark `bench/trt_passes.py`, candidate for a default pass).
- `manager.prepare_for_save()` upcasts fp16 LinearCompressed to fp32 and
  dequantizes; call it only before `save_pretrained`, never before export.

### QNN: decode-speed options

| Option | Where | Default | Notes |
|---|---|---|---|
| `rmsnorm_impl` (alias `rmsnorm_prescale`) | `convert_options` → `blocks/stacks/config.py` → `blocks/norm/` atom | `"rmsnorm_prescaled"` | `"rmsnorm"` (or `rmsnorm_prescale=False`) drops the max-abs rescale in front of each RMSNorm so ExecuTorch folds it into one `aten.rms_norm` (as Qualcomm's reference does), saving ~245 HTP dispatches/token. Not bit-identical — rescaling changes how `eps` enters the variance — so keep it on where fp16 overflow is a real risk. CLI: `--no-rmsnorm-prescale` |
| `convert_linear_to_conv2d` | `QNNExportConfig` | `False` | Vendor-recommended 1×1-conv lowering. Off because it showed no measurable DDR change here and ExecuTorch 1.3.1's pass breaks on fp16 weights and on tied weights — `_resolve_linear_to_conv2d` detects both and warns instead of failing deep in lowering. CLI: `--linear-to-conv2d` |

`MHACausal`'s `attn_impl="manual"` kernel (`blocks/attention/kernels/manual.py::attention`) computes
GQA by **broadcast**, not by materialising a `repeat_interleave`'d KV cache, and
permutes the cache *before* the group expansion rather than after. Do not reorder
those two steps: expanding first makes the permute run over a `group_size`-times
larger tensor, which overflows VTCM and cost 537 MB of spill/fill per token on
Qwen2-0.5B. The math is unchanged either way.

`config.max_seq_len` now seeds `convert_options["max_cache_len"]` in
`export_to_backend` when the caller did not set it explicitly — before, the KV cache
was always 2048 slots regardless of `--max-sequence-length`.

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
- Effective kept dim is single-sourced in
  `compression/structured_pruning_methods/pruning_dims.py` (`effective_out_features`);
  `convert_for_export` reads the pruned shapes from the source layers, so a
  hard-pruned model converts/exports with matching shapes.

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
and the coupled cascade live in `compression/structured_pruning.py` (the first
compressor in a group computes the shared mask; siblings reuse it); the module
chains to cascade into are resolved by `CompressionScheme.coupled_target_chains`
(`compression/scheme.py`). The manager only builds groups and iterates `scheme.apply`.

Indexing annotations (`models/*/indexing_*.py`, under a `pruning` key):
`output_dependence` (coupling targets), `coupled_masks` (share a mask within a
block), `coupled_masks_all` (share a mask across all blocks — the residual/hidden
writers), `norm_layers` (transparent hops), `position_linked` (RoPE-indexed rows).

Scope: both MLP and attention (q/k/v) structured pruning are wired end-to-end
(prune → cascade → convert → export). Attention hard pruning changes `head_dim`
(GQA); RoPE projection geometry and the pruned KV-cache are resolved per-kv-group
at conversion time (`blocks/attention/base.py::MHABase.finalize_rope_pruning()`, then
`MHACausal.finalize_cache_geometry()` resizes the internal KV buffers to the pruned
key/value head dims), and a hard-pruned `v_proj` cascades into `o_proj` via
`coupled_repeated_pattern`.
**Never resize module buffers inside `forward()`:** `torch.export` traces the reassignment
as a fresh tensor per call (`aten.full`), not a mutable buffer. That is how head_dim-pruned
mutable-cache exports (QNN, XNNPACK) silently lost their KV cache until 2026-10-13;
`forward()` now only checks the geometry and raises. `io_*` caches (TensorRT) are graph
inputs sized by `cache_shape()` and were never affected.

### `"block_pruning"` — Whole-Block (Attention-Head) Removal

Separate from `structured_pruning`: removes complete row blocks, e.g. whole query heads, instead of
channels (structured pruning on attention shrinks `head_dim`). Code: `compression/block_pruning.py`;
model truth: the indexing's `pruning.block_pruning` (per prunable layer: `owner`, `block_size_attr`,
`blocks_per_unit_attr`, `input_dependence`). Wired for `self_attn.q_proj` in `qwen2_c` and `llama_c`.

| param | default | valid values |
|---|---|---|
| `ratio` | `0.0` | `float` in `[0, 1)` — `int(ratio × blocks_per_unit)` heads removed in **every** KV group |
| `method` | `"magnitude"` | `"magnitude"` (‖q block‖·‖o columns‖), `"random"` (torch RNG) |
| `blocks` | `None` | explicit global head indices to remove (overrides ratio/method); must hit every KV group equally |

```python
manager.set("block_pruning", "blocks", [1, 6], criteria=[["self_attn.q_proj", 0]])  # 1 head per KV group
manager.apply(hard=True)
```

- Uniform per KV group so GQA stays regular; removing a whole group (all its query heads) raises.
- Soft apply zeroes the heads' `o_proj` input columns (zeroed query rows would not silence a head); restore undoes it.
- Hard apply removes q rows (+bias) and `o_proj` columns, sets `self_attn.num_key_value_groups` to the kept
  heads per KV head (HF keeps running), and records `q_proj.block_prune_mask`. `convert_for_export` adopts the head
  count via `blocks/attention/base.py::MHABase.finalize_head_pruning()`; the manifest/cache geometry is unchanged (KV heads stay).
- Not combinable with structured (head_dim) pruning of the same q_proj (raises).

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
| `method` | `"vanilla"` | `"vanilla"` (symmetric max-abs RTN), `"gptq"` (needs `"covariance"` calibration), `"hqq"` (scale-only half-quadratic, calibration-free) |
| `precision` | `"full"` | `"full"`, `"binary"`, or `int` in `[2, 16]` (e.g. `8`, `4`, `2` — NOT the strings `"int8"`/`"int4"`/`"int2"`); hard apply: `int` in `[2, 8]` |
| `granularity` | `"per_tensor"` | `"per_tensor"`, `"per_channel"`, `"per_group"` |
| `group_size` | `128` | positive `int` dividing `in_features` (used by `"per_group"`) |
| `group_scale_bits` | `None` | `None`, or `int` in `[1, 8]`: `per_group` + `vanilla` only; group scales become `channel_scale × m`, `m` an integer in `[1, 2**bits]`, `channel_scale = max group scale / 2**bits` (`m` rounded up: no clipping). `4` = Qualcomm HTP LPBQ, the only group format QNN exports |
| `sparsity` | `0.0` | `float` in `[0, 1)` |
| `sparse_method` | `"magnitude"` | `"magnitude"`, `"random"` |
| `precision_activation` | `"full"` | same as `precision` |
| `method_activation` | `"maxmin"` | `"maxmin"` (static, calibrated hooks), `"dynamic_per_token"` (int8 asym per-token at run time; hard only, no calibration) |
| `scheme_activation` | `"asymmetric"` | `"symmetric"`, `"asymmetric"` |
| `quantize_output_activation` | `True` | `False` = static quantization of the input only (TensorRT W8A8) |
| `eps` | `1e-6` | `float > 0` |

**Hard apply is computed by tsurgeon, not torchao** (`_apply_hard_quantization`).
`METHOD_FUNCTIONS[method]` produces the integers and scales; torchao's
`IntxUnpackedToInt8Tensor` is only the container (scale stored in the weight
dtype, zero point 0). Every exporter reads those exact numbers back
(`compression/quantization.py::export_quant_info`, collected by
`export/quant_info.py::extract_layer_quant_info` → `scale`, `precision`,
`per_channel`, `group_size`, `act_dynamic`), so eager == exported up to kernel
arithmetic. Before 2026-09 the hard path called torchao's own RTN regardless
of `method` (hard GPTQ was silently RTN) — do not reintroduce `quantize_()` there.
LRD layers quantize both factors (U with the covariance projected through V);
LRD+quant used to be exported *float* silently — now a stale pre-LRD scale raises.
`method_activation="dynamic_per_token"` is stored on the weight tensor
(`activation_quantization=INT8_ASYM_PER_TOKEN`), so the eager model simulates
exactly what XNNPACK's qd8 kernels compute.

Per-backend expressibility (see `export/`): TensorRT/ONNX: INT8 per-channel,
INT4 per-channel or per-group (exact block DQ); INT8 per-group raises (no
TensorRT INT8 block DQ). XNNPACK (`quant_path="torchao"`): any of the above for
4/8 bit with dynamic int8 activations. QNN: per-channel only (HTP block quant is
LPBQ, which cannot hold arbitrary group scales; per-group raises).

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

1. **Folder rules** (table above). Adding a model touches only `models/`; a compression method only
   `compression/`; a calibration statistic only `calibration/`; a backend only `export/` (plus new atoms in
   `blocks/` when an operation does not exist yet). Code outside `models/` never branches on a model name.
2. **`Compressor.apply()` always receives a compressible layer** (`LinearCompressed`, ...), never a plain
   `nn.Linear`: listed layers are replaced at model init by `compression/compressible_layers.py::replace_layers_upon_init`.
3. **Compressors never read paths or sentinels.** Topology (block index, `path_template`, `'preprocessing'`/
   `'final_layer'`/`'final_norm'`) is resolved by `CompressionScheme` (`coupled_target_chains`, `resolve_in_block`).
4. **Only `compression/quantization.py` reads the private quantisation state** (`_torchao_*`, `_act_quant_*`, ...);
   everyone else goes through `export_quant_info` / `copy_quant_state`.
5. **`manager.set()` is purely declarative**; the model changes only in `manager.apply()`. Soft apply is reversible,
   hard apply is not.
6. **`blocks/` is a frozen, already-compressed graph.** It never decides what to compress; `convert_for_export`
   reads shapes, ranks and biases from the source layers. Weight-dependent geometry (RoPE pruning projection,
   pruned cache sizes) is resolved once at conversion (`finalize_*` methods), never inside `forward()`;
   never resize buffers in `forward()` (`torch.export` would trace a fresh tensor, not a mutable buffer).
7. **Exporters find special atoms by their `EXPORT_ROLE` class attribute** (`"attention"` on `MHACausal`,
   `"decoder_stack"` on `TransformerDecoder`), never by class name.
8. **`init_vcon()` must be called before `apply()`** when using VCON.
9. **`manager.prepare_for_save()` before `export_to_hf()`** when applying manually (passing `manager=` to
   `export_to_hf` applies and prepares for you). Never call it before a backend export: it upcasts and dequantises.
10. **`path_template` is a format string** with `{block_index}` and `{path}`.
11. **`attn_impl="custom_sdpa"` requires `cache_impl="mutable"` and `q_head_dim == value_head_dim`** (validated).

---

## Anti-Patterns

- **Don't instantiate `CompressionScheme` directly**; the manager builds schemes from the model dictionary.
- **Don't assume `weight.shape == [out, in]` after LRD**: `weight` is `[out, rank]`, `linear_V.weight` is `[rank, in]`.
- **Don't call `restore()` after `hard=True`.**
- **Don't put model-specific logic outside `models/`** (not even in comments: describe the feature, e.g. "a
  composite embedding module", not the model).
- **Don't hard-code an implementation choice in an exporter**: add it to `BACKEND_OPERATIONS`.
- **Don't use `"standard"` calibration mode with `"aa-svd"`** (needs `manager.set_calibration_mode("cascade")`), and
  don't use `"cascade"` with a family whose dictionary sets `'no_cascade_calibration': True` (`bert_c`, `modernbert_c`).
- **Don't call `reapply_masks()` before `manager.apply()`.**

---

## Extension Recipes

### Adding a model family

1. Create `models/<m>_c/indexing_<m>_c.py` (model dictionary; copy the closest family). Required keys per
   component: `num_blocks_attr`, `path_list`, `path_template`; for export also `structure`, `attn_type`,
   `mlp_type`, `norm_type`, `rope_type` (`"auto"` takes the HF configuration's RoPE type). Encoders also take
   `norm_position`, `use_final_norm` and `layer_scale` (per-channel branch scaling, as in `dinov2_c`).
   `mlp_hidden_dim_attr` can be omitted when the HF configuration has no MLP width (the width then comes from
   the weights).
2. Create `models/<m>_c/family_<m>_c.py` (family dictionary: `indexing`, `hf_config`, `hf_model`,
   `class_names`, `auto_class`, optional `sub_configs`, `checkpoint_key_mapping`) and call `build_family`.
3. Add `__init__.py` re-exporting both modules, and one line to `models/__init__.py::FAMILIES`.
   `mistral_c` was added this way (copy of `llama_c` with the HF classes changed): nothing else is needed.
4. For tests, add a tiny builder to `test/_helpers/model_factory.py::FAMILIES`.

### Adding a compression method variant

One file in `compression/<type>_methods/` and one entry in its `METHOD_FUNCTIONS` (`VALID_METHODS` derives
from it); if it needs statistics, list the summary names in the compressor's `CALIBRATED_METHOD_DICT`.

### Adding a compression type

1. Subclass `compression/abstract.py::Compressor` (`set_calibration_store`, `needs_calibration`, `apply`,
   `restore`, `_to_compress`, `__repr__`; optionally `prepare_for_save`, `check_coupling`).
2. Register it in `compression/registry.py` (`COMPRESSOR_DICT`, `COMPRESSION_REGISTRY` with defaults and
   validators built from `compression/validators.py`, and `APPLICATION_ORDER`).

### Adding a calibration summary / raw-data collector

Subclass `calibration/summaries/base.py::CalibrationSummary` (or `calibration/raw_data/base.py::RawDataCollector`)
and register an instance in the folder's `registry.py`; reference the summary name from the compressor.

### Adding an atom

One class per file in the right `blocks/` subfolder, one entry in `blocks/catalog.py::BLOCK_CATALOG`
(`selected_by: "model"` or `"export"`). If the export chooses it, add it to the backend options in
`export/operations.py::BACKEND_OPERATIONS`.

### Adding a backend

Subclass `export/exporter.py::Exporter` (`backend`, `config_class`, `lower`, optional `check_available`) in
`export/backends/`, decorate it with `@register_exporter`, import it in `export/backends/__init__.py`, and add
the backend's column to every entry of `BACKEND_OPERATIONS`.
