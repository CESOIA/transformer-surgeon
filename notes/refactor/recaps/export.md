# Recap: transformersurgeon/export/

Read fully on 2026-10-28. Line numbers are per file. "(guess)" marks unverified inference.

## 1. Purpose and call flow

Purpose: take a compressed HF model (or already-converted components) and write a device artifact:
ExecuTorch `.pte` (XNNPACK, QNN/HTP), or `.onnx` + `.manifest.json` (+ optional TensorRT `.engine`).
Quantization is driven only by per-layer tags that `compression/quantization.py` sets on `nn.Linear`
(`_torchao_*`, `_soft_quant_*`, `_act_quant_*`); export never takes a precision argument.
There is no exporter class. Each backend is a function `export_with_X(components, *, config)` plus a
dataclass config, looked up in a dict (`registry.py`).

### Common prefix (every backend)
`export_to_backend(model_or_graph, config)` (export.py:78)
1. `config.convert_options` copy; inject `max_cache_len = config.max_seq_len` unless set (92-95).
2. `_resolve_model_components_for_export` (12-76): dict / 3-tuple / HF model -> (embedding, decoder, final_layer, hf_config).
   HF model: embedding and lm_head via `model.indexing["text"]["preprocessing"/"final_layer"]` else
   `get_input_embeddings()` / `lm_head`; decoder via `utils.convert_for_export(model, options)["text"]`
   (or the model itself if already a `TransformerDecoder`).
3. `_normalize_component_devices` -> everything `.to("cpu")` (106); optional cast to `config.float_type` (108-111).
4. `EXPORT_ROUTINES[config.backend]["export"](dict, config=config)` (120).
Then in each backend: `common.resolve_components_and_wrapper` (265) -> `LLMWrapper` + example inputs
(`input_ids (1,)`, `pos_id [1]`, + zero KV caches when `cache_impl != "mutable"`) -> `extract_layer_quant_info(wrapper)` (321;
mutates the model: dequantizes hard-quant weights, removes act fake-quant hooks) -> backend-specific lowering ->
`finalize_export_result` (1106): optional weight-mismatch check, verbose inference stats, writes `<stem>.manifest.json`
(`build_llm_manifest`), returns a `BackendExportResult` subclass.

### XNNPACK (`export_with_xnnpack`, xnnpack_export.py:100) -> `.pte`
common prefix -> `quant_path`:
- `"pt2e"` (default): `torch.export` -> if layer_info: `build_quantizer_from_layer_info` (executorch_exporters/common.py:69) ->
  `prepare_pt2e` -> `calibrate_pt2e_observers` -> `inject_scales_into_pt2e_observers` -> `convert_pt2e` -> `torch.export` ->
  `to_edge_transform_and_lower(partitioner=[XnnpackPartitioner()])` -> `to_executorch(MemoryPlanningPass(alloc_graph_input=False))` -> write bytes ->
  `finalize_executorch_result` (weight-mismatch check on the quantized program; stats via ExecuTorch `Runtime`).
- `"torchao"` -> `_export_torchao` (191): `_torchao_quantize_linears` (swap weights for `IntxUnpackedToInt8Tensor`, 8daXw) -> `torch.export` (no PT2E) ->
  partitioners `[XnnpackDynamicallyQuantizedPartitioner(), XnnpackPartitioner()]`, `_check_ir_validity=False` -> same write; mismatch check off.

### QNN (`export_with_qnn`, qnn_export.py:484) -> `.pte`
`is_qnn_available()` guard -> common prefix -> LPBQ detection (any `group_size`) => `_untie_shared_weights` -> `torch.export` under `inference_mode` ->
`_decompose_wrap_with_set_grad_enabled`, `_canonicalize_pow_tensor_scalar` (in-place fx fixes) -> if `layer_info` or `quantize_float_model`:
build `QnnQuantizer` (HTP, chipset), per-QuantDtype layer predicates, block-size map, optional discard nodes, optional
`_split_float_linear_bias` -> `prepare_pt2e` -> calibrate -> inject scales -> `convert_pt2e` -> `torch.export` ->
optional `copy.deepcopy` as `result.reference_module` -> `generate_htp_compiler_spec` / `generate_qnn_executorch_compiler_spec` ->
optional graph sharding pass (`num_shards>1`) -> `_BreakQuantAttrsBeforeFp16Linear` pass injected into QNN pass manager (quantized exports) ->
`to_edge_transform_and_lower_to_qnn(convert_linear_to_conv2d = lpbq or _resolve_linear_to_conv2d)` ->
`to_executorch(MemoryPlanningPass(alloc_graph_input=True, alloc_graph_output=True))` -> write -> `finalize_executorch_result`.

### ONNX (`export_with_onnx`, onnx/onnx_export.py:135) -> `.onnx` (+ `.onnx.data`, `.manifest.json`)
common prefix -> checks (mutable cache rejected; `max_input_len>1` needs `io_inplace`; `add_batch_dim` rejected; `io_paged` needs a translation) ->
optional dynamic `seq` dim -> `torch.onnx.export(dynamo=True, opset=24, custom_translation_table)` ->
`apply_weight_quantization` / `apply_activation_quantization` (rewrite the ModelProto in place, Q/DQ) -> `build_llm_manifest(+onnx extras)` ->
`graph_passes` callbacks `(model, manifest)` -> `onnx.save_model(external data)` -> `onnx.checker` -> `finalize_export_result(manifest=...)`.

### TensorRT (`export_with_tensorrt`, tensorrt/tensorrt_export.py:255) -> ONNX + manifest + `.engine`
Builds the `graph_passes` / `custom_translations` list from config flags (final order:
`[fp32_attention_scores?, user passes, edgellm_int4_pass?, edgellm_attention_pass, fp32_accumulate_lrd?]`), calls `export_with_onnx`, then
`build_engine_from_manifest` (manifest -> `engine_profiles` -> `engine.build_engine`: strongly-typed ONNX parse + builder) if `build_engine`.
Runtime side (not export): `TensorRTLLMSession` (session.py) drives engine from the manifest. CLI: `python -m ...tensorrt_export manifest.json`.

### Shared vs different
Shared: steps 1-4 of the prefix, wrapper/example inputs, `extract_layer_quant_info`, PT2E calibrate/inject (XNNPACK+QNN only), finalize/manifest.
Differ: quantizer construction, graph lowering, memory planning, artifact writing (3 near-identical `open(...).write(buffer)`), runtime-stats function,
ONNX/TRT do Q/DQ by editing the ONNX proto instead of PT2E.

## 2. File tree (lines)

```
export/
  __init__.py (1)            from .export import *; from .config import *
  config.py (49)             BackendExportConfig
  export.py (125)            export_to_backend + model resolution
  registry.py (25)           EXPORT_ROUTINES dict
  common.py (1180)           wrapper, quant-info extraction, PT2E scale injection, calibration, mismatch check, manifest, finalize
  linear_quantizer.py (110)  backend-agnostic linear-only PT2E quantizer
  executorch_exporters/
    __init__.py (1)          from .xnnpack import *; from .qnn import *
    common.py (196)          ExecuTorch result/config bases, XNNPACK qconfig builder, runtime stats, re-exports of ../common
    xnnpack/__init__.py (1)  xnnpack/xnnpack_export.py (226)
    qnn/__init__.py (1)      qnn/qnn_export.py (822)
  onnx/
    __init__.py (20)  onnx_export.py (215)  quantization.py (220)
  tensorrt/
    __init__.py (15)  tensorrt_export.py (174)  engine.py (343)  session.py (140)
    edgellm_attention.py (89)  edgellm_int4.py (138)  attention_passes.py (74)  precision_passes.py (89)
```
Total 4254 lines; `common.py` (28%) and `qnn_export.py` (19%) dominate.

## 3. Per-file API

### config.py
- `BackendExportConfig` (dataclass, 6): `output_path, backend, max_seq_len=2048, weight_mismatch_eps=1e-4, run_weight_mismatch_check=True,
  convert_options={"use_sdpa": False}, check_ir_validity=True, verbose=False, allow_backend_fallback=False, float_type=None`.
- `.from_kwargs(*, output_path, backend, **kw)` (21): strict-key constructor.

### export.py
- `_resolve_model_components_for_export(model_or_graph, *, convert_options, verbose) -> (emb, decoder, final, cfg)` (12-76): HF-attribute-specific.
- `export_to_backend(model_or_graph, config)` (78-121): see flow.

### registry.py
- `EXPORT_ROUTINES = {"xnnpack"|"qnn"|"onnx"|"tensorrt": {"export": fn, "config_class": cls}}`. Eagerly imports all backend packages (they lazy-import heavy deps, so OK).

### linear_quantizer.py
- `_is_annotated(node)` (37), `_mark_annotated(nodes)` (42).
- `annotate_linear(gm, quantization_config, filter_fn=None)` (49): annotate only `aten.linear.default` nodes (act in/out, weight, bias qspecs).
- `LinearOnlyQuantizer(Quantizer)` (89): `set_module_name(name, qconfig)`, `annotate(model)`, `validate`. Used only by XNNPACK (executorch_exporters/common.py:83).
  Docstring says TensorRT also uses it; it does not (stale).

### common.py (generic layer)
Tables: `_PRECISION_TO_QRANGE` {4:(-8,7), 8:(-127,127)} (68); `_PRECISION_TO_EFFECTIVE_DENOM` (76); `SUPPORTED_QUANT_CONFIGS` (83, unused); `_CAL_SENTENCES` (799, 16 English sentences); `MANIFEST_FORMAT="tsurgeon-llm/1"` (1024).
- `BackendExportResult` (97): `output_path, backend, precision, weight_mismatches, inference_stats, manifest_path`.
- `ExporterConfig(BackendExportConfig, ABC)` (114): adds `dynamic_shapes` (always ignored, warns at 308).
- `build_zero_caches(decoder)` (124): zero KV caches per block from `block.attn.cache_shape("key"|"value"|"kv")`, `attn.dtype`.
- `LLMWrapper(embedding, decoder, final_layer)` (145): `forward(input_ids, pos_id_tensor, key_caches=None, value_caches=None)` with 3 signatures keyed on `decoder.cache_impl`
  (mutable / io_* / io_paged); `_head` last-token slice, rank-2 trick for QNN (195-211); reads `decoder.add_batch_dim`.
- `_normalize_component_devices(emb, dec, final, device="cpu")` (214).
- `build_wrapper(components, *, model_config)` (228; `model_config` unused). `build_example_inputs(model_config, *, config, add_batch_dim)` (243; vocab default 100; the `pos_id > max_seq_len` check is dead since pos is fixed at 1).
- `resolve_components_and_wrapper(model_or_graph, *, config) -> (wrapper, model_config, example_inputs)` (265).
- `extract_layer_quant_info(wrapper) -> {layer: info}` (321) via `_extract_weight_quant_info` (362) and `_extract_act_quant_info` (442).
  info keys: `scale, precision, hard, per_channel, group_size, group_scale_bits, act_dynamic, act_scale/zero_point/precision/scheme, act_out_*`.
  Raises if the stashed scale rows mismatch the weight (quantized before LRD).
- `_layer_names_from_observer` (478, 53 lines, BFS depth 6 over fx users, prints diag), `_layer_names_from_output_observer` (534), `_LINEAR_TARGETS` (566).
- `inject_scales_into_pt2e_observers(prepared_model, layer_info) -> int` (569-674) **105 lines, flag**: classifies each observer as weight / input-act / output-act, patches `calculate_qparams`; raises if any layer unmatched.
- `_inject_weight_observer` (677), `_inject_act_observer(observer, info, use_out)` (709-792, 84 lines, flag; clips asymmetric act range to 10x, rescales unsigned->signed zp).
- `calibrate_pt2e_observers(prepared, model_config, config, example_inputs=None)` (819-890, 72 lines): teacher-forced decode over `config.calibration_token_ids` or tokenizer-encoded `_CAL_SENTENCES` (`AutoTokenizer.from_pretrained(model_config._name_or_path)`), else 32 random single-token passes.
- `dequant_from_exported_state_dict` (897), `find_weight_mismatches(wrapper, quantized_exported, eps)` (924; shape-based matching).
- `to_device` (987, unused), `reference_logits` (999), `logit_error_stats` (1007).
- Manifest: `manifest_path_for` (1027), `build_llm_manifest(wrapper, model_config, *, backend, artifact, **extra)` (1032), `load_llm_manifest` (1081), `zero_caches_from_manifest` (1093).
- `finalize_export_result(*, output_path, backend, precision, wrapper, exported_for_mismatch, run_weight_mismatch_check, weight_mismatch_eps, verbose, inference_stats_fn, result_cls, model_config, manifest)` (1106-1180, 75 lines).

### executorch_exporters/common.py
- Re-exports 19 names from `..common` (26-44, "backward compat").
- `ExecuTorchExportResult(BackendExportResult)` + `.pte_path` (52). `ExecutorchExporterConfig(ExporterConfig)` (60; empty).
- `build_quantizer_from_layer_info(layer_info)` (69): per layer `get_symmetric_quantization_config(is_per_channel, is_dynamic = act_scale is None, weight_qmin/qmax)`; raises on `group_size`.
- `run_simple_inference_stats(wrapper, pte_path, inputs)` (115): runs `.pte` in ExecuTorch Runtime, flattens nested cache args. `_get_method_names` (162).
- `finalize_executorch_result(...)` (169).

### xnnpack/xnnpack_export.py
- `XNNPACKExportConfig(ExecutorchExporterConfig)` (22): `quant_path: "pt2e"(default)|"torchao"`.
- `_torchao_quantize_linears(wrapper, layer_info) -> {layer: granularity}` (36-97): int weights recomputed from float weight and tsurgeon scale (per_tensor / per_channel / group{g}); bits 2-8; dynamic int8 per-token activation if `act_dynamic`; rejects static act calibration.
- `export_with_xnnpack` (100-188, 89 lines, flag), `_export_torchao` (191-226).

### qnn/qnn_export.py
- `is_qnn_available()` (24; needs `executorch`, `cpuinfo`, `QNN_SDK_ROOT|QUALCOMM_SDK_ROOT`), `_QNN_UNAVAILABLE_MSG` (45).
- `_is_wrap_with_set_grad_enabled` (53), `_decompose_wrap_with_set_grad_enabled(gm)->int` (62-129, 68 lines; uses QNN `merge_decomposed_graph`).
- `_BreakQuantAttrsBeforeFp16Linear` (132, edge pass class): insert `_to_copy(fp16)` before fp16 linears whose input carries `QCOM_QUANT_ATTRS`.
- `_canonicalize_pow_tensor_scalar(gm)` (202): `pow.Tensor_Scalar` -> `pow.Tensor_Tensor` with a buffer.
- `QNNExportConfig` (240): `soc_model="SM8650", is_online_prepare=False, use_fp16=True, num_shards=1, convert_linear_to_conv2d=False,
  quant_scope="linear"|"all", split_float_linear_bias=True, quantize_float_model=False, keep_reference_module=False, float_attention_core=False, calibration_token_ids=None`.
- `_resolve_qcom_chipset` (296), `_resolve_linear_to_conv2d(wrapper, config)` (323; falls back with warning if non-fp32 weights or tied weights),
  `_untie_shared_weights(wrapper)` (389), `_in_layers` (405), `_split_float_linear_bias(gm, quantized_layers)` (411),
  `_linear_block_sizes(gm, block_layers)` (438), `_float_attention_nodes(gm)` (458).
- `export_with_qnn` (484-822) **338 lines, flag (largest function in the folder)**: contains inline `_PREC_TO_QDTYPE`, `_make_layer_predicate`, the whole quantizer setup (529-641), and a nested `SplitGraphBlocks(ExportPass)` class (707-743) plus pass-manager wiring.

### onnx/onnx_export.py
- `_tensor_scatter(cache, update, write_index)` (68; opset-24 `TensorScatter axis=2`), `default_translations()` (74) -> {`tsurgeon::kv_cache_update` -> `_tensor_scatter`}.
- `ONNXExportResult(BackendExportResult)` (+`.onnx_path`) (82). `ONNXExportConfig(ExporterConfig)` (91): `max_input_len=1, opset=24, optimize=True, int4_block_size=None, custom_translations={}, graph_passes=[]`.
- `io_names(num_layers, paged=False)` (121). `export_with_onnx` (135-225, 91 lines, flag).

### onnx/quantization.py
- `_QRANGE` (264), `_SCALE_FLOOR=2**-14`, `_default_int4_block(in_features)` (271; 128/64/32 dividing and `< in_features`), `_find_initializer` (283).
- `apply_weight_quantization(model, layer_info, *, int4_block_size=None)` (290-386, 97 lines, flag; contains a dead `if group_size is not None: pass` at 341-342).
- `_gemm_consumers` (389), `apply_activation_quantization(model, layer_info)` (396-452, 57 lines).

### tensorrt/
- tensorrt_export.py: `DECODE_PROFILE=0, PREFILL_PROFILE=1`, `TensorRTExportResult(+engine_path)` (177), `TensorRTExportConfig(ONNXExportConfig)` (182):
  `build_engine=True, int4_backend="dequantize"|"edgellm_plugin", engine_path, optimization_level, workspace_bytes, timing_cache_path, plugin_libraries, fp32_attention_scores=True, fp32_accumulate_lrd=True`;
  `engine_profiles(manifest)` (216), `build_engine_from_manifest(...)` (226), `export_with_tensorrt` (255), `main(argv)` (283).
- engine.py: `_trt()`, `_TRT_TO_TORCH_DTYPE`, `_logger`, `_file_stream_reader(path, chunk_bytes=64MiB)` (51), `load_plugin_libraries` (74),
  `OptimizationProfile` (81), `EngineBuildConfig` (96), `build_engine(onnx_path, config)->bytes` (118; STRONGLY_TYPED), `TensorRTEngineRunner` (170; bind/allocate_outputs/run/capture_graph/replay), `time_cuda` (319).
- session.py: `TensorRTLLMSession(manifest_path, engine_path=None, *, device, plugin_libraries)` (29): `reset, prefill, step, generate` (greedy, CUDA-graph decode). Runtime code, not export.
- edgellm_attention.py: `PLUGIN_KEY`, `_register_schema` (116), `_attention_plugin` (143), `attention_translations()` (156) -> {`tsurgeon::paged_attention` -> AttentionPlugin}, `edgellm_attention_pass(model, manifest)` (163).
- edgellm_int4.py: `PLUGIN_NAME/ENV/_DOMAIN/_OP/_GROUP=128`, `_edgellm_repack` (212), `_plugin_group_scales` (224), `edgellm_int4_pass(model, manifest)` (234-298, 65 lines), `resolve_plugin_libraries(manifest, explicit)` (301).
- attention_passes.py: `fp32_attention_scores(model, manifest)` (37-89, 53 lines): Cast MatMul->Add->Softmax to fp32. No-op if `AttentionPlugin` present; raises if no pattern.
- precision_passes.py: `_lrd_layer_prefixes` (341), `_weight_source` (349), `fp32_accumulate_lrd(model, manifest)` (359-402, 44 lines): fp32 accumulation for LRD GEMMs.

## 4. Inventory: block type -> backend operation (raw material for the dictionary)

Key finding: `export/` itself encodes **no** attention-kernel/RMSNorm/MLP/RoPE choice for ExecuTorch. Those choices live in `blocks/` and
`utils/convert.py` as `convert_options` keys, and `export/` only forwards them (export.py:92-95, config.py:14). The only per-backend implicit
choices are listed below. There is NO QNN-fused RMSNorm op and NO XNNPACK-specific MLP/RoPE op anywhere in export/ (verified by reading all files).

### 4.1 Choices selected via convert_options (defined in blocks/, listed because the dictionary must own them)
| Block | Option key | Values (default first) | Where defined | Per-backend note |
|---|---|---|---|---|
| Attention kernel | `attn_impl` / `use_sdpa` | `manual` (default; derived from `use_sdpa=False`), `sdpa`, `custom_sdpa`, `paged` | blocks/mha.py:482,577-594 | `custom_sdpa` = ExecuTorch `llama::custom_sdpa` + `update_cache` (requires `cache_impl="mutable"`, fp32 batched caches, q_head_dim==v_head_dim): XNNPACK only, tested only in test/e2e/test_export_pipelines.py:123. `paged` = `tsurgeon::paged_attention`: TRT only (ONNX rejects, onnx_export.py:171). |
| KV cache | `cache_impl` | `mutable` (default), `io_scatter`, `io_concat`, `io_inplace`, `io_paged` | blocks/mha.py:481 | mutable = XNNPACK/QNN (ONNX rejects, onnx_export.py:143). io_inplace = ONNX/TRT (TensorScatter, in-place alias). io_paged = TRT AttentionPlugin only. `max_cache_len` default 2048 -> `config.max_seq_len`. |
| Batch dim | `add_batch_dim` | False | blocks/decoder.py:39 | ONNX/TRT reject (onnx_export.py:150); XNNPACK/QNN accept. |
| RMSNorm | `rmsnorm_prescale`, `rmsnorm_upcast` | True, False | blocks/decoder.py:40-41,184-186 | Not set by export; scripts do not override. |
| Softmax | `masked_softmax` | False | blocks/decoder.py:36 | requires `attn_impl="manual"` (mha.py:591). |

### 4.2 Choices encoded in export/ code
**Wrapper / graph I/O**
- Head: last token only, rank-2 linear input (common.py:209-211) -- chosen to keep QNN `ConvertLinearToConv2d` working; applied to all backends.
- I/O contract by cache_impl: mutable `(ids,pos)->logits`; io_* `(ids,pos,K[],V[])->(logits,K[],V[])`; io_paged `(ids,pos,kv_pools[])->(logits,pools[])` (common.py:179-193). Names `past_key_i/present_key_i/past_value_i/present_value_i/past_kv_i/present_kv_i` hard-coded in common.py:1044-1053 and onnx_export.py:121-132 (duplicate).
- Cache layout in manifest: `{"io_inplace":"BHSD","io_paged":"PAGED"}`, default `SHD` (common.py:1065).

**Compressed linear (LinearCompressed / LRD) -> quantized op**
- Quantization recipe is read from layer tags (common.py:362-437): hard (torchao scale, `precision` 4|8, per_channel | per_group `group_size`, `group_scale_bits`, `act_dynamic`), soft (`_soft_quant_precision`, scale re-derived by calibration), static input act (`_act_quant_*`) and output act (`_act_out_quant_*`).
- LRD: `linear_V` factor visited as its own `nn.Linear`; ONNX act Q/DQ goes on the V factor (quantization.py:421-424); TRT fp32 accumulation targets `<path>.linear_V.weight` (precision_passes.py:342-346).

**XNNPACK**
- Default `quant_path="pt2e"` (xnnpack_export.py:33): symmetric ExecuTorch qconfig per layer; `is_dynamic = (act_scale is None)` i.e. weight-only layers get dynamic activations (executorch_exporters/common.py:100); `is_per_channel = info.per_channel`; qrange 4 or 8 bit; per-group -> NotImplementedError (96-99). Only linears annotated (linear_quantizer.py docstring: stock XNNPACKQuantizer clips attention add/mul/cat).
- Option `"torchao"`: `IntxUnpackedToInt8Tensor` (per_tensor/per_channel/group), 8da4w/8da8w, `INT8_ASYM_PER_TOKEN` activations if `act_dynamic`; dynamic-quant partitioner first then `XnnpackPartitioner` (xnnpack_export.py:36-97,203-207). Weight-only layers warn "may not be delegated".
- Partitioner: `XnnpackPartitioner()` only (152-156). Memory plan `alloc_graph_input=False` (168; zero-copy inputs).

**QNN (HTP)**
- Backend `kHtpBackend`, default SoC `SM8650`, `use_fp16=True`, `online_prepare=False`, `use_multi_contexts = num_shards>1` (qnn_export.py:242-245, 562-565, 664-672).
- `quant_scope` default `"linear"` (271): only compressed linears quantized, `{4: use_16a4w, 8: use_8a8w}` (545-548), rest fp16, per-channel weights; default config replaced by a no-op sentinel (607-614). Option `"all"`: `{4: use_16a4w, 8: use_16a8w}`, `set_default_quant_config(use_16a16w)`, `MinMaxObserver` activations (592-606). LPBQ: precision 4 + `group_scale_bits==4` required (576-583) -> `use_16a4w_block` + `block_size_map (1, group_size)` (584,617).
- `float_attention_core=True` (default False) with scope all: discard (leave fp16) the MHACausal non-linear nodes and decoder-root nodes (458-481, 620-622).
- `split_float_linear_bias=True` (default; scope all only): float biased linears -> `linear` + `add` (279, 411-435, 624-627) to avoid int32 bias overflow.
- `convert_linear_to_conv2d=False` default (260); forced True when LPBQ (509-511, 790; with `_untie_shared_weights`); auto-disabled with warning for non-fp32 or tied weights (323-386).
- fp16 linear after a quantized layer: inserted `_to_copy(fp16)` pass, dep-table hacks (132-199, 760-784).
- Graph fixes always applied: `wrap_with_set_grad_enabled` decomposition (62), `pow.Tensor_Scalar` -> `Tensor_Tensor` (202).
- Sharding (`num_shards>1`): `llama.fallback.default` inserted at layer boundaries found via regex `blocks\.(\d+)` (677-752).
- `quantize_float_model` (False) needs scope all: 16a16w graph with no compression (526-528).
- Memory plan `alloc_graph_input=True, alloc_graph_output=True` (794-799) (opposite of XNNPACK).

**ONNX**
- opset 24, `optimize=True`, `dynamo=True` (onnx_export.py:97-118, 174-185). `tsurgeon::kv_cache_update` -> opset-24 `TensorScatter(axis=2)` (68-78); `io_paged` must be supplied via `custom_translations`.
- Weights: INT8 -> per-channel `DequantizeLinear(axis=0)`; INT4 -> `DequantizeLinear(axis=1, block_size)`; default block = largest of 128/64/32 dividing in_features and `< in_features` (TRT 10.16 NaNs on a single row-wide block), `int4_block_size` option re-derives max-abs per-block scales; per-group (LPBQ-like) INT4 only; soft-quant -> max-abs per-channel scale (quantization.py:290-386). Scale floor 2^-14.
- Activations: only INT8 symmetric per-tensor input Q/DQ; output act quant and asymmetric rejected (409-417).
- Precision label: `"mixed"` if any layer quantized else `"full"`.

**TensorRT**
- Default `int4_backend="dequantize"` (plain DQ->Gemm); option `"edgellm_plugin"`: `Int4GroupwiseGemmPluginV2` for Gemm with N%64==0, K%128==0, transB, scales constant over 128-groups, else falls back per layer (edgellm_int4.py:234-298).
- Attention: default MatMul/Softmax path (fp32 score cast pass `fp32_attention_scores=True`, attention_passes.py:37) for io_scatter/concat/inplace; `io_paged` -> Edge-LLM `AttentionPlugin` (XQA decode, fp16 only, decode-only, `enable_tree_attention=0`, `enable_fp8_kv_cache=0`) (edgellm_attention.py:116-176). The fp32 pass is a no-op when the plugin is present.
- LRD GEMMs: `fp32_accumulate_lrd=True` casts operands to fp32 (precision_passes.py:359).
- Engine: strongly typed, no precision flags, profile 0 decode (seq=1), profile 1 prefill `(1, min(128,max_input_len), max_input_len)` when `max_input_len>1` (tensorrt_export.py:216-223); plugin library from `EDGELLM_PLUGIN_PATH`.
- Prefill: `max_input_len>1` only with `io_inplace`; default 1.

**Calibration recipe (XNNPACK+QNN)**: teacher-forced decode of up to 512 positions of `_CAL_SENTENCES` or `calibration_token_ids`, else 32 random tokens; hard-quant weight scales and stored act scales override observers; act range clipped to 10x asymmetry (common.py:743-752).

## 5. Dependencies

### Imports from elsewhere in transformersurgeon
- export.py: `..blocks.TransformerDecoder`, `..utils.convert_for_export`, `..utils.utils.get_submodule`.
- onnx_export.py:76 lazy `...blocks.kv_cache_ops` (registers `torch.ops.tsurgeon.kv_cache_update`); edgellm_attention.py:157 lazy `...blocks.paged_attention_ops` (registers `tsurgeon::paged_attention`).
- Implicit coupling (no import): attribute names `decoder.blocks[i].attn.{cache_shape, dtype, key_cache, attn_impl}`, `decoder.{cache_impl, max_cache_len, add_batch_dim, config.rope_type}`, `model.indexing["text"]`, and layer tags from compression/quantization.py:415-422,497-501.
- Third-party: torch, torchao (pt2e, IntxUnpacked tensor), executorch (xnnpack, qualcomm, exir), onnx / onnxscript / ml_dtypes / numpy, tensorrt, transformers (`AutoTokenizer`, common.py:858).

### Who imports export/ (whole repo; `transformersurgeon/__init__.py` and `hf/` do NOT import it, benchmarks/ do not)
- `export_to_backend`: scripts/executorch/{xnnpack,qnn}/exporter_function_test.py, scripts/executorch/qnn/full_quant_export.py, scripts/tensorrt/export_and_generate.py, test/e2e/test_export_pipelines.py, test/unit/{test_onnx_tensorrt_export,test_paged_attention,test_qnn_activation_only,test_qnn_bias_split}.py.
- `export.common` (`LLMWrapper`, `build_zero_caches`, `build_wrapper`, `extract_layer_quant_info`, `calibrate_pt2e_observers`, `build_llm_manifest`, `zero_caches_from_manifest`, `load_llm_manifest`): test/unit/{test_attention_semantics,test_rope_variants,test_block_pruning,test_paged_attention,test_onnx_tensorrt_export,test_qnn_bias_split,test_export_fixes}.py, test/e2e/{test_gqa_attention_pruning,test_export_pipelines}.py, scripts/compression/compress_infer_qwen.py:44, scripts/executorch/xnnpack/inference_exported_test.py:9, examples/04_cache_impl_export_interface.py:43.
- `export.executorch_exporters.{xnnpack,qnn}` configs: scripts above, test/e2e/test_export_pipelines.py, test/unit/{test_qnn_*,test_known_bugs}.py (`is_qnn_available`), test_qnn_bias_split.py:50 imports private `_decompose_wrap_with_set_grad_enabled`, `_split_float_linear_bias` from `qnn.qnn_export`.
- `export.onnx` (`ONNXExportConfig, apply_weight_quantization, apply_activation_quantization`): test_onnx_tensorrt_export.py, test_paged_attention.py.
- `export.tensorrt` (`TensorRTExportConfig`, `engine`, `session.TensorRTLLMSession`, `precision_passes.fp32_accumulate_lrd`): same tests, scripts/tensorrt/export_and_generate.py, test/unit/test_precision_passes.py.
- `export.executorch_exporters.common`: imported by nobody outside export/ (only xnnpack/qnn siblings).
- Docs only: docs/api/export/*.md, docs/concepts.md, docs/investigations/*.md, AGENTS.md, README.md.

## 6. Violations

### Model-specific names / special cases
No `if model == "qwen"` branch exists. Qwen appears only in comments/docstrings: attention_passes.py:6,10 (Qwen2.5-1.5B, token 659), precision_passes.py:10, tensorrt_export.py:68, qnn_export.py:264,266,274, common.py:745,876. Move rationale to docs/dictionary notes.
Structural knowledge that is model/block-class specific (should be a block contract, not strings in export):
- qnn_export.py:476-478 `"MHACausal" in type`, `"Linear" in type`, `"TransformerDecoder"` string matching on `nn_module_stack`.
- qnn_export.py:703 regex `blocks\.(\d+)`; 693 `wrapper.decoder.blocks`; common.py:135,1041-1071 `decoder.blocks[i].attn.*`; `block.attn` attribute name.
- onnx/quantization.py:422 and precision_passes.py:342 `.linear_V.weight` naming of LRD factors.
- common.py:867-869 resets caches by buffer name containing `"cache"`; 864 relies on HF `config._name_or_path` and `AutoTokenizer`.
- export.py:82-100 HF attribute protocol (`get_input_embeddings`, `lm_head`, `indexing["text"]["preprocessing"/"final_layer"]`); common.py:1073 `config.rope_type`; 251 `vocab_size`.
- common.py:209-211: head slicing designed around QNN (a backend quirk in generic code).

### Compression logic that belongs in compression/
- `extract_layer_quant_info` / `_extract_weight_quant_info` / `_extract_act_quant_info` (common.py:321-471): reads private tags, dequantizes torchao tensors in place, strips hooks; it is the read-side of compression/quantization.py (tag names duplicated). Better: compression exposes a `QuantizedLinearSpec` (scale, bits, granularity, group_size, act spec) and a `strip_for_export(model)`.
- Quant range / observer math: `_PRECISION_TO_QRANGE`, `_PRECISION_TO_EFFECTIVE_DENOM` (common.py:66-79), `_inject_weight_observer`, `_inject_act_observer` incl. the 10x asymmetry clip heuristic (709-792).
- `calibrate_pt2e_observers` + `_CAL_SENTENCES` (799-890): calibration data/loop; there is already a calibration manager in compression (guess: reusable).
- `_torchao_quantize_linears` (xnnpack_export.py:36-97): integer weight re-derivation from scales.
- onnx/quantization.py:290-452: int quantization of weights (max-abs scale at 339, 369-370; scale floor), act Q/DQ validation.
- qnn_export.py:571-585: LPBQ validity rules (4-bit, `group_scale_bits==4`) belong with the quantization spec validation.

### Layer-manipulation helpers that belong in utils/ (or blocks/)
- `_untie_shared_weights` (qnn:389), `_normalize_component_devices` (common:214), `to_device` (common:987, unused), `_resolve_model_components_for_export` (export.py:12) -> utils.
- fx graph helpers: `_decompose_wrap_with_set_grad_enabled` (62-129), `_canonicalize_pow_tensor_scalar` (202), `_split_float_linear_bias` (411), `_in_layers` (405), `_linear_block_sizes` (438), `_float_attention_nodes` (458), `SplitGraphBlocks` (707) -> a `passes/` or `utils/fx` module.
- ONNX proto helpers/passes (`_find_initializer`, `_gemm_consumers`, `fp32_*`, `_producers`) -> `utils/onnx_graph.py` or `export/onnx/passes.py`.

### Block-definition logic that belongs in blocks/
- `LLMWrapper` (common.py:145-211): assembles embedding+decoder+head and defines the graph I/O contract per `cache_impl` -> a `CausalLM`/`ExportableLM` block.
- `build_zero_caches` (common.py:124): cache geometry; blocks/mha.py:849 already references it in a comment. Likewise `zero_caches_from_manifest`.
- Cache I/O names and layouts (common.py:1044-1065) duplicate `io_names` (onnx_export.py:121); should come from the cache implementation.
- Paged-attention eager contract vs plugin mapping (edgellm_attention.py) is device mapping and may stay, but `_attention_plugin` argument order must mirror blocks/paged_attention_ops.py:52 (coupled).

## 7. Duplicated code

| Duplicate | Locations | Unify as |
|---|---|---|
| model input parsing (dict / 3-tuple) x3 | export.py:12-76, common.py:228-240 (`build_wrapper`), common.py:265-292 | single `ExportComponents` dataclass built once; backends take it |
| `_normalize_component_devices` called twice | export.py:106, common.py:290 | once in generic exporter |
| linear-target set x3 | common.py:491-495, 547-551, 566 | one module constant |
| quant ranges x4 | common.py:68-71, onnx/quantization.py:264, xnnpack_export.py:80-81, common.py:733-737 | one `QuantScheme` helper in compression/ |
| write `.pte` + `finalize_executorch_result` x3 | xnnpack:171-188, 211-224, qnn:801-817 | `ExecuTorchExporter._finish(edge_program)` |
| xnnpack lowering twice (pt2e vs torchao) | xnnpack_export.py:149-172 vs 193-210 | one `_lower(export_for_edge, partitioners, check_ir)` |
| `os.makedirs(dirname(output_path))` | xnnpack:105, qnn:492, onnx:138 | generic exporter |
| PT2E prepare/calibrate/inject/convert/re-export sequence | xnnpack:128-147, qnn:629-641 | `Pt2eQuantFlow.run(exported, quantizer, ...)` |
| cache/io tensor names | common.py:1044-1053, onnx_export.py:121-132, session.py:50-59 (reads manifest, ok) | derive from cache impl once |
| `QCOM_*` imports + `get_capture_program_passes` blocks | qnn:681-690, 761-770, 769-770 | helper `_qnn_pass_job()` |
| `ExporterConfig(ABC)` / `ExecutorchExporterConfig` empty layers | common.py:114, executorch common.py:60 | drop one layer |
| re-export list | executorch_exporters/common.py:26-44 | remove |
| group-granularity naming `group{g}` | xnnpack:71, onnx quantization:332 | shared enum |
| `_linear_targets` ad hoc tuples | qnn:443 `(aten.linear.default,)` | constant |
Other folders (guess, not verified): calibration loop vs compression calibration manager; tag names vs compression/quantization.py:415-422,497-501; `build_zero_caches` vs mha.py cache allocation (mha.py:570-640).

## 8. Stale / unused (grep over repo)
- `registry.py` `"config_class"` (all 4 entries): never read; only `["export"]` is used (export.py:120).
- `BackendExportConfig.from_kwargs` (config.py:21): no caller anywhere. `allow_backend_fallback`: only declared (config.py:18).
- `SUPPORTED_QUANT_CONFIGS` (common.py:83): referenced only by the re-export in executorch_exporters/common.py:27.
- `_PRECISION_TO_EFFECTIVE_DENOM` (common.py:76): only used as `in` check (685); the stated "max_abs = scale*denom" injection no longer exists (observer is monkey-patched).
- `to_device` (common.py:987), `tensorrt/engine.time_cuda` (319): no caller outside definition.
- `ExporterConfig.dynamic_shapes`: always ignored, warns (common.py:308); ONNX uses a local variable of the same name (onnx_export.py:159).
- `build_wrapper(model_config=...)`: unused param (common.py:228). `build_example_inputs` `pos > max_seq_len` check is unreachable (pos fixed 1).
- `check_ir_validity`: honored only by xnnpack pt2e path; torchao path hardcodes False (205); QNN ignores it.
- `linear_quantizer.py` docstring claims TensorRT uses it: false (only XNNPACK).
- `onnx/quantization.py:341-342` `if group_size is not None: pass` dead branch.
- `qnn_export.py:705` `_QCOM_QUANT_ATTRS` alias redundant; `_diag` diagnostics + `print` calls (common.py:655, qnn:627,701).
- `executorch_exporters/common.py` re-exports "for backward compat": nothing outside export/ imports this file; `SUPPORTED_QUANT_CONFIGS`, qrange tables, `LLMWrapper`, `build_wrapper`, `build_example_inputs`, `dequant_from_exported_state_dict`, `find_weight_mismatches` are re-exported but never used from it.
- `QNNExportConfig.calibration_token_ids` declared only for QNN but read via `getattr` for all backends (common.py:853) -> XNNPACK users cannot set it by config class (XNNPACK config rejects unknown kwargs only in `from_kwargs`, which is unused).
- `result.reference_module` / `result.quantized_layers` assigned dynamically to dataclass instances (not declared fields).
- `TensorRTLLMSession` not re-exported by `tensorrt/__init__.py` (tests import `.session`); it is runtime, not export.
- `xnnpack quant_path="torchao"`: no test or script uses it (grep `quant_path` in test/ scripts/ -> none; only AGENTS.md); possibly experimental (guess).

## 9. Test coverage (grep test/)
Many tests require executorch / QNN SDK / TensorRT+GPU / HF download and skip otherwise (test/_helpers/capabilities.py).
- `export_to_backend`: ONNX (test_onnx_tensorrt_export.py:48,68; test_paged_attention.py:86), TRT (GPU), QNN (SDK), XNNPACK (e2e, HF download). Resolve path: dict form heavily; full-HF-model form via TRT tests; **tuple/list form untested**.
- `LLMWrapper`, `build_zero_caches`, `build_wrapper`: heavily (attention_semantics, rope_variants, block_pruning, paged, gqa, onnx).
- `extract_layer_quant_info`: test_qnn_bias_split.py:49, e2e:194 (hard quant). **Soft-quant path, per-group tags, stash-mismatch RuntimeError: untested.**
- `inject_scales_into_pt2e_observers`, `_inject_*`, `_layer_names_*`: only indirectly through XNNPACK/QNN e2e (need executorch); no unit test. **10x clip untested.**
- `calibrate_pt2e_observers`: only the random fallback (test_export_fixes.py:44); tokenizer / `calibration_token_ids` paths exercised in QNN tests only (test_qnn_activation_only, test_qnn_bias_split).
- `find_weight_mismatches`, `dequant_from_exported_state_dict`: **untested**; every test sets `run_weight_mismatch_check=False`.
- `build_llm_manifest`, `zero_caches_from_manifest`: test_export_fixes.py:56-72; `load_llm_manifest`: only scripts. `finalize_export_result`: indirectly via ONNX tests.
- `LinearOnlyQuantizer`/`annotate_linear`, `build_quantizer_from_layer_info`, `run_simple_inference_stats`, `finalize_executorch_result`: only via XNNPACK/QNN e2e.
- XNNPACK: e2e pt2e + custom_sdpa only; `_torchao_quantize_linears`, `_export_torchao`: **untested**.
- QNN: `_split_float_linear_bias` + `_decompose_wrap_with_set_grad_enabled` (test_qnn_bias_split.py), `quant_scope` validation and activation-only (test_qnn_activation_only.py), `is_qnn_available` (test_known_bugs.py:93), e2e export. **Untested**: `_canonicalize_pow_tensor_scalar`, `_BreakQuantAttrsBeforeFp16Linear`, `_untie_shared_weights`, `_resolve_linear_to_conv2d`, `_linear_block_sizes`/LPBQ, `_float_attention_nodes`, sharding, `_resolve_qcom_chipset`.
- ONNX: I/O contract+manifest, mutable rejection, paged rejection, `apply_weight_quantization` (4/8 bit exact), `apply_activation_quantization` (+ rejects). **Untested**: `int4_block_size`, per-group INT4 branch, soft-quant max-abs, `add_batch_dim` rejection, `_default_int4_block` None case.
- TRT: `fp32_accumulate_lrd` (unit, no GPU), engine streaming (test:230), session (4 GPU tests), edgellm int4 and attention plugin (skip without plugin). **Untested**: `fp32_attention_scores` directly (GPU only), `engine_profiles`, `main` CLI, `time_cuda`, `build_engine_from_manifest` options.

## 10. Proposed layered structure

```
export/
  __init__.py            export_to_backend, ExportConfig, get_exporter
  exporter.py            class Exporter (generic, ABC)
  config.py              ExportConfig (generic) + per-backend config dataclasses next to each backend
  components.py          ExportComponents(embedding, decoder, final_layer, config) + from_model/from_tuple/from_dict (replaces 3 parsers)
  result.py              ExportResult (+ pte_path/onnx_path aliases as properties)
  manifest.py            build/load manifest, io names (cache-impl driven)
  op_dictionary.py       the human-readable dictionary (below) + resolve()
  quant/
    layer_spec.py        QuantizedLinearSpec extraction (calls into compression/)
    pt2e.py              prepare/calibrate/inject/convert shared flow; LinearOnlyQuantizer
    calibration.py       calibrate loop (or reuse compression/)
  passes/
    fx.py                fx graph passes (pow, set_grad, bias split, untie)
    onnx.py              ONNX proto passes (qdq, fp32 attn, fp32 lrd)
  backends/
    executorch_base.py   ExecuTorchExporter (to_edge, memory plan, write, runtime stats)
    xnnpack.py           XNNPACKExporter
    qnn.py               QNNExporter (+ qnn_passes.py, qnn_sharding.py)
    onnx.py              ONNXExporter
    tensorrt/            TensorRTExporter(ONNXExporter), engine.py, plugins (edgellm_*.py)
  runtime/session.py     TensorRTLLMSession (not an exporter)
```
Moves out of export/: `LLMWrapper`, `build_zero_caches` -> blocks/; quant tag reading, qrange tables, act/weight observer math -> compression/; `_untie_shared_weights`, device normalization, model resolution -> utils/.

### Generic interface
```python
class Exporter(ABC):
    backend: str
    config_cls: type[ExportConfig]
    def export(self, model_or_components, config) -> ExportResult:   # template method
        comps = ExportComponents.resolve(model_or_components, config)   # convert_for_export + cpu + dtype
        comps = self.apply_ops(comps, config)        # op_dictionary -> convert_options (attn_impl, cache_impl, norm)
        wrapper, inputs = comps.wrap(config)         # LLMWrapper from blocks/
        spec = extract_quant_spec(wrapper)           # compression/
        artifact = self.lower(wrapper, inputs, spec, config)   # backend-specific
        return self.finalize(artifact, wrapper, config)        # manifest + checks + result
    @abstractmethod def lower(...); def validate_config(...); def default_convert_options(self) -> dict
```
Backend classes: `XNNPACKExporter(ExecuTorchExporter)`, `QNNExporter(ExecuTorchExporter)`, `ONNXExporter`, `TensorRTExporter(ONNXExporter)`; registry maps name -> class and config class (use `config_class`, currently dead).
`ExecuTorchExporter` owns: `torch.export`, PT2E flow hook, `to_edge`, memory plan flag, write bytes, Runtime stats.

### Dictionary layout (proposal)
Python module of plain dicts, one entry per block type, each: `default`, `options`, per-backend support, and constraints. Example:
```python
OPS = {
  "attention": {
    "convert_key": "attn_impl",
    "default": {"xnnpack": "manual", "qnn": "manual", "onnx": "manual", "tensorrt": "manual"},   # today: manual everywhere
    "options": {
      "manual":      {"backends": ["xnnpack","qnn","onnx","tensorrt"], "needs_cache": ["mutable","io_*"]},
      "sdpa":        {"backends": ["xnnpack","qnn","onnx"]},
      "custom_sdpa": {"backends": ["xnnpack"], "needs_cache": ["mutable"], "needs": "executorch llm custom_ops"},
      "paged":       {"backends": ["tensorrt"], "needs_cache": ["io_paged"], "device_op": "AttentionPlugin (XQA)"},
    }},
  "kv_cache": {"convert_key": "cache_impl",
    "default": {"xnnpack": "mutable", "qnn": "mutable", "onnx": "io_inplace", "tensorrt": "io_inplace"},   # onnx/trt default is a proposal; today caller must pass it
    "options": {"mutable": ..., "io_scatter": ..., "io_concat": ..., "io_inplace": {"onnx_op": "TensorScatter"}, "io_paged": ...}},
  "norm_rms":   {"convert_keys": ["rmsnorm_prescale","rmsnorm_upcast"], "default": {"*": {"prescale": True, "upcast": False}}, "options": {...}},
  "linear_compressed": {
    "weight_quant": {
       "xnnpack": {"default": "pt2e_symmetric", "options": ["pt2e_symmetric","torchao_8daXw"]},
       "qnn":     {"default": "pt2e_qnn_linear_scope", "options": ["linear_scope","all_scope_16a","lpbq_block"]},
       "onnx":    {"default": "qdq_initializer", "options": ["int8_per_channel","int4_block"]},
       "tensorrt":{"default": "qdq_initializer", "options": ["dequantize","edgellm_int4_plugin"]}},
    "act_quant": {...}, "lrd": {"tensorrt": "fp32_accumulate_lrd"}, "lm_head": {...}},
  "softmax_scores": {"tensorrt": {"default": "fp32_cast_pass", "options": ["none"]}},
  "linear_lowering": {"qnn": {"default": "fully_connected", "options": ["conv2d_1x1"], "forced_by": "lpbq"}},
}
```
Each backend exporter reads its slice of the dictionary at `apply_ops`/`lower`; config fields (`quant_path`, `quant_scope`, `int4_backend`, `convert_linear_to_conv2d`) become dictionary keys with the config only holding overrides. Backend-specific graph fixes (set_grad, pow, `_to_copy` fp16, bias split, sharding) stay as pass lists in the backend, registered by name in the dictionary so they are discoverable.
