# Recap: scripts/

All imports below were grep-verified against `transformersurgeon/` (note: package dir is
`transformer-surgeon/transformersurgeon/`, no `src/`). **No script imports a removed symbol**; every
import resolves today. No script has a test; none is imported by other code (only README/docs mention them).

## 1. File tree (lines)
```
scripts/
  compression/compress_infer_qwen.py        373
  compression/compress_infer_vit.py         315
  summaries/extract_summaries_qwen.py       244
  tensorrt/export_and_generate.py           103
  executorch/qnn/full_quant_export.py       480
  executorch/qnn/exporter_function_test.py  456
  executorch/qnn/run.sh                      17  (LD_LIBRARY_PATH launcher -> exporter_function_test.py)
  executorch/qnn/run_full_quant.sh           17  (same launcher -> full_quant_export.py)
  executorch/qnn/artifacts/*.pte   2 binaries, 4.8 MB + 5.5 MB (qwen2_qnn_sm8850_..._full_w8[_a8].pte)
  executorch/xnnpack/exporter_function_test.py 308
  executorch/xnnpack/inference_exported_test.py 167
  executorch/xnnpack/run_export.sh           74  (6 export configs)
  executorch/xnnpack/run_inference.sh        34  (loops over 6 pte names)
```
(no `__init__.py`, no output dir for summaries; `scripts/summaries/output/` is created at runtime.)

## 2. Per-script detail

### compression/compress_infer_qwen.py
Purpose: load Qwen2, optionally compress via manager (`configure_compression()` hand-edit block), run HF
`generate()`; with `--convert` convert to block model and greedy-decode token-by-token.
CLI: `--model-name` (Qwen/Qwen2-0.5B-Instruct), `--seed 42`, `--num-cal-samples 4`, `--cal-seq-len 512`,
`--hard`, `--prompt`, `--max-new-tokens 60`, `--skip-compress`, `--convert`,
`--cache-impl {mutable,io_scatter,io_concat}`, `--max-cache-len`.
Library API used:
- `transformersurgeon.Qwen2ForCausalLMCompress.from_pretrained(name, torch_dtype=float32)`
- `transformersurgeon.models.qwen2_c.Qwen2CompressionSchemesManager(model)`
- manager: `.set_calibration_data(loader)`, `.set_calibration_loss(fn)`, `.auto_groups()`,
  `.set("structured_pruning", {share_mask,reduce_op,method,granularity,repeated_pattern,ratio}, group="group1")`
  (commented: `set("lrd",...)`, `set("quantization",...)`, `coupled_repeated_pattern`), `.apply(hard=, criteria=None, verbose=True)`
- `transformersurgeon.utils.convert_for_export(model, options={"use_sdpa":False,"cache_impl","max_cache_len"})` -> `converted["text"]`
- `transformersurgeon.export.common.LLMWrapper(embed, converted["text"], lm_head)`; `.cache_impl`, `.decoder` attrs;
  call signatures `wrapper(tok,pos)` (mutable) / `wrapper(tok,pos,kc,vc)->(logits,kc,vc)` (io)
- `transformersurgeon.export.common.build_zero_caches(wrapper.decoder)`
Hardcoded: see 4. Hardcoded group name `"group1"` (auto_groups output ordering) and `granularity=64`.

### compression/compress_infer_vit.py
Purpose: same for ViT CIFAR-100: compress (incl. Conv2d patch-embed), evaluate top-1 on 200 test images;
`--convert` evaluates through converted encoder one image at a time.
CLI: `--model-name` (Ahmed9275/Vit-Cifar100), `--data-root`, `--seed`, `--num-cal-samples 32`,
`--num-test-samples 200`, `--batch-size 16`, `--hard`, `--skip-compress`, `--convert`.
API: `transformersurgeon.ViTForImageClassificationCompress.from_pretrained`;
`models.vit_c.ViTCompressionSchemesManager`; manager `.set_calibration_data`, `.auto_groups`, `.set(...)`
(lrd rank/method, structured_pruning share_mask/reduce_op/method/ratio on `group="group13"`, quantization
precision/granularity), `.apply(hard,criteria,verbose)`; `utils.convert_for_export(model, options={"use_sdpa":False})`
-> `converted["vit"]` encoder; then manual `model.vit.embeddings(pix)` -> `encoder(emb)` -> `model.classifier(hidden[0:1])`.
Docs comment references `coupled_masks_all` in `indexing_vit_c.py` and `preprocessing_conv` sentinel (models layer).
Hardcoded `group13`, which depends on auto_groups numbering for this checkpoint.

### summaries/extract_summaries_qwen.py
Purpose: run calibration and dump `manager.extract_summaries()` to a `.pt`.
CLI: `--model-name`, `--seed`, `--num-cal-samples 4`, `--cal-seq-len 512`,
`--output-path ./output/qwen_summaries.pt` (relative to CWD, not scripts dir; docstring example says
`scripts/summaries/output/...` — inconsistent; "" skips save).
API: `Qwen2ForCausalLMCompress`, `Qwen2CompressionSchemesManager`; manager `.set_calibration_data`,
`.iter_filtered(criteria=)` -> scheme objects with `.path` and `.get_compression_module().bias`,
`.extract_summaries(criteria=, summaries=[input_activation,output_activation,weight,bias], show_progress=True)`.
Docstring references `SUPPORTED_SUMMARIES` (summary names: weight, bias, input/output_activation,
activation_range, output_activation_range, covariance, shifted_covariance, cross_covariance, weight_grad).
Imports `build_wikitext_loader` NOT from the compress script (docstring claims "reused" but it is a copy).

### tensorrt/export_and_generate.py
Purpose: Qwen2 -> ONNX -> TensorRT engine, then benchmark `TensorRTLLMSession.generate`.
CLI: `--model-name`, `--out-dir artifacts/qwen2_trt`, `--max-cache-len 1024`, `--max-input-len 512`,
`--quant {4,8}`, `--int4-backend {dequantize,edgellm_plugin}`, `--no-build`, `--skip-export`, `--prompt`, `--max-new-tokens 64`.
API: `Qwen2ForCausalLMCompress.from_pretrained(name, dtype=float16)` (note `dtype=`, others use `torch_dtype=`);
`Qwen2CompressionSchemesManager.set("quantization", precision|granularity)` + `.apply(hard=True, criteria=["self_attn","mlp"])`;
`transformersurgeon.export.export_to_backend(model, config=)`;
`transformersurgeon.export.tensorrt.TensorRTExportConfig(output_path, backend="tensorrt", max_input_len,
convert_options={cache_impl:"io_inplace", max_cache_len, rmsnorm_prescale:False, rmsnorm_upcast:True},
int4_backend, build_engine)`; result `.onnx_path/.manifest_path/.engine_path`;
`transformersurgeon.export.tensorrt.session.TensorRTLLMSession(manifest).generate(ids, n)` (NOT in
`export.tensorrt.__all__`; imported via `.session`). Docstring cites `python -m transformersurgeon.export.tensorrt.tensorrt_export`.
Tokenizer chat template via `apply_chat_template` (good; no hand-written template).

### executorch/qnn/exporter_function_test.py
Purpose: export Qwen2 to QNN `.pte`, optional MLP-only quantization (+static act calib), diagnostics log of
unsupported aten ops on failure.
CLI: `--model-name`, `--float-type`, `--soc-model SM8850`, `--online-prepare`, `--fp16/--no-fp16`,
`--num-shards 1`, `--max-sequence-length 1024`, `--mode {hf,direct}`, `--out-dir artifacts`, `--verbose`,
`--quant-mlp`, `--quant-precision {4,8}`, `--quant-mode {hard,soft}`, `--granularity`, `--calibrate`,
`--act-precision`, `--num-calibration-samples 128`, `--calibration-seq-len 512`,
`--cache-impl {mutable,io_scatter,io_concat}`, `--rmsnorm-prescale/--no-...` (default on),
`--linear-to-conv2d/--no-...` (default off), `--aten-log-file`.
API: `Qwen2ForCausalLMCompress.from_pretrained(torch_dtype=)`; `Qwen2CompressionSchemesManager`
`.set("quantization", {precision,method,granularity,sparsity,sparse_method,eps,precision_activation})`,
`.set_calibration_data`, `.apply(hard,criteria=["mlp"],verbose)`; `utils.convert_for_export(options={use_sdpa,
cache_impl,rmsnorm_prescale}, verbose=False)`; `export.export_to_backend`; `export.executorch_exporters.qnn.QNNExportConfig(
output_path, backend="qnn", float_type, soc_model, is_online_prepare, use_fp16, num_shards, max_seq_len,
convert_linear_to_conv2d, convert_options, verbose)`; result fields `.pte_path .backend .precision
.weight_mismatches[{matched_weight,max_abs_err,mean_abs_err}] .inference_stats{max_abs_err,mean_abs_err,mse,rmse}`.
`mode=direct` builds the dict `{"embedding","decoder","final_layer","config"}` -> this is an implicit
`export_to_backend` input contract (see 3).
Env: `QNN_SDK_ROOT` (artifact naming). `run.sh` hardcodes `/workspace/qairt/2.37.0.250724/...` and
`/workspace/android-ndk-r27d/...`.

### executorch/qnn/full_quant_export.py
Purpose: like above but ALWAYS quantizes every attn+mlp linear (`criteria=["self_attn","mlp"]`), independent
act toggle. ~90% copy-paste of exporter_function_test.py.
CLI deltas: no `--quant-mlp/--calibrate/--rmsnorm-*/--linear-to-conv2d`; adds `--precision {4,8}`,
`--quant-method {vanilla,gptq}`, `--quantize-activations`, `--act-method maxmin`, `--act-scheme {symmetric,asymmetric}`.
API deltas: manager `.set("quantization", "method"|"method_activation"|"scheme_activation", ...)`;
`QNNExportConfig(...)` WITHOUT `convert_linear_to_conv2d`; `convert_options` omits `rmsnorm_prescale`
(so uses the library default — differs from exporter_function_test default True; check what default is).
Output stem always starts `qwen2_qnn_<soc>_qairt<ver>_full_w<N>[_a<N>][_soft]` (the "full" = unpruned, so
same stem word as exporter_function_test's no-quant case -> filename ambiguity).

### executorch/xnnpack/exporter_function_test.py
Purpose: Qwen2 -> XNNPACK `.pte`, MLP-only quant optional.
CLI: as QNN minus QNN-specific, plus `--backend {xnnpack}`, `--attn-impl {manual,sdpa,custom_sdpa}`.
API: `XNNPACKExportConfig(output_path, backend, float_type, max_seq_len, convert_options{use_sdpa,attn_impl,
cache_impl,max_cache_len}, verbose)` from `export.executorch_exporters.xnnpack`; `export_to_backend`;
`convert_for_export`; same manager calls; result also `.manifest_path`.
Stem: `qwen2_xnnpack_[full|w{N}[_a{N}]_{mode}]`.

### executorch/xnnpack/inference_exported_test.py
Purpose: run a `.pte` with ExecuTorch `Runtime`, iterative prefill + decode, tok/s.
CLI: `--pte-path artifacts/export_hf_xnnpack_auto.pte`, `--model-name Qwen/Qwen2.5-0.5B`, `--prompt`,
`--max-new-tokens 32`, `--temperature 0`.
API: `transformersurgeon.export.common.load_llm_manifest(pte_path)` (reads sidecar; keys used:
`attn_impl`, `cache_io`, `num_layers`, `cache_impl`), `zero_caches_from_manifest(manifest)`;
`executorch.runtime.Runtime`, `executorch.extension.llm.custom_ops.custom_ops` (for custom_sdpa).
Input convention: `method.execute([ids, pos, *kc, *vc])` -> `[logits, *kc, *vc]` (duplicates
`LLMWrapper` calling convention + `generate_converted` in compress_infer_qwen).

### Shell launchers
- `xnnpack/run_export.sh`: 6 runs of exporter_function_test.py (full, w8, a8w8, w4, a8w4, a4w4); env MODEL_NAME/OUT_DIR/MODE.
- `xnnpack/run_inference.sh`: loops over expected pte names `qwen2_xnnpack_{full,w8_hard,w8_a8_hard,w4_hard,w4_a8_hard,w4_a4_hard}.pte`.
  Names match `_build_pte_stem` of xnnpack exporter (OK).
- `qnn/run.sh`, `qnn/run_full_quant.sh`: identical except target script; hardcoded `/workspace/...` SDK paths.

## 3. Logic that duplicates / belongs in the library
1. **WikiText-2 calibration loader** copied 5x with tiny drift: compress_infer_qwen.py:106-134 (adds `labels`),
   extract_summaries_qwen.py:61-83, qnn/exporter_function_test.py:206-232, qnn/full_quant_export.py:248-275,
   xnnpack/exporter_function_test.py:122-148 (all `_build_wikitext_calibration_loader`/`build_wikitext_loader`;
   dataset `EleutherAI/wikitext_document_level` hardcoded). Candidate: `transformersurgeon.calibration` helper
   (library already has a `calibration/` pkg). Also: no `torch.manual_seed`, uses `random.Random(seed)`.
2. **Manager quantization recipe** (precision/method/granularity/sparsity=0/sparse_method="magnitude"/eps=1e-6
   boilerplate): qnn/exporter_function_test.py:235-257, qnn/full_quant_export.py:278-312, xnnpack/exporter_function_test.py:151-173,
   tensorrt/export_and_generate.py:50-55. Candidate: manager convenience (e.g. `quantize_linears(precision, ...)`).
   Setting sparsity/sparse_method/eps to their defaults suggests defaults are non-default or unknown to script author.
3. **ATen-op diagnostics log** (`ATEN_PATTERN`, `_extract_aten_ops_from_text`, `_write_aten_diagnostics_log`,
   try/except/finally around `export_to_backend`): identical in qnn/exporter_function_test.py:261-309,340-382 and
   qnn/full_quant_export.py:315-363,396-438. Belongs in `export/executorch_exporters/qnn` (on failure path).
4. **QAIRT version / pte stem builder** `_qairt_version`, `_build_pte_stem`: qnn two scripts (lines ~160-203 / ~200-243).
   `_qairt_version` is pure QNN SDK logic -> belongs in qnn exporter module.
5. **`--mode direct` pre-conversion** (`convert_for_export` then dict `{"embedding","decoder","final_layer","config"}`):
   qnn/exporter_function_test.py:393-405, qnn/full_quant_export.py:~446, xnnpack/exporter_function_test.py:215-230;
   also `build_converted_wrapper` in compress_infer_qwen.py:230-246 (same pattern -> `LLMWrapper`). The dict shape
   is an undocumented contract of `export_to_backend` input; a `build_llm_parts(model, options)` helper in
   `export.common` or `convert.py` would remove it.
6. **Float-type cast** `_DTYPE_MAP` + `load_dtype` logic duplicated in 3 exporter scripts (library already casts via `float_type`).
7. **Result printing** (pte_path, mismatches top-5, inference_stats) duplicated 3x: qnn 384-430 / 441-487, xnnpack 267-304. Belongs on the result dataclass (`__str__`/`summary()`).
8. **Token-by-token prefill + greedy decode loops** with cache threading: compress_infer_qwen.py:249-292
   (`generate_converted`) and xnnpack/inference_exported_test.py:105-150. Plus two **ChatML templates**
   (compress_infer_qwen.py:56-60, inference_exported_test.py:94-98; the latter differs: extra `\n` before `<|im_end|>`
   and no trailing consistent system text) vs tensorrt script using `tokenizer.apply_chat_template`. Library should
   own generate-on-converted-model (a `generate()` on `LLMWrapper`/a runtime session, like `TensorRTLLMSession`);
   there is no ExecuTorch equivalent of `TensorRTLLMSession`.
9. **ViT converted-eval** compress_infer_vit.py:200-214: manually reproduces embeddings -> encoder -> classifier head
   pipeline (no ViT wrapper analogous to `LLMWrapper`).
10. `print_model_structure` copied 3x (qwen, vit, summaries).
11. Auto-group naming: scripts depend on `manager.auto_groups()` returning "group1"/"group13" – brittle if manager refactor changes numbering.

## 4. Model-specific hardcoding
- `Qwen/Qwen2-0.5B-Instruct` default in: compress_infer_qwen.py:46, extract_summaries_qwen.py:41, tensorrt:34,
  qnn x2, xnnpack exporter, run_export.sh/run_inference.sh (via MODEL_NAME env).
- `Ahmed9275/Vit-Cifar100` (compress_infer_vit.py:44); dataset `uoft-cs/cifar100`; `ViTImageProcessor`.
- **Qwen2.5 mismatch**: inference_exported_test.py:`--model-name` default `Qwen/Qwen2.5-0.5B` (exports use Qwen2-0.5B-Instruct);
  uses `Qwen2TokenizerFast` explicitly (Qwen2-only) and Qwen-only ChatML string. Default pte path
  `artifacts/export_hf_xnnpack_auto.pte` is a stale name (exporter now emits `qwen2_xnnpack_*`).
- `Qwen2ForCausalLMCompress` / `Qwen2CompressionSchemesManager` directly in all LLM scripts; layer names
  `self_attn`, `mlp`, `mlp.gate_proj/up_proj/down_proj`, `q_proj` in criteria; `granularity=64` = Qwen2-0.5B head_dim,
  GQA `group_size=7` in comments (14 heads/2 KV).
- Soc/HW: `SM8850` default (qnn scripts); `soc_model` lowercase in filename; `/workspace/qairt/2.37.0.250724`,
  `/workspace/android-ndk-r27d` in run*.sh; `qwen2_qnn_` filename prefix hardcoded despite `--model-name`.
- TensorRT: `rmsnorm_prescale False`, `rmsnorm_upcast True`, `cache_impl io_inplace` hardcoded (Jetson/Edge-LLM tuning).
- Calibration dataset `EleutherAI/wikitext_document_level`, "wikitext-2-raw-v1", seed 42.
- Memory note (QNN/backend unification): QNN-vs-XNNPACK config drift is real here — QNN `full_quant_export.py`
  quantizes attn+mlp, `exporter_function_test.py` (both backends) only `mlp`, tensorrt attn+mlp w/ no calibration;
  QNN scripts set `rmsnorm_prescale`/conv2d, xnnpack has `attn_impl`; `use_sdpa: False` hardcoded everywhere.

## 5. Stale / superseded / one-off
Verified-existing imports (no breakage): `Qwen2ForCausalLMCompress`, `ViTForImageClassificationCompress`
(models/qwen2_c/define_qwen2_c.py:22, vit_c/define_vit_c.py:22, via `transformersurgeon.__init__` star import),
`*CompressionSchemesManager` (define_*_c.py:30/:56), `utils.convert_for_export` (utils/convert.py:269),
`export.export_to_backend` (export/export.py:78), `export.common.{LLMWrapper:145,build_zero_caches:124,
load_llm_manifest:1081,zero_caches_from_manifest:1093}`, `QNNExportConfig`, `XNNPACKExportConfig`,
`TensorRTExportConfig`, `TensorRTLLMSession` (export/tensorrt/session.py:29), manager methods
(set:143, auto_groups:319, set_calibration_data:527, set_calibration_loss:544, extract_summaries:626, apply:701,
iter_filtered:1175), `scheme.get_compression_module` (utils/scheme.py:275).
Evidence of staleness / problems:
- **README.md:147-148 references `scripts/compression/prune_qwen.py` and `prune_quant_export_tensorrt.py` — neither
  exists** (broken doc refs; `ls scripts/compression` shows only the 2 compress_infer files). README:185 / FRAMEWORK_STRUCTURE:159 refs to tensorrt script are valid.
- `xnnpack/inference_exported_test.py`: default `--pte-path artifacts/export_hf_xnnpack_auto.pte` and
  `--model-name Qwen/Qwen2.5-0.5B` are stale defaults (exporter writes `qwen2_xnnpack_*.pte`; Qwen2 vs 2.5); error
  message tells user to run exporter_function_test.py (fine). No QNN analogue (QNN runs on-device elsewhere).
- `qnn/exporter_function_test.py` vs `qnn/full_quant_export.py`: ~90% overlap; full_quant_export is a superset in
  quantization (all layers, gptq, activation scheme) but lacks `--rmsnorm-prescale` and `--linear-to-conv2d` that
  exporter_function_test has. Consolidation candidate (single QNN script with `--quant-scope {none,mlp,all}`). Names
  `*_function_test.py` are not tests (no assertions/pytest) — they are export CLIs; real tests live in `test/e2e/test_export_pipelines.py`.
- `qnn/artifacts/*.pte` (10 MB of committed binaries, sm8850 qairt2.37.0.250724): one-off outputs, should be gitignored/removed;
  also repo-root `artifacts/aten_op_problems_qwen2_qnn_sm8650_full.log` is a leftover from the diagnostics log
  (sm8650 = old default; library `QNNExportConfig.soc_model` default is `SM8650`, scripts override to SM8850).
- `extract_summaries_qwen.py`: one-off inspection utility; default output `./output/...` is CWD-relative
  (docstring path mismatch); no sibling for other models; summary of calibration API better served by `examples/`
  (see `examples/02_calibrated_lrd_svd_llm_v2.py`, `01_basic_lrd.py`, `03_hf_export_roundtrip.py`, `04_cache_impl_export_interface.py`
  — a parallel set of demo scripts outside scripts/; compress_infer_qwen.py overlaps with 01/02).
- compress_infer_qwen.py docstring says "Full parameter reference: transformer-surgeon/AGENTS.md" (exists). Its
  `--convert` path uses `cache_impl` values {mutable, io_scatter, io_concat}; TensorRT script uses `io_inplace`
  which is NOT in the choices of any script's `--cache-impl` -> check `utils/convert.py` valid values when refactoring.
- `--backend {xnnpack}` arg in xnnpack exporter has a single choice (vestigial from a multi-backend script).
- Mixed `dtype=` (tensorrt) vs `torch_dtype=` (others) in `from_pretrained`: HF deprecates `torch_dtype`; pick one.
- No script is referenced by CI/tests (grep `scripts/` outside scripts/ hits only README, FRAMEWORK_STRUCTURE, docs/investigations/plan-custom-sdpa-xnnpack.md).

## Refactor checklist (APIs these scripts will break on if moved)
`transformersurgeon.{Qwen2ForCausalLMCompress,ViTForImageClassificationCompress}`,
`transformersurgeon.models.{qwen2_c,vit_c}.*CompressionSchemesManager`,
`transformersurgeon.utils.convert_for_export` (options keys: use_sdpa, cache_impl, max_cache_len, attn_impl,
rmsnorm_prescale, rmsnorm_upcast; returns dict with keys "text"/"vit"),
`transformersurgeon.export.{export_to_backend}`, `export.common.{LLMWrapper,build_zero_caches,load_llm_manifest,
zero_caches_from_manifest}`, `export.executorch_exporters.{qnn.QNNExportConfig,xnnpack.XNNPACKExportConfig}`,
`export.tensorrt.{TensorRTExportConfig, session.TensorRTLLMSession}`, manager `set/apply/auto_groups/
set_calibration_data/set_calibration_loss/extract_summaries/iter_filtered`, scheme `.path`/`.get_compression_module()`,
result dataclass fields (`pte_path, manifest_path, backend, precision, weight_mismatches, inference_stats,
onnx_path, engine_path`), manifest keys (`attn_impl, cache_io, cache_impl, num_layers`).
