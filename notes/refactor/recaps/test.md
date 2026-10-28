# Recap: test/ (+ pytest.ini)

Note: pytest is NOT installed in base python or in conda env py312_torch, so `--collect-only` could not run. Test lists below come from reading sources (parametrization counts not expanded). Docs referenced by tests live in docs/investigations/ (FRAMEWORK_PROBLEMS.md etc.).

## 1. Structure

pytest.ini (repo root): testpaths = `test/unit test/e2e`; python_files `test_*.py *_test.py`; addopts `-ra --strict-markers --import-mode=importlib`.
Markers registered: unit, e2e, slow, download, gpu, executorch, tensorrt, qnn, xfail_known_bug. (`gpu`, `executorch`, `tensorrt`, `xfail_known_bug` are registered but NEVER applied by any test; gating is done with skipif objects instead.)

```
test/
  conftest.py                         40
  .gitignore                           3   (*.pte *.pt *.yml tmp)
  _helpers/__init__.py                13   (exports capabilities, model_factory)
  _helpers/capabilities.py           108
  _helpers/model_factory.py          200
  unit/   (18 files, 2 003 lines)
    test_add_batch_dim 154, test_attention_semantics 60, test_block_pruning 101,
    test_cascade_calibration 84, test_export_fixes 79, test_extract_summaries 89,
    test_gqa_rope_pruning 153, test_io_inplace_cache 100, test_known_bugs 156,
    test_lpbq_quantization 60, test_mha_causal_custom_sdpa 116, test_onnx_tensorrt_export 290,
    test_paged_attention 139, test_precision_passes 33, test_qnn_activation_only 75,
    test_qnn_bias_split 120, test_repeated_pattern 126, test_rope_variants 71, test_smoothquant 78
  e2e/    test_export_pipelines 216, test_gqa_attention_pruning 191, test_model_families 160
  compression_tests/  executorch_tests/{simple,exporter_function/{qnn,xnnpack}}/  hf_export_tests/
        -> contain ONLY __pycache__ (see section 4)
```
Total ~3 015 lines of .py. No `__init__.py` in test/, unit/, e2e/ (importlib mode + sys.path hack).

### Import plumbing
conftest.py inserts `test/` on sys.path so `from _helpers import capabilities` / `from _helpers.model_factory import ...` works in every file. (Docstring in capabilities.py shows `from test._helpers.capabilities import ...` which is NOT what tests use.)

### Capability gating (_helpers/capabilities.py)
Flags computed once at import via `importlib.util.find_spec` (no real import) :
- HAS_CUDA (torch.cuda.is_available), HAS_EXECUTORCH (find_spec executorch), HAS_QNN (executorch AND env QNN_SDK_ROOT or QUALCOMM_SDK_ROOT; no import probe), HAS_ONNX (onnx + onnxscript), HAS_TENSORRT (tensorrt pkg AND HAS_CUDA), HAS_EDGELLM_PLUGIN (HAS_TENSORRT + `tensorrt_edgellm` pkg + env EDGELLM_PLUGIN_PATH is a file), HAS_EXECUTORCH_CUSTOM_SDPA (find_spec executorch.extension.llm.custom_ops.custom_ops), HF_OFFLINE (env HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1).
- skipif marks: requires_cuda, requires_executorch, requires_onnx, requires_tensorrt (ONNX and TENSORRT), requires_edgellm_plugin (ONNX and plugin), requires_qnn, requires_custom_sdpa, requires_hub (not HF_OFFLINE).
- `summary()` string printed by conftest `pytest_report_header` ("transformer-surgeon capabilities: cuda=.. executorch=.. ...").
- Mixed style: some tests also do runtime `pytest.importorskip("executorch")` / `is_qnn_available()` skips.

### conftest.py fixtures
- `torch_device` (session): cuda:0 if available else cpu. (grep: not used by any test.)
- `out_dir(tmp_path) -> str`: creates `tmp_path/artifacts`, returns path str. Used by e2e/test_export_pipelines.

### _helpers/model_factory.py
- `@dataclass FamilySpec(name, build() -> nn.Module, manager(model) -> *CompressionSchemesManager, sample_inputs(model) -> dict, lrd_criteria, mlp_prune_criteria, is_causal=False, known_broken="" (xfail reason; currently unused/empty for all), struct_prune_supported=True)`.
- Builders (tiny random weights, vocab 256, hidden 64): `_qwen2()`, `_llama()`, `_bert()`, `_modernbert()`, `_distilbert()`, `_vit()` (32x32 pixels), `_make_vl(model_attr, vis_attr, txt_attr, cfg_attr)` for qwen2_vl / qwen2_5_vl. All go through `transformersurgeon` top-level names (`T.Qwen2ForCausalLMCompress`, `T.Qwen2ConfigCompress`, ...).
- `_ids(n=8)` (random input_ids), `_pixels`, `_mgr(attr)` (manager factory by class name).
- `FAMILIES: Dict[str, FamilySpec]` (qwen2, llama, bert, modernbert [struct_prune_supported=False], distilbert, vit, qwen2_vl, qwen2_5_vl), `ALL_FAMILY_NAMES`, `CAUSAL_FAMILY_NAMES` (qwen2, llama; the VL families are is_causal=False).
- Private builders `_qwen2` is imported directly by test_export_fixes and test_onnx_tensorrt_export.

## 2. Per-file test inventory
Legend: DL = downloads model, GPU, BE = needs backend. Unless stated "none": no download, no GPU, no backend; all tiny random models run on CPU.

### unit/
**test_add_batch_dim.py** (none) - blocks.mha.MHACausal, blocks.decoder.TransformerDecoder, blocks.config.CustomDecoderConfigCompress, blocks.rope (per-kv-group pruned RoPE helpers)
- test_add_batch_dim_matches_unbatched[manual|sdpa|custom_sdpa]: batched MHACausal == unbatched numerics (custom_sdpa variant needs executorch custom ops, check inside file).
- test_add_batch_dim_matches_unbatched_under_pruned_rope: same parity with per-group pruned RoPE.
- test_transformer_decoder_add_batch_dim_output_shape_and_parity: decoder-level shape + parity.

**test_attention_semantics.py** (none) - T (Qwen2 pruned), export.common.LLMWrapper, utils.convert_for_export
- test_within_head_pruning_keeps_score_scale: hard-pruned converted decoder == soft-pruned HF (1/sqrt(head_dim) preserved).
- test_masked_softmax_equals_additive_mask: masked_softmax option equals additive mask.

**test_block_pruning.py** - compression.block_pruning (BlockPruner via manager "block_pruning"), LLMWrapper, convert_for_export
- test_soft_equals_hard_equals_converted[qwen2|llama]: soft (zeroed o_proj cols) == hard == converted decoder with fewer heads.
- test_soft_restore_and_head_removal_changes_output; test_ratio_removes_uniformly_per_group[magnitude|random]; test_rejects_uneven_groups_and_other_layers.

**test_cascade_calibration.py** [unit] - utils.cascade.apply_cascade via manager, FAMILIES
- test_bert_cascade_calibration_raises_clear_error / test_modernbert_...: `no_cascade_calibration` indexing flag raises clear error.
- test_qwen2_cascade_calibration_still_works.

**test_export_fixes.py** (none) - blocks.rope.rope_theta_from_config, decoder norms eps, export.common.calibrate_pt / manifest (export_to_backend with ONNX? -> manifest per layer)
- test_rope_theta_from_config (rope_parameters / rope_theta / default 1e6).
- test_converted_decoder_uses_config_eps_and_rope_theta.
- test_random_calibration_positions_stay_in_range.
- test_manifest_describes_per_layer_caches[io_scatter|io_inplace]: per-layer cache manifest content (uses tmp_path; ONNX/export_to_backend; check file for requires_onnx gating - none visible at decorator level, so may fail without onnx).

**test_extract_summaries.py** [unit] - utils.manager.extract_summaries, calibration.summaries.*
- test_extract_summaries_returns_nested_dict_for_all_kinds; ..._bias_raises_clear_error_when_module_has_no_bias; ..._requires_nonempty_summaries; ..._rejects_unknown_summary_name.

**test_gqa_rope_pruning.py** (none) - blocks.mha.MHACausal/MHAEncoder, blocks.rope (derive_per_kv_group_patterns, build projection)
- test_derive_per_kv_group_patterns_distinct; test_build_projection_selects_per_group_frequencies; 4 x test_projection_rejects_* (split real/imag, heads disagree in group, unequal counts, q/k mismatch).
- test_encoder_forward_with_per_group_pruned_rope[use_sdpa]; test_causal_pruned_kv_cache_dims[mutable|io_scatter|io_concat].

**test_io_inplace_cache.py** (none) - blocks.kv_cache_ops.kv_cache_update, TransformerDecoder (cache_impl="io_inplace")
- test_kv_cache_update_is_functional_and_writes_rows; test_io_inplace_matches_io_scatter_token_by_token[manual|sdpa]; test_io_inplace_multi_token_prefill_matches_single_token_steps.

**test_known_bugs.py** [unit] - manager (bert), compression.quantization validator, coupled pruning guards, export.executorch_exporters.qnn (is_qnn_available, export_with_qnn), manager.create_group/auto_groups
- test_bert_manager_builds; test_quantization_rejects_int8_string_with_clear_error; test_quantization_precision_int_is_the_real_api; test_partial_coupled_gate_prune_is_guarded; test_both_coupled_gate_up_pruned_independently_is_guarded; test_qnn_unavailable_raises_clear_error... ; test_cross_tower_group_is_rejected; test_auto_groups_still_works_per_tower; test_create_group_rejects_scheme_already_in_another_group.
- NB: despite docstring, no `xfail` marker exists any more - all are plain assertions (bugs fixed). test_qnn_unavailable... hard-asserts `is_qnn_available() is False` and FAILS on a QNN-equipped machine (not gated).

**test_lpbq_quantization.py** (none) - compression.quantization_methods.vanilla.{lpbq_group_scales, quantize_vanilla}, manager (group_scale_bits)
- test_lpbq_scales_are_integer_multiples_of_channel_scale; test_zero_rows_stay_finite; test_manager_hard_lpbq_and_validation.

**test_mha_causal_custom_sdpa.py** [pytestmark=requires_custom_sdpa -> needs executorch llm custom ops] - blocks.mha.MHACausal
- test_causal_custom_sdpa_matches_manual; test_custom_sdpa_requires_mutable_cache_impl; test_custom_sdpa_rejects_unknown_attn_impl; test_custom_sdpa_rejects_pruned_q_head_dim_mismatch.

**test_onnx_tensorrt_export.py** - export.onnx (ONNXExportConfig, apply_weight_quantization, apply_activation_quantization), export.tensorrt (TensorRTExportConfig, session.TensorRTLLMSession, engine), export_to_backend, extract_layer_quant_info
- requires_onnx: test_onnx_export_io_contract_and_manifest; test_onnx_export_rejects_mutable_cache; test_weight_quantization_pass_is_exact[4|8]; test_activation_quantization_pass_places_qdq_on_gemm_input; test_activation_quantization_pass_rejects_non_tensorrt_schemes[2 params].
- requires_tensorrt (BE: TensorRT + CUDA): test_tensorrt_w8a8_static_activations_match_eager[smooth F/T]; test_tensorrt_session_matches_eager; test_engine_file_is_streamed_in_chunks (monkeypatch engine_mod).
- requires_edgellm_plugin: test_edgellm_int4_plugin_matches_dequantize_backend.

**test_paged_attention.py** - blocks.paged_attention_ops.KV_PAGE_SIZE (+op via decoder), export.common.{LLMWrapper,build_zero_caches}, convert_for_export, export_to_backend
- test_paged_matches_io_inplace_across_pages[remove_heads]; test_paged_rejects_rope_pruned_attention; test_paged_attention_requires_paged_cache; test_onnx_backend_rejects_paged_cache (requires_onnx); test_tensorrt_attention_plugin_matches_eager[remove_heads] (requires_edgellm_plugin).

**test_precision_passes.py** (needs `onnx` package, no gating decorator -> ImportError at collection w/o onnx) - export.tensorrt.precision_passes.fp32_accumulate_lrd
- test_only_lrd_gemms_accumulate_in_fp32.

**test_qnn_activation_only.py** [unit] - export_to_backend, QNNExportConfig (quant_scope, quantize_float_model), LLMWrapper
- test_requires_scope_all: importorskip(executorch) + skip if not is_qnn_available(); asserts ValueError "quantize_float_model" for quant_scope="linear". (runs only where QNN toolchain present, not marked qnn)
- test_float_model_lowered_fixed_point [requires_qnn, qnn]: float model lowered as fixed-point on HTP.

**test_qnn_bias_split.py** [unit] - QNNExportConfig.split_float_linear_bias; qnn_export builder
- test_split_only_float_biased_linears_and_keeps_outputs (pure CPU, no SDK).
- test_qnn_mlp_only_fake_quant_tracks_float[split F/T] [requires_qnn, qnn]: end-to-end export, fake-quant reference tracks float only with split.

**test_repeated_pattern.py** (none) - compression.structured_pruning_methods (reduce_pattern_scores, tile_pattern_mask, reduce_member_to_patterns, reduce_scores, build_structured_mask)
- 8 tests: repeat chunks grouping/divisibility, tile repeat, user example, max behaviour unchanged, duplicate smaller member, auto-GQA shared mask layout, auto-GQA rope ties real/imag.

**test_rope_variants.py** - blocks.rope (precompute_rope_inv_freqs, resolve rope type, ROPE_INV_FREQ), LLMWrapper, convert_for_export
- test_llama3_inv_freq_matches_hf; test_converted_llama_matches_hf[default|llama3]; test_resolve_rope_type.

**test_smoothquant.py** - compression.smoothquant via manager.smooth_activations
- test_smoothing_preserves_float_output_and_tames_outliers[qwen2|llama]; test_smoothing_criteria_selects_whole_groups_only; test_smoothing_refuses_quantized_weights; test_smoothing_requires_calibration_data.

### e2e/
**test_model_families.py** [e2e; all 8 families, tiny random, none] - T.*ForCausalLMCompress etc., *CompressionSchemesManager, LRD, quantization, unstructured + coupled structured pruning, convert_for_export. Each parametrized over ALL_FAMILY_NAMES:
- test_instantiate_and_forward; test_build_manager; test_lrd_soft_apply_and_restore; test_quantization_int8_apply; test_unstructured_pruning_apply_and_restore; test_structured_prune_hard_mlp (skip if not struct_prune_supported, i.e. modernbert); test_convert_for_export (causal families only).

**test_gqa_attention_pruning.py** [e2e; none] - Qwen2CompressionSchemesManager, structured pruning auto q/k masks, convert_for_export, LLMWrapper, build_zero_caches
- test_auto_qk_soft_mask_layout; test_auto_qk_hard_prune_convert_decode; test_vproj_prune_cascades_to_oproj_via_coupled_repeated_pattern; test_auto_qk_hard_prune_io_cache_sizes; test_auto_qk_hard_prune_export_keeps_kv_cache[mutable|io_scatter]; test_unfinalized_pruned_cache_raises.
- Has `sys.path` hack + `# noqa: E402` imports (redundant with conftest).

**test_export_pipelines.py** [pytestmark e2e, download, slow, requires_hub] - downloads `Qwen/Qwen2-0.5B` (fp16, CPU; skips if load fails). Fixture `qwen_classes` (module scope) returns (Qwen2ForCausalLMCompress, Qwen2CompressionSchemesManager, convert_for_export). Helpers `_load_float_model`, `_quantize_two_mlp_layers`, `_components`.
- test_hf_export_roundtrip: LRD rank 64 on layer0 down_proj, `hf.export_to_hf`, reload, forward.
- test_convert_for_export_graph: "text" in converted.
- test_export_xnnpack [requires_executorch]: export_to_backend with XNNPACKExportConfig -> .pte.
- test_export_xnnpack_custom_sdpa [executorch + custom_sdpa].
- test_export_tensorrt[None|dequantize|edgellm_plugin] [requires_tensorrt, GPU; edgellm param runtime-skips w/o plugin].
- test_export_qnn [requires_qnn].
Not marked `gpu`/`executorch`/`qnn` markers, so `-m "not slow"` is the only filter; default testpaths DO include this file (collects, skips only on caps/offline).

## 3. Reverse index: transformersurgeon module -> test files
- blocks/mha.py: unit/test_add_batch_dim, test_gqa_rope_pruning, test_mha_causal_custom_sdpa; e2e/test_gqa_attention_pruning (indirect)
- blocks/rope.py: test_add_batch_dim, test_gqa_rope_pruning, test_mha_causal_custom_sdpa, test_rope_variants, test_export_fixes
- blocks/decoder.py, blocks/config.py: test_add_batch_dim, test_io_inplace_cache (direct); all converted-model tests indirect
- blocks/kv_cache_ops.py: test_io_inplace_cache. blocks/paged_attention_ops.py: test_paged_attention (KV_PAGE_SIZE + indirect)
- blocks/mlp, norm, linear_compressed, embedding_compressed, encoder: only indirect via e2e/test_model_families
- compression/structured_pruning(+_methods), coupled_pruning: test_repeated_pattern (methods direct), test_known_bugs, e2e/test_model_families, e2e/test_gqa_attention_pruning
- compression/block_pruning: test_block_pruning
- compression/quantization(+vanilla): test_lpbq_quantization, test_known_bugs, e2e/test_model_families, export tests (via manager)
- compression/smoothquant: test_smoothquant, test_onnx_tensorrt_export (smooth=True)
- compression/lrd: e2e/test_model_families, e2e/test_export_pipelines (indirect only)
- compression/unstructured_pruning: e2e/test_model_families (indirect only)
- calibration/*: test_extract_summaries, test_cascade_calibration, test_smoothquant (indirect, via manager)
- export/common.py: test_export_fixes (calibrate_pt, manifest), test_attention_semantics, test_block_pruning, test_rope_variants, test_paged_attention, test_onnx_tensorrt_export (extract_layer_quant_info), e2e/test_gqa
- export/export.py (export_to_backend): test_onnx_tensorrt_export, test_paged_attention, test_export_fixes, test_qnn_*, e2e/test_export_pipelines
- export/onnx/*: test_onnx_tensorrt_export, test_paged_attention
- export/tensorrt/*: test_onnx_tensorrt_export (config, session, engine), test_precision_passes, test_paged_attention (plugin; skipped w/o plugin)
- export/executorch_exporters/qnn: test_known_bugs, test_qnn_activation_only, test_qnn_bias_split, e2e/test_export_pipelines (last 3 mostly skipped w/o SDK)
- export/executorch_exporters/xnnpack: e2e/test_export_pipelines ONLY (needs executorch + downloaded model)
- hf/hf_export.py: e2e/test_export_pipelines::test_hf_export_roundtrip ONLY (needs Hub)
- models/*_c (qwen2, llama, bert, modernbert, distilbert, vit, qwen2_vl, qwen2_5_vl): e2e/test_model_families (all via model_factory); test_cascade_calibration, test_known_bugs
- utils/manager.py: nearly every test; utils/convert.py (convert_for_export): ~10 files; utils/cascade.py: test_cascade_calibration; utils/scheme.py, grouping.py: indirect, test_known_bugs for group guards
- utils/modeling.py, configuration.py, utils.py, interface.py: indirect only

## 4. Stale / dead / duplicated
- Empty-of-source folders (only `__pycache__/*.pyc`, cpython-310 pytest-8.4.2): 
  - test/compression_tests: cache_impl_parity_test, cascade_test, compression_test, test_messages
  - test/executorch_tests/simple: io_cache_export_test; test/executorch_tests/exporter_function/{qnn,xnnpack}: test_pretrained_quant_export
  - test/hf_export_tests: quantize_test, test_hf_roundtrip
  No .py sources exist anywhere (find confirmed); nothing outside test/ references these dirs. They are not in pytest testpaths. Safe to delete (6 dirs + their pycache); `test_hf_roundtrip` is superseded by e2e test_hf_export_roundtrip, cascade/ io-cache/ compression by unit/e2e tests.
- test/__pycache__, unit/e2e/_helpers pycache contain both py310 (pytest 8.4.2) and py312 (pytest 9.1.1) pycs -> suite was run under both interpreters. No .pyc without a source in unit/ e2e/ _helpers/.
- Stale docs/comments: test_known_bugs.py docstring claims tests are `xfail`-marked with strict=False; none are (all bugs fixed -> plain assertions). model_factory `known_broken` field and `xfail` branches in e2e/test_model_families (lines ~30, 114) are dead (no family sets it). capabilities.py docstring shows `from test._helpers.capabilities` import (wrong). Markers gpu/executorch/tensorrt/xfail_known_bug declared but unused; `torch_device` fixture unused.
- Environment-fragile: test_known_bugs::test_qnn_unavailable... asserts QNN absent (breaks on QNN boxes); test_precision_passes imports onnx at module top without gating (collection error w/o onnx); test_export_fixes::test_manifest_describes_per_layer_caches has no requires_onnx decorator though it appears to use export_to_backend (verify); test_qnn_activation_only::test_requires_scope_all is not marked qnn/gated by decorator.
- Duplication: `_components(model, ...)` helper (embedding/decoder/final_layer/config dict) copy-pasted in test_qnn_activation_only, test_qnn_bias_split, test_export_pipelines, test_onnx_tensorrt_export (likely), test_paged_attention; token-stepping `run(m)` loop duplicated in the two QNN tests. Candidates for _helpers. e2e/test_gqa_attention_pruning duplicates unit coverage partially (test_gqa_rope_pruning/test_repeated_pattern) at higher level - intentional.
- No removed-API usage detected: all imports resolve to existing modules (checked names exist: export_to_backend, QNNExportConfig, is_qnn_available, export_with_qnn, TensorRTLLMSession, apply_*_quantization, fp32_accumulate_lrd, calibrate_pt, extract_layer_quant_info, kv_cache_update, KV_PAGE_SIZE, lpbq_group_scales). Static grep only, not executed.

## 5. Gaps (public classes/functions never named anywhere under test/)
Name-based grep; many are exercised indirectly through the manager/export (marked *).
- blocks: AtomicSum, CustomEncoderConfigCompress, Conv2dCompressed/Conv3dCompressed, TransformerDecoderBlock, EmbeddingCompressed*, TransformerEncoder(+Block), LinearCompressed*, MHABase, MHAEncoderFusedProj, MLPGated*, VCONBlock, pruning_dims (effective_num_pruned, effective_out_features, pruning_config_dims), paged_attention_ops (num_kv_pages, identity_page_table, paged_attention - indirect via decoder), rope (precompute_mrope_inv_freqs, precompute_mrope_cos_sin_half, apply_rope_multihead - Qwen-VL mRoPE has NO numeric test).
- calibration: backbone.run_compression_calibration*, all raw_data collectors (Activation/ShiftedActivation/OutputActivation/Weight/Bias/WeightGrad, RawDataCollector, normalize_calibration_batch, instantiate_raw_collector), summaries classes (ActivationRange, Covariance/Shifted/Cross, Weight/Bias, Input/OutputActivation, WeightGrad), CalibrationSummary, SummaryRuntime, get_summary, unique_summaries. Only reachable via extract_summaries/smooth tests.
- compression: LRDer + validate_lrd_*, low_rank_svd / aa_svd / svd_llm_v2 (NO test of any LRD algorithm numerics, only soft apply/restore in e2e), Quantizer + all validate_quantization_*/activation/precision validators (except via test_known_bugs error message), quantize_gptq, quantize_hqq, compute_activation_scale_zp_maxmin, StructuredPruner/CoupledPruner/BlockPruner classes + validators, score_magnitude/gradient/random, UnstructuredPruner + build_unstructured_mask/mask_magnitude/gradient/random, smoothquant internals (resolve_smoothing_groups, collect_input_absmax, apply_smoothquant), compression/registry.py, abstract.py.
- export: BackendExportResult, ExporterConfig, BackendExportConfig, build_example_inputs, resolve_components_and_wrapper, inject_scales_into_pt2e_observers, dequant_from_exported_state_dict, find_weight_mismatches, to_device, reference_logits, logit_error_stats, manifest_path_for, load_llm_manifest, finalize_export_result; executorch_exporters.common (ExecuTorchExportResult, build_quantizer_from_layer_info, run_simple_inference_stats, finalize_executorch_result); xnnpack export_with_xnnpack (only e2e w/ download); export.linear_quantizer (annotate_linear, LinearOnlyQuantizer); onnx default_translations, ONNXExportResult, io_names, export_with_onnx (indirect via export_to_backend); tensorrt attention_passes.fp32_attention_scores, edgellm_attention (attention_translations, edgellm_attention_pass), edgellm_int4 (edgellm_int4_pass, resolve_plugin_libraries), engine (load_plugin_libraries, OptimizationProfile, EngineBuildConfig, build_engine, time_cuda), tensorrt_export (engine_profiles, build_engine_from_manifest, export_with_tensorrt, main). export/registry.py untested by name. Real QNN lowering (qnn_export.py, 822 lines) is covered only by SDK-gated tests; its pure-logic parts (apart from split_float_linear_bias) have no CPU test.
- hf: hf_export.export_to_hf only in the Hub-gated e2e test (no offline roundtrip, no quantized-model roundtrip).
- models: vit_c/legacy_conversion (build_legacy_vit_key_mapping, peek_checkpoint_keys); no per-family indexing file test; no test for encoder/ViT/VL convert_for_export (only causal qwen2/llama).
- utils: configuration.init_compressed_config*, grouping.SchemeGroup*, interface (FallbackProgressBar, build_progress_bar), modeling.replace_layers_upon_init*, scheme.CompressionScheme*, utils.py helpers (flatten_index_paths, get_submodule, infer_model_device, move_to_device); manager.py (1280 lines) has no dedicated test file (covered by feature tests); utils/cascade.py (994 lines) only has 3 cascade tests (error-guard + one happy path).
- Structural gaps: no tests for save/load of compressed models via Hub (except download e2e), no GPU-only non-export tests, no test for examples/ or scripts/, no coverage config.
