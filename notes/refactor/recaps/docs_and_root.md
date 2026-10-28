# Recap: docs/, mkdocs.yml and repo-root files

Repo: transformer-surgeon (package `transformersurgeon`). Generated 2026-10-28. Read-only survey.

## 1. File tree with line counts
```
mkdocs.yml 73 | README.md 260 | AGENTS.md 617 | CLAUDE.md 66 | FRAMEWORK_STRUCTURE.md 194
llms.txt 25 | setup.py 25 | pytest.ini 17 | __init__.py 1 | .gitignore 255
docs/ (3200 lines total)
  index.md 92 | concepts.md 406 | compression_methods.md 241
  examples/ 01_basic_lrd.md 72 | 02_calibrated_lrd_svd_llm_v2.md 91 | 03_hf_export_roundtrip.md 97
  api/ (each 6-20 lines, a title + short prose + `::: module.Class` directives)
    blocks/   linear_compressed 9, vcon_block 8
    calibration/ activation 11, covariance 16, raw_data_base 6, summaries_base 11
    compression/ abstract 7, registry 9, lrd 14, quantization 19, structured_pruning 16, unstructured_pruning 14
    utils/    manager 7, scheme 7
    export/   common 9, qnn 6, registry 9, xnnpack 7, config 8, export 7, onnx 10, tensorrt 20
  investigations/ FRAMEWORK_PROBLEMS 313 | QNN_DECODE_SPEED_FIX 300 | QNN_SPEED_SUMMARY 160
     XNNPACK_DECODE_SPEED_FIX 582 | XNNPACK_QNN_BRANCH_REVALIDATION 80 | XNNPACK_SPEED_SUMMARY 238
     plan-custom-sdpa-xnnpack 86 | TENSORRT_ONNX_EXPORT 214
notes/   -> refactor/ (recaps/ - this file)
artifacts/ -> aten_op_problems_qwen2_qnn_sm8650_full.log   (only file; .gitignore has `*artifacts/` and `*.log`)
```
Also at repo root (not in scope but relevant): examples/ has 4 scripts (01..04; docs cover only 01-03; `04_cache_impl_export_interface.py` undocumented), scripts/{compression,executorch,summaries,tensorrt}, test/{unit,e2e,_helpers, plus empty-ish legacy compression_tests, hf_export_tests, executorch_tests}, `transformer_surgeon.egg-info`, `qwen-vl-finetune`, `vlm-modules-importance`, `benchmarks`.

## 2. mkdocs.yml
- site_name TransformerSurgeon; repo_url github.com/CESOIA/transformer-surgeon.
- theme: material (indigo, features navigation.tabs, navigation.sections, toc.integrate, search.suggest).
- plugins: search, mkdocstrings (python handler; docstring_style google, show_source, show_root_heading, show_if_no_docstring false, members_order source).
- markdown_extensions: admonition, pymdownx.highlight (anchor_linenums), pymdownx.superfences, toc(permalink).
- nav: Home, Concepts, Compression Methods, Examples (3), API Reference with groups Compression (6 pages), Calibration (4), Manager (2), Blocks (2), Export (8: export, config, registry, common, xnnpack, qnn, onnx, tensorrt). All 38 files referenced exist (nav is complete vs docs/ tree).
- No `docs_dir` set -> default `docs/`, so `docs/investigations/*.md` are NOT in nav (mkdocs will warn "not included in nav" but still build them). `/site` is gitignored.
- **Installed?** NO. `import mkdocs` fails in base python and in all conda envs checked (base, py312_torch, py310-executorch-qnn, py312_executorch_xnnpack, py312_tensorrt). `material` and `mkdocstrings` also missing in py312_torch (others were not individually checked beyond mkdocs). Docs have never been built here; needs `pip install mkdocs mkdocs-material "mkdocstrings[python]"`. setup.py has no `docs` extra.
- mkdocstrings requires the package to be importable (torch/transformers/executorch imports at module top could fail for `export.*` pages: tensorrt/onnx/qnn modules import optional deps; the docs build env must tolerate or mock them).

## 3. Per-doc-page summary and staleness
- **docs/index.md** - Landing page: install, feature list (4 compression methods, manager/scheme, LinearCompressed, VCON, export fns), table of 8 supported model families. Mostly duplicate of README "What's Included" (near-identical text). Slightly stale: export bullet says `xnnpack`, `qnn` or `tensorrt` but omits `onnx` backend; no mention of block_pruning, SmoothQuant, paged attention.
- **docs/concepts.md** - Mental model: three-layer (Compressor/Scheme/Manager), LinearCompressed internals, Compressor lifecycle, structured pruning coupling and grouping, calibration pipeline (RawDataCollector -> CalibrationSummary), standard vs cascade (and `no_cascade_calibration`), VCON, Export Backends (4 backends, shared export/common.py steps, TensorRT example), extension guide (method, collector, summary, model family, export backend). Export section is CURRENT (onnx, linear_quantizer, io_inplace), BUT the LinearCompressed and Compressor-lifecycle sections are STALE: they describe `weight_2` (concepts.md lines 42-104) while the code uses `linear_V` (blocks/linear_compressed.py:39). Gaps: no block_pruning, SmoothQuant (`manager.smooth_activations` exists, manager.py:302), QNN quant_scope/LPBQ, `utils/cascade.py`.
- **docs/compression_methods.md** - Plain-language reference for lrd, structured_pruning, unstructured_pruning, quantization (parameters, methods, soft/hard, grouped pruning, LRD interaction), criteria language, see-also. Mostly current; missing `block_pruning` section, `group_scale_bits`/LPBQ, `dynamic_per_token` details (these are in AGENTS.md only); one mention of smooth* only.
- **docs/examples/01..03** - Prose walkthroughs of examples/01_basic_lrd.py, 02_calibrated_lrd_svd_llm_v2.py, 03_hf_export_roundtrip.py (run commands + step explanation). Script files exist. 04_cache_impl_export_interface.py has no page.
- **docs/api/** - Thin stubs (heading + 1 table/paragraph + mkdocstrings `:::` directives). All referenced symbols spot-checked present in code: LRDer, Quantizer, StructuredPruner, UnstructuredPruner, Compressor, CompressionSchemesManager, CompressionScheme, LinearCompressed, VCONBlock, RawDataCollector/ActivationCollector/OutputActivationCollector/ShiftedActivationCollector, CalibrationSummary/SummaryRuntime, covariance summaries, export.* modules (module paths all exist: export/{common,config,export,registry}.py, executorch_exporters/{qnn/qnn_export,xnnpack/xnnpack_export}.py, onnx/{onnx_export,quantization}.py, tensorrt/{tensorrt_export,engine,session,edgellm_int4}.py). Missing API pages for: `compression/block_pruning.py`, `coupled_pruning.py`, `smoothquant.py`, `utils/grouping.py` (SchemeGroup), `utils/convert.py` (convert_for_export), `hf/hf_export.py` (export_to_hf), `blocks/{mha,decoder,mlp,norm,rope,config,pruning_dims,kv_cache_ops,paged_attention_ops,embedding_compressed,conv_compressed,atomic}`, `export/linear_quantizer.py`, `export/tensorrt/{precision_passes,attention_passes,edgellm_attention}`, `calibration/backbone.py`, activation_range/raw_activation/weight_grad summaries, weight_grad/params raw_data. Not verified: whether raw_data/summaries `params.py` are public.
- **docs/investigations/** - see section 6 (not in nav).

## 4. setup.py
- name `transformer-surgeon`, version `0.10.0`, description "Transformer model compression and optimization tools", `find_packages()`, python_requires >=3.10.
- install_requires: torch>=2.9.0, transformers>=5.9.0, torchao (unpinned).
- extras: `executorch` (executorch>=1.0.0); `onnx` (onnx>=1.19, onnxscript>=0.7, ml_dtypes; tensorrt installed separately); `test` and `dev` both = [pytest] (deliberately identical). No `docs`, no `tensorrt`, no `qnn` extra.
- No pyproject.toml; `find_packages()` also picks up any dir with __init__.py (root `__init__.py` makes `.` a package but find_packages only scans subdirs; `test/_helpers`, `scripts` may be included if they have __init__.py - unverified).
- pytest.ini: testpaths test/unit test/e2e; `--import-mode=importlib`, `--strict-markers`; markers unit, e2e, slow, download, gpu, executorch, tensorrt, qnn, xfail_known_bug.
- .gitignore gotchas: ignores `*.txt` (re-includes only `!llms.txt` - any new requirements*.txt would be ignored), `*.json`, `*.csv`, `*.log`, `*.ipynb`, `models/` (re-includes `!transformersurgeon/models/` only), `data/`, `results*/`, `*artifacts/`, `experiments/`, `transformers/`, `/site`. A new docs/ asset folder named `models/`, `data/`, or `*.json` assets would be silently ignored.

## 5. Design constraints (AGENTS.md / CLAUDE.md / FRAMEWORK_STRUCTURE.md)
CLAUDE.md says "read AGENTS.md before code changes"; llms.txt is a pointer index to README, docs/concepts.md, AGENTS.md, FRAMEWORK_STRUCTURE.md, examples/*.py, test/.

### Core loop / semantics
- Core loop: `manager.set(...)` -> `manager.apply(hard=False)` -> `model.generate()` -> `manager.restore()`. `hard=True` irreversible, final export only.
- Three layers: Compressor (what), CompressionScheme (where), CompressionSchemesManager (how many); one direction.
- Criteria language: None/"all"=all; int=block index; str=substring; `[[a,b]]`=AND; `[a,b]`=OR.

### Invariants (AGENTS.md "Coding Invariants", CLAUDE.md "Critical invariants")
1. `Compressor.apply()` always gets a `LinearCompressed`, never nn.Linear (replaced at init by `replace_layers_upon_init`).
2. `manager.set()` is purely declarative; model changes only in `apply()`.
3. Soft apply reversible; hard permanent; `restore()` after hard silently wrong.
4. `init_vcon()` must precede `apply()`.
5. `prepare_for_save()` before `export_to_hf()` when applying manually (or pass `manager=` to export_to_hf); it upcasts fp16 to fp32 and dequantizes: only before `save_pretrained`, NEVER before export.
6. `calibration_store` is a plain dict per scheme; only CalibrationSummary writes, Compressor reads.
7. `INDEXING["path_template"]` is a format string with `{block_index}` and `{path}`.
8. `attn_impl="custom_sdpa"` requires `cache_impl="mutable"` and q_head_dim == value_head_dim (ValueError/RuntimeError); XNNPACK only; needs executorch LLM custom-ops extension importable at export AND at inference.
9. After LRD, `weight` is `[out, rank]` and `linear_V.weight` `[rank, in]`.
10. Hard-prune: output rows removed, cascades to next layer inputs via coupled_pruning driven by indexing `pruning.output_dependence`; effective kept dim single-sourced in `blocks/pruning_dims.py::effective_out_features`.
11. Never resize module buffers inside `forward()` (torch.export traces it as fresh tensor, KV cache silently lost; fixed 2026-10-13). Geometry resolved once at conversion (`finalize_rope_pruning`, `finalize_cache_geometry`, `finalize_head_pruning`).
12. Attention scores always scale by original `1/sqrt(head_dim)` (`MHABase.score_scale`) even after head_dim pruning.
13. Causal-mask constant must exceed model's logit range (now -65504, was -10000 -> bug).
14. Hard quantization is computed by tsurgeon (`_apply_hard_quantization`, METHOD_FUNCTIONS), torchao `IntxUnpackedToInt8Tensor` only a container; do NOT reintroduce `quantize_()` in hard path.
15. All exporters read the exact numbers via `export/common.py::extract_layer_quant_info`; LRD+quant stale pre-LRD scale raises.
16. `rope_type` in indexing: fixed name or "auto"; mismatch or unimplemented type raises at conversion (no silent fallback); new RoPE variant = function in `blocks/rope.py::ROPE_INV_FREQ`, attention scaling must stay 1.
17. `share_mask`/GROUP_OPTIONS settable only via `group=`, not criteria.
18. Manager `set_calibration_mode("cascade")` required for `aa-svd`; families with `'no_cascade_calibration': True` (bert_c, modernbert_c) raise on cascade.
19. Block pruning (whole heads): uniform per KV group, cannot remove whole group, not combinable with structured pruning of same q_proj; wired for q_proj in qwen2_c and llama_c only.
20. SmoothQuant (`manager.smooth_activations`) must run BEFORE quantization apply; refuses already-quantized weights; groups from `indexing['smoothing']`.

### `blocks/` frozen-model rule (CLAUDE.md)
- `blocks/` (mha, mlp, decoder, encoder, linear_compressed, pruning_dims, config, rope) = converted, already-compressed graph; must NEVER decide what to prune/compress. Decisions made earlier on HF `*_c` model by manager; `utils/convert.py::convert_for_export()` copies decided weights/shapes/masks only.
- `pruning_dims.py`, `linear_compressed.py`, `config.py` are passive (turn decided ratio/rank/mask into static geometry).
- No path may hand a `blocks/` model to CompressionSchemesManager or mutate it post-construction (except one-time `finalize_*` hooks from `convert_for_export`). `blocks/indexing.py` (CUSTOM_DECODER/ENCODER_INDEXING) was removed for violating this. New pruning-aware block = pruning_dims pattern or `finalize_*`, not a check in `forward()`.

### Anti-patterns (AGENTS.md)
- Don't instantiate CompressionScheme directly. Don't assume weight shape after LRD. Don't `hard=True` then `restore()`. No model-specific logic in manager (belongs in `indexing_*.py`). Don't use standard mode with aa-svd. Don't use cascade with `no_cascade_calibration` families. Don't call `reapply_masks()` before `apply()` (masks created in apply; reapply is for STE fine-tuning; `remove_masks()` drops them). Don't reorder GQA broadcast/permute in `attention()` (VTCM spill, 537 MB/token on Qwen2-0.5B). Don't use "int8" strings for precision (must be int 2-16 / "full"/"binary").

### Backend-specific rules (AGENTS.md Export Backends - large section, much of it belongs in design docs)
- All backends via `export_to_backend(model, config)` -> `EXPORT_ROUTINES` registry: xnnpack, qnn (ExecuTorch .pte), onnx, tensorrt. Each writes `<stem>.manifest.json` (cache impl/layout/dtype, per-layer KV shapes); runners use `common.load_llm_manifest`/`zero_caches_from_manifest`.
- cache_impl: `mutable` (ExecuTorch), `io_inplace` (ONNX/TRT; BHSD caches as graph I/O via `tsurgeon::kv_cache_update` -> TensorScatter opset 24; `mutable` rejected for ONNX), `io_paged` (TRT Edge-LLM AttentionPlugin; decode only, needs EDGELLM_PLUGIN_PATH, equal q/k/v head dims, unpruned RoPE).
- attn_impl: manual (default) / sdpa / custom_sdpa. `add_batch_dim` experimental, XNNPACK-only until validated.
- XNNPACK `quant_path`: "pt2e" legacy vs "torchao" (dynamic int8 activations qd8 kernels); keep XNNPACK exports fp32.
- QNN: `quant_scope="all"` is vendor recipe (one HTP partition) vs legacy "linear"; `keep_reference_module`; `split_float_linear_bias` default True (bias overflow in 16a16w); `quantize_float_model` activation-only reference; PT2E calibration decodes from pos 0 with filling cache, pass `calibration_token_ids`; `masked_softmax`; RMSNorm must fuse (norm.py computes `x*x` not pow); LPBQ is the only group-quant format QNN supports (precision=4, per_group, group_scale_bits=4), forces ConvertLinearToConv2d and unties embeddings; per-group non-LPBQ raises; `rmsnorm_prescale` default True; `convert_linear_to_conv2d` default False.
- TensorRT: weight-only INT4 blocked DQ; INT8 per-group raises; W8A8 recipe; `fp32_accumulate_lrd` default True; `rmsnorm_upcast=True` for fp16; engine built on target device; INT4 plugin opt-in (`int4_backend="edgellm_plugin"`).
- `config.max_seq_len` seeds `convert_options["max_cache_len"]` when unset.
- Open TODOs: `TODO(P3a)` EmbeddingCompressed has no weight-quant path (embeddings/tied lm_head fp32); `TODO(custom_sdpa-encoder)` MHAEncoder lacks custom_sdpa.

### AGENTS.md file map / extension recipes
- File map table (28 rows) maps tasks -> files (compression/*.py, *_methods/ dirs, registry, indexing/define, manager.iter_filtered, calibration collectors/summaries, blocks, hf_export, convert, export/*). 
- Recipes: new Compressor (subclass `Compressor`, implement 6 abstract methods `set_calibration_store, needs_calibration, apply, restore, _to_compress, __repr__`, register in COMPRESSOR_DICT and COMPRESSION_REGISTRY); new model family (indexing with required `num_blocks_attr, path_list, path_template`; define_*: Config/Model/Manager classes with exact pattern; export in models/__init__); new CalibrationSummary (SUMMARY_REGISTRY singleton); new RawDataCollector (RAW_DATA_REGISTRY).
- FRAMEWORK_STRUCTURE.md adds: file tree, 6 core concepts, cascade calibration_groups (list or dict form), end-to-end data flow (11 steps), new-architecture steps.

### Contradictions / stale statements
1. FRAMEWORK_STRUCTURE.md says LinearCompressed low-rank uses `weight` + `weight_2`; code uses `linear_V` (and AGENTS.md/CLAUDE.md say `linear_V`). `weight_2` also appears in docs/concepts.md (lines 42-104; README.md does not).
2. FRAMEWORK_STRUCTURE.md tree is incomplete vs code: missing `blocks/{atomic,conv_compressed,embedding_compressed,kv_cache_ops,paged_attention_ops}.py`, `compression/{block_pruning,smoothquant}.py`, `utils/{cascade,interface}.py`, `calibration/summaries/{activation_range,raw_activation,weight_grad,params}.py`, `export/linear_quantizer.py` (mentioned in prose only), `export/onnx/`, `export/tensorrt/{attention_passes,edgellm_attention,precision_passes}.py`. Its tree labels `export/` with tab indentation and omits onnx. It lists `models/qwen2_c ... vit_c` correctly (8 families incl. distilbert).
3. AGENTS.md test section says "All 7 families (qwen2, llama, bert, modernbert, distilbert, vit, qwen2_vl, qwen2_5_vl)": that lists 8; CLAUDE.md also says "all 7 families". Code has 8 model dirs. Also says default pytest includes only `test/e2e/test_model_families.py` while pytest.ini collects all of test/e2e (includes test_export_pipelines, test_gqa_attention_pruning; capability-gated).
4. AGENTS.md "Compression Parameter Reference" lists structured_pruning, unstructured, quantization, lrd, block_pruning; README/index/concepts "four compression methods" omit block_pruning (a 5th registry entry, unverified in registry.py but AGENTS claims it).
5. AGENTS.md says ONNX export "docs/investigations/TENSORRT_ONNX_EXPORT.md" and refers to `../trt-models/` benchmark workspace (outside repo; not present in repo).
6. AGENTS.md Dev Setup says `pip install -e ".[dev]"`; README/index say `[test]`; equivalent (setup.py comment) but inconsistent wording.
7. AGENTS.md `Open TODOs` says `git grep` but repo is not a git repo in this working copy (no .git here); investigations reference branches (`debug-qnnspeed`, etc.).
8. CLAUDE.md says MHA "cascades v->o via coupled_repeated_pattern" and block_pruning not mentioned; AGENTS.md covers both: consistent but CLAUDE.md incomplete.
9. AGENTS.md references `scripts/executorch/xnnpack/inference_exported_test.py` and `exporter_function_test.py`; scripts/executorch/{qnn,xnnpack} exist (files not individually verified).
10. AGENTS.md mentions `Quantizer` hard path precision "int in [2,8]" vs soft [2,16]: internal, consistent.
11. Memory notes (user) say "QNN: no SmoothQuant" and "same config on all backends unless justified in UNMATCHING_CONFIGURATIONS.md": AGENTS.md has no mention of that backend-unification rule nor the QNN pos-1 calibration conclusions; likely belongs in the new docs.
12. AGENTS.md is a 617-line mixture of agent rules, API reference, backend measurements and investigation findings: the refactor should split (rules vs per-backend design docs vs parameter reference).

## 6. docs/investigations/ (historic; not in nav)
- FRAMEWORK_PROBLEMS.md (2026-07-28) - Problems found comparing tsurgeon XNNPACK export vs ExecuTorch export_llama on TinyLlama (P1.. P3a/P3b embeddings, etc.).
- XNNPACK_DECODE_SPEED_FIX.md - Detailed log (branch debug-xnnpackspeed, 2026-07-29/09-10) closing XNNPACK decode-speed gap vs Meta's export; origin of `add_batch_dim`, `quant_path`.
- XNNPACK_SPEED_SUMMARY.md - Plain-language companion to the above.
- plan-custom-sdpa-xnnpack.md - Plan for opt-in `attn_impl="custom_sdpa"` (ExecuTorch llama custom ops).
- QNN_DECODE_SPEED_FIX.md - Engineering log (branch debug-qnnspeed) bringing QNN export close to Qualcomm's reference; GQA expand-before-transpose fix, rmsnorm_prescale, linear-to-conv2d (AOT-only, x86).
- QNN_SPEED_SUMMARY.md - Plain-language companion (537 MB/token spill finding).
- XNNPACK_QNN_BRANCH_REVALIDATION.md - Confirms QNN fix doesn't regress XNNPACK.
- TENSORRT_ONNX_EXPORT.md - ONNX -> plain TensorRT path for Jetson Orin, benchmarked vs TensorRT Edge-LLM (branch debug-tensorrtspeed, v0.10.0).
Note: none cover the Oct 2026 QNN work (bias overflow, KV-cache pruneNhd, LPBQ) - that lives only in AGENTS.md prose and in `artifacts/aten_op_problems_qwen2_qnn_sm8650_full.log`.

## 7. Root __init__.py
- 1 line: `# Transformer Surgeon package`. No code. It turns the repo root into a package directory, but the real package is `transformersurgeon/` (its `__init__.py` does `from .models import *` plus guarded imports of `precompute_rope_inv_freqs` and `convert_for_export`). Not needed for install (setup.py uses find_packages from root, root __init__ is not a package member) nor imports; harmless, could be deleted (risk: pytest rootdir/rootpath with importlib mode is fine without it). Recommend removing.
- `transformersurgeon/__init__.py` has no `__version__`; the only version is setup.py 0.10.0.
