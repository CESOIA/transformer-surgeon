# Recap: examples/

Four standalone, user-facing demo scripts (no package, no tests, no `__init__`). All import only
from the public top-level `transformersurgeon` namespace plus three deeper modules (marked DEEP).
Only 01-03 are referenced by docs; 04 is undocumented.

## File tree (lines)
```
examples/01_basic_lrd.py                      84
examples/02_calibrated_lrd_svd_llm_v2.py     142
examples/03_hf_export_roundtrip.py           165
examples/04_cache_impl_export_interface.py   184
```
Doc references: `docs/examples/01..03_*.md` (mirrors, 04 has none), `mkdocs.yml:35-37` (01-03),
`docs/compression_methods.md:239-240` (01, 02), `llms.txt:19-21` (01-03). 04 is in none.

## Per file

### 01_basic_lrd.py - reversible LRD on 2 layers, generate, restore
CLI: `--model-name` (Qwen/Qwen2-0.5B-Instruct), `--prompt`, `--lrd-rank` (640), `--max-new-tokens` (32).
transformersurgeon API:
- `from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress` (:16)
- `Qwen2ForCausalLMCompress.from_pretrained(name, torch_dtype="auto")` (:42)
- `Qwen2CompressionSchemesManager(model)` (:53)
- `manager.set("lrd","rank",R,criteria=[[0,"self_attn.q_proj"]])` and `[[1,"mlp.up_proj"]]` (:54-55)
  -> criteria mini-language: list of AND-lists, int = block index, str = name substring.
- `manager.apply(hard=False)` (:56), `manager.restore()` (:77)

### 02_calibrated_lrd_svd_llm_v2.py - SVD-LLM-v2 with WikiText-2 calibration
CLI: `--model-name` (Qwen/Qwen2-0.5B), `--lrd-rank` (64), `--num-calibration-samples` (8), `--seq-len` (2048).
API:
- same two imports as 01 (:23)
- `manager.set("lrd","method","svd-llm-v2",criteria="all")` (:94)
- `manager.iter_filtered(criteria="all")` yielding scheme objects; uses `scheme.get_module()` and
  `scheme.path` (:97-99)
- `manager.set("lrd","rank",R,criteria=scheme.path)` (:99) and `criteria=<str suffix>` (:103)
- `manager.set_calibration_data(DataLoader)` (:108)
- `manager.apply(hard=False, criteria="all", device=device, offload_to_cpu=True)` (:110)
Does NOT call `run_calibration()` explicitly: relies on `apply()` triggering calibration internally.

### 03_hf_export_roundtrip.py - VCON + hard LRD -> export_to_hf -> reload
CLI: `--model-type {qwen2_5_vl_c,qwen2_vl_c}` (qwen2_5_vl_c), `--rank` (128), `--out-dir` (temp dir).
API:
- DEEP: `from transformersurgeon.hf import export_to_hf` (:25)
- lazy imports in `_get_classes` (:47-69): `Qwen2VLForConditionalGenerationCompress`, `Qwen2VLConfigCompress`,
  `Qwen2VLCompressionSchemesManager`, `Qwen2_5_VLForConditionalGenerationCompress`,
  `Qwen2_5_VLConfigCompress`, `Qwen2_5_VLCompressionSchemesManager` (all from top-level; ConfigCompress is
  returned but never used afterwards - dead value)
- `manager.set("lrd","rank",R,criteria=[["language_model","mlp.down_proj",26],[...,27]])` (:99-104)
- `manager.init_vcon()` (:106), `manager.set_vcon_beta(beta=1.0)` (:107), `manager.apply(hard=True)` (:108),
  `manager.update_config()` (:109)
- `export_to_hf(model, repo_id=, base_model=, out_dir=, readme=, embed_code=True, token=None, private=True, exist_ok=True)` (:121-130)
  (does not pass `manager=`; docstring of export_to_hf says either pass manager or call
  `apply(hard=True)` + `prepare_for_save()` yourself - example skips `prepare_for_save()`; harmless for LRD only)
- `model_class.from_pretrained(export_path)` reload (:141) - relies on `embed_code`/autoload.py to resolve custom classes.
Fragile: export_to_hf appends `repo_id.split('/')[-1]` to out_dir (hf_export.py:47) and also returns it (:109);
the example ignores the return value and re-discovers the subdir by `os.listdir` (:134-136).
(Note: llms.txt describes it as "push to HuggingFace Hub"; it is local only, `token=None`.)

### 04_cache_impl_export_interface.py - shows exported-model I/O contract per `cache_impl`
CLI: `--model-name` (Qwen/Qwen2-0.5B-Instruct), `--prompt`, `--cache-impl {all,mutable,io_scatter,io_concat}` (all),
`--max-new-tokens` (24), `--max-cache-len` (256).
API:
- `from transformersurgeon import Qwen2ForCausalLMCompress` (:57)
- DEEP: `from transformersurgeon.utils import convert_for_export` (:58)
  (top-level `__init__` wraps this import in try/except -> None; here it is imported from `.utils` directly)
- DEEP: `from transformersurgeon.export.common import LLMWrapper, build_zero_caches` (:59)
- `convert_for_export(model, options={"use_sdpa":False,"cache_impl":..,"max_cache_len":..})` -> dict, uses key `["text"]` (:84-93)
- `LLMWrapper(model.get_input_embeddings(), decoder, model.lm_head)` (:94) - this is the same 3-component
  assembly the exporters do.
- Introspected internals (break on refactor): `wrapper.cache_impl`, `wrapper.decoder.blocks`,
  `blocks[0].attn.max_cache_length / .kv_num_heads / .head_dim` (:104-117); `wrapper(token, pos_id[, kc, vc])`
  call contract and `(logits, new_kc, new_vc)` return (:137-140).
All verified present: mha.py:561 `max_cache_length`, common.py:124/145, convert.py:269 and options keys 309-338.
Does not mention `io_paged` (exists in common.py) though the docstring lists only 3 modes.

## Import verification (grep against transformersurgeon/)
All still resolve: Qwen2 / Qwen2VL / Qwen2_5_VL `*Compress` + `*CompressionSchemesManager` classes
(`models/qwen2_c/define_qwen2_c.py:22,30`, `qwen2_vl_c/define_qwen2_vl_c.py:33,42,51`,
`qwen2_5_vl_c/define_qwen2_5_vl_c.py:35,44,53`), `hf/hf_export.py:20 export_to_hf`, `utils/convert.py:269`,
`export/common.py:124,145`. Manager methods: set :143, init_vcon :479, set_vcon_beta :502,
set_calibration_data :527, apply :701, restore :978, iter_filtered :1175, update_config :1229
(`transformersurgeon/utils/manager.py`). No stale imports found. Not executed (needs model downloads).

## Duplicated library logic
- 02:47-73 `_build_wikitext_calibration_loader` is copy-pasted (near identical) in
  `scripts/executorch/qnn/full_quant_export.py:245`, `scripts/executorch/qnn/exporter_function_test.py:227`,
  `scripts/executorch/xnnpack/exporter_function_test.py:154`, and `benchmarks/nonhom_*/run_benchmark.py:build_loaders`.
  No library helper exists (`transformersurgeon/calibration/` could host one).
- `param_count` redefined in 01:34, 02:76, 03:73.
- 04:99-117 `describe_interface` re-derives cache I/O contract that lives in `LLMWrapper`/decoder.
- 04:120-157 greedy token-by-token generate loop duplicates what the ExecuTorch runners/test scripts do.

## Model-specific hardcoding
- 01: Qwen2 classes; layer criteria `[0,"self_attn.q_proj"]`, `[1,"mlp.up_proj"]`; default rank 640 is sized for Qwen2-0.5B (hidden 896).
- 02: Qwen2 classes; `MLP_RANK_OVERRIDES` = 497 for up/gate/down (:30-34) (Qwen2-0.5B-specific); default rank 64.
- 03: Qwen2-VL / Qwen2.5-VL only; model names `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`;
  criteria `["language_model","mlp.down_proj",26/27]` assume >=28 decoder layers and the `language_model` submodule name;
  `tokenizer("Hello")` text-only forward on a VLM.
- 04: Qwen2ForCausalLMCompress only; uses `tokenizer.apply_chat_template`; `model.lm_head` and
  `model.get_input_embeddings()` assumed (breaks for tied/different heads or VLMs).
Generic-factory refactors (e.g. AutoCompress / model registry) will touch every one of these class imports.

## Stale / superseded assessment
- No file references a removed API. None are dead.
- 03 is the weakest: `Config*Compress` returned but unused (:46-70 -> :84), `prepare_for_save` skipped, subdir
  discovery hack, and llms.txt blurb says "push to Hub" (inaccurate).
- 04 lacks docs/mkdocs/llms.txt entry (docs gap, not staleness).
- Examples are not exercised by tests (`grep` of test/ finds no reference), so API breakage will be silent;
  suggest a smoke test or at least an import check after refactor.
