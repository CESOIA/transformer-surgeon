# Recap: transformersurgeon/hf/ and transformersurgeon/__init__.py

## 1. Purpose and call flow
`hf/` = Hugging Face-side export: save a (compressed) `*_c` HF model with `save_pretrained`, patch `config.json` for
`trust_remote_code`/`auto_map`, write an `autoload.py` that imports the custom classes from `transformersurgeon`, optionally
copy the whole package next to the model and upload to the Hub. Top-level `__init__.py` re-exports all model families plus two
optional helpers.
Flow `export_to_hf(model, repo_id, ...)`: (if `manager`) `manager.apply(hard=True)`; `manager.prepare_for_save()` ->
`out_dir/<repo name>` -> `model.save_pretrained` -> optional tokenizer/processor from `base_model` -> optional README ->
patch config.json (+`autoload.py`) -> optional `shutil.copytree(transformersurgeon)` -> optional `HfApi.upload_folder` if `token`.
Used only for the HF round trip (not for ExecuTorch/ONNX/TensorRT, which live in `export/`).

## 2. File tree (lines)
```
transformersurgeon/__init__.py   10  (tab-indented, no trailing newline)
transformersurgeon/hf/__init__.py 0 by wc (single line "from .hf_export import *", no trailing newline)
transformersurgeon/hf/hf_export.py 109
```

## 3. Per file
**hf/__init__.py**: `from .hf_export import *` (no `__all__` in hf_export, so star-export = every public name incl. leaked `os, json, shutil, HfApi, create_repo, PreTrainedModel, export_to_hf`; underscore `_infer_auto_map` not exported).

**hf/hf_export.py**
- `_infer_auto_map(model) -> dict` (:6): returns `{"AutoConfig": "autoload.<ConfigCls>", "AutoModel": "autoload.<ModelCls>"}`; if `"ForConditionalGeneration"` in class name also adds `AutoModelForConditionalGeneration` (string heuristic on class name, :14). Note: no `AutoModelForCausalLM` entry for `*ForCausalLMCompress` (guess: loading with AutoModelForCausalLM would not resolve; only AutoModel does).
- `export_to_hf(model, repo_id, manager=None, base_model=None, readme=None, out_dir=None, embed_code=False, token=None, private=False, exist_ok=False) -> out_dir` (:20): see flow. Quirks: output dir = `join(out_dir or "", repo_id.split("/")[-1])` (so `out_dir` is a parent dir); `AutoTokenizer` and `AutoProcessor` failures swallowed by bare `except Exception: pass` (:44-52; `AutoProcessor` imported twice); config.json forced `trust_remote_code=True`; `embed_code` copies `ts.__path__[0]` (fails with a printed warning if dir exists); repo creation errors only printed; `ignore_patterns=["__pycache__"]`. No registry.

**transformersurgeon/__init__.py**
- `from .models import *` (all families via `models/__init__.py::_safe_star_import`, each wrapped in try/except so a broken family is silently missing).
- `precompute_rope_inv_freqs` from `.blocks` (rope.py:154) in try/except -> None on failure.
- `convert_for_export` from `.utils` in try/except -> None on failure.
- Does NOT import `hf`, `calibration`, `export`, `compression`, `utils` managers eagerly (except via models/blocks/utils side effects).

## 4. Dependencies
`hf_export.py` imports: `huggingface_hub (HfApi, create_repo)`, `transformers.PreTrainedModel` (+ lazy `AutoTokenizer`, `AutoProcessor`), stdlib. No intra-package imports at module level; it dynamically does `import transformersurgeon as ts` (:88) and emits strings importing `from transformersurgeon import <ModelCls>/<ConfigCls>` into generated `autoload.py` (:82-83) -> requires those names to be top-level exports of `transformersurgeon` (true only for classes re-exported through `models/__init__.py`).
Duck-typed manager API: `manager.apply(hard=True)`, `manager.prepare_for_save()` (`utils/manager.py`).
Imported by: `hf/__init__`; `test/e2e/test_export_pipelines.py:79,85` (`from transformersurgeon.hf import export_to_hf`); `examples/03_hf_export_roundtrip.py:28,122`; docs (`docs/examples/03_hf_export_roundtrip.md`, `docs/index.md:43`, README.md:33, AGENTS.md:88,508, CLAUDE.md:46, FRAMEWORK_STRUCTURE.md:56,150). NOT imported by anything in `transformersurgeon/` itself, scripts/, or benchmarks/. NOT re-exported from top-level `__init__` (README claims `export_to_hf` is a package-level API; code requires `transformersurgeon.hf`).
Top-level `__init__` consumers: everything doing `import transformersurgeon as T` / `from transformersurgeon import ...`: test/_helpers/model_factory.py:42, test_lpbq_quantization.py:12, test_rope_variants.py:12, test_attention_semantics.py:10, test_paged_attention.py:15, test_block_pruning.py:11, test_export_pipelines.py:34, test_model_families.py:145 (`convert_for_export`), benchmarks/nonhom_{quant,lrd}/run_benchmark.py (`Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress`), generated `autoload.py` files. `precompute_rope_inv_freqs` top-level re-export has NO consumer: test_add_batch_dim.py:13-16 imports it from `transformersurgeon.blocks.rope`.

## 5. Violations / smells
- Model names: `hf_export.py:14` hard-codes the `"ForConditionalGeneration"` class-name heuristic and the `Auto*` mapping (no per-family registry; comment :16-17 "add more heuristics here"). Should come from model-family metadata in `models/` (each family declares its Auto classes).
- Compression application inside export: `hf_export.py:42-44` calls `manager.apply(hard=True)` + `prepare_for_save()` (compression logic triggered from an export function; also `hard=True` hard-codes the policy). Better: caller (or a `manager.finalize_for_save()`) does it; `export_to_hf` only serializes.
- Export logic is in `hf/` (OK for this folder), but it is separate from the `export/` backend package -> two "export" locations (guess: should become `export/hf/` backend or stay as "save" utility).
- `__init__` top-level: swallowing import errors (`except Exception`) hides real bugs (also in models/__init__). `convert_for_export`/`precompute_rope_inv_freqs` re-exports: `convert_for_export` is export logic exposed at top-level; `precompute_rope_inv_freqs` is a block helper with no external users in package code.
- Package embed (`embed_code`) copies the live package dir (including `__pycache__`, tests?) into the model folder: fragile (:86-91).
- `import os, json, shutil` etc. star-exported (no `__all__`).

## 6. Duplicated code
- Config/class-name -> autoload strings duplicated between `_infer_auto_map` (:7-15) and the generated autoload.py text (:80-84); derive both from one helper.
- Possible overlap with `export/` code that writes configs/manifests (not inspected in depth; unknown). `AutoTokenizer`/`AutoProcessor` save logic likely similar in `export/common.py` (unverified guess).
- `models/__init__.py::_safe_star_import` + top-level try/except pattern repeated three times.

## 7. Stale / unused
- `hf/__init__.py` star import: only symbol used externally is `export_to_hf`.
- `PreTrainedModel` import used only as type hint. `AutoProcessor` imported twice (:46 inside first try, :51).
- `hf/` dirs `test/hf_export_tests/` and `test/compression_tests/` contain only `__pycache__` (no tests) -> stale test dirs (verified by `ls`).
- `precompute_rope_inv_freqs` top-level re-export: no importer via the top-level name anywhere (test_add_batch_dim.py:13 imports from `blocks.rope`); blocks internals use `blocks/rope.py` directly.
- No `__version__` in top-level `__init__`.

## 8. Test coverage
- `export_to_hf`: one e2e test `test/e2e/test_export_pipelines.py::test_hf_export_roundtrip` (:76-85): Qwen2 tiny model, `manager=mgr`, `exist_ok=True`, local dir only; (LRD rank 64 on one layer; asserts dir exists, `from_pretrained(saved)` reload and a forward).
- Untested: `_infer_auto_map` (esp. `ForConditionalGeneration` branch, qwen2_vl_c only via example `examples/03_hf_export_roundtrip.py --model-type qwen2_vl_c`), `base_model` tokenizer/processor save, `readme`, `embed_code=True`, Hub upload (`token`, `private`, `exist_ok` repo handling), `manager=None` path, config patching of non-existing config.json.
- Top-level `__init__`: implicitly exercised by every `import transformersurgeon as T`; the fallback-to-None branches are untested.
