# Recap: vlm-modules-importance/

## 1. File tree (4 files, 288 lines; all 4 git-tracked)
```
vlm-modules-importance/
  main_eval_bi.py                      43   driver script (top-level code, no functions)
  importance_eval_methods/BI.py       163   Block Influence scoring
  indexing/qwen_indexing.py            21   QWEN2_5_VL_IMP_INDEXING dict
  utils/utils.py                       61   sort_bi_scores
```
No `__init__.py`, README, or tests. Missing: `data/QwenVL_calibDataset.py` (imported by
main_eval_bi.py: `QwenVLCalibDataset`, `collate_fn_qwen`) -- not in this folder nor anywhere
in the repo (find found none). So main_eval_bi.py cannot run as committed.

## 2. Purpose
Measures per-block importance of the Qwen2.5-VL vision tower and text decoder via "Block
Influence" (BI; cosine-similarity-based, as in ShortGPT) to decide what to prune/compress.
Output: sorted lists of blocks and of attn/mlp sub-blocks for vision and text.

- main_eval_bi.py: loads `Qwen/Qwen2.5-VL-7B-Instruct` processor/tokenizer and
  `Qwen2_5_VLForConditionalGenerationCompress`, builds COCO QA calibration DataLoader
  (hardcoded `/ibex/user/antonic/CESOIA/datasets/COCO/...` paths, batch 4), calls
  `eval_bi`, `sort_bi_scores`, prints four ranked lists.
- importance_eval_methods/BI.py: `compute_bi(x_i,z_i,y_i)` (1 - cos sim between block
  input/output, per layer and for attn/mlp), `get_module_by_name(model, path)`,
  `register_hooks(model, indexing)` (forward hooks at paths from the indexing dict),
  `run_vlm_inference(model, processor, batch)` (uses `qwen_vl_utils.process_vision_info`),
  `eval_bi(model, loader, indexing, processor)` -> {"vision":{i:{layer_bi_score,attn_bi_score,mlp_bi_score}}, "text":{...}}.
- indexing/qwen_indexing.py: dict with keys 'vision'/'text': config_attr, num_blocks_attr,
  path_list, path_template (`model.visual.blocks.{block_index}.{path}`,
  `model.language_model.layers.{block_index}.{path}`). Standalone copy in the same style as the
  library's model INDEXING, but not the library's.
- utils/utils.py: `sort_bi_scores(final_bi)` -> four descending-sorted lists.

## 3. transformersurgeon imports / API
Only one, in main_eval_bi.py L7-9:
- `from transformersurgeon import Qwen2_5_VLForConditionalGenerationCompress` -> EXISTS
  (models/qwen2_5_vl_c/define_qwen2_5_vl_c.py, re-exported through `models/__init__.py` with
  defensive import; failure to import silently drops the name).
It is used only as a vanilla `from_pretrained` model; no compression manager used. BI.py,
utils.py, qwen_indexing.py import nothing from the library (torch, qwen_vl_utils only). Other
repo code does not reference this folder. Imports in main are relative to the folder
(`indexing.`, `importance_eval_methods.`, `utils.`, `data.`) so it must be run with cwd
= folder; `utils` also shadows the name of the library's `transformersurgeon/utils`
conceptually (not an actual clash).

## 4. Vendored / side project?
Side project, not vendored. No licenses/headers/copyright; code is local-authored (same
author paths `/ibex/user/antonic/CESOIA`). The BI metric itself follows the published
ShortGPT "Block Influence" idea (reimplemented, no copied header). Hyphenated folder name
is not importable as a Python package.

## 5. Recommendation: KEEP the logic, MOVE it (do not leave at repo root)
- It is a small (288 lines), original, useful analysis tool (importance ranking feeding
  pruning decisions) and its only library dependency is a model class that still exists.
- But it is outside `transformersurgeon/`, unimportable (hyphen, no `__init__`), missing its
  dataset module, and has hardcoded cluster paths. Options: (a) fold BI into the library
  as e.g. `transformersurgeon/analysis/block_influence.py` (generalise `eval_bi` to any model
  via the existing per-model INDEXING instead of the duplicate `QWEN2_5_VL_IMP_INDEXING`),
  with main_eval_bi.py becoming `examples/`; or (b) move whole folder under `examples/` or
  `benchmarks/`. Either way recover or rewrite `QwenVL_calibDataset` (ask the author; may
  exist only on the IBEX cluster) and parametrise paths/model name.
- Lower priority than qwen-vl-finetune; no refactor-blocking coupling beyond the
  `Qwen2_5_VLForConditionalGenerationCompress` export name.
