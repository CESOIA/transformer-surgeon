# Recap: qwen-vl-finetune/

## 1. File tree (line counts; all 21 files git-tracked, `git ls-files` = 21)
```
qwen-vl-finetune/
  README.md                       311  (tracked)  verbatim upstream Qwen-VL README
  ibex_train_3b.sbatch             23  (tracked)  KAUST IBEX SLURM wrapper
  train_3b.sh                      59  (tracked)  multi-GPU launcher
  train_3b_single_GPU.sh           46  (tracked)  single-GPU launcher -> train_qwen_c.py
  scripts/sft.sh, sft_7b.sh, sft_32b.sh   65 each (tracked)  upstream launchers -> train_qwen.py
  tools/check_image.py            168  validate_data(json, media_folder)
  tools/pack_data.py              191  read_data, write_data, DataArguments, MultimodalProcessor,
                                       calculate_tokens, pack_data (binpacking)
  demo/images/{10095.png,10149.png,COCO_train2014_000000580957.jpg}  binary
  demo/videos/*.mp4 (3 files)     binary (wc counts meaningless; ~MBs)
  qwenvl/train/argument.py         38  ModelArguments, DataArguments, TrainingArguments
  qwenvl/train/trainer.py         418  flash-attn monkeypatch (_flash_attention_forward,
                                       _update_causal_mask, replace_qwen2_vl_attention_class),
                                       print_trainable_parameters(_visual), create_optimizer
  qwenvl/train/train_qwen.py      179  upstream entry: rank0_print, safe_save_model_for_hf_trainer,
                                       set_model, train()
  qwenvl/train/train_qwen_c.py    777  LOCAL entry: VCON_cancel, VCON_beta_upd callbacks, train()
  qwenvl/train/test_messages.py    15  hardcoded `messages` list (abs path /ibex/user/antonic/...)
  qwenvl/train/example_image.jpg  binary
```
Missing (referenced but NOT in folder): `qwenvl/data/` (`data_qwen.py`, `data_qwen_packed.py`,
`rope2d.py`, `__init__.py` dataset configs), `tools/process_bbox.ipynb`. So the package is
NOT runnable as committed: `from qwenvl.data.data_qwen import ...` fails (train_qwen.py L38-39,
and train_qwen_c.py similarly). No `__init__.py` in `qwenvl/` or `qwenvl/train/`.

## 2. Purpose
Fine-tuning harness for Qwen2/2.5-VL (SFT with DeepSpeed). Upstream Qwen-VL trainer plus one
local script, `train_qwen_c.py`, which fine-tunes a *compressed* Qwen-VL (low-rank decomposition
of layers, optionally VCON blocks that blend original and compressed branches with a decaying
beta) -- i.e. post-compression recovery training.

Per file:
- train_qwen.py: upstream. Parses args, loads Qwen2(.5)-VL, selects frozen/trainable parts
  (vision/mlp/llm), builds data module, HF Trainer, saves.
- train_qwen_c.py (the only transformersurgeon-aware file): hardcoded config block
  (`model_type = "qwen2_5_vl_c"`, `use_vcon`, `vcon_beta`, `vcon_epochs`, model name
  Qwen2.5-VL-3B), builds `managerClass(model)`, ~18 hand-written `mgr.set("lrd","rank",N,[paths])`
  calls (L159-416, tab-indented, contains commented-out blocks), optional VCON init (L677-678),
  `mgr.apply(verbose=True)` (L680), then trains. Callbacks VCON_cancel (cancels VCON at end of
  vcon_epochs) and VCON_beta_upd (linearly decays beta per grad-acc step). Imports
  `from test_messages import messages`, `qwen_vl_utils.process_vision_info`, and
  `from trainer import replace_qwen2_vl_attention_class` (relative import, plus
  `import qwenvl.train.trainer` with sys.path hack) -- fragile.
- trainer.py: replaces Qwen2-VL/2.5-VL attention with flash_attn varlen (needs flash_attn).
- tools/*: dataset checking/packing; no relation to transformersurgeon.
- shell/sbatch: hardcoded cluster paths (`conda activate /ibex/user/antonic/CESOIA/env`,
  `module load cuda/12.2`); train_3b*.sh are local adaptations of scripts/sft.sh.

## 3. transformersurgeon imports / API (train_qwen_c.py only)
| Call | Where | Exists now? |
|---|---|---|
| `transformersurgeon.Qwen2VLForConditionalGenerationCompress` | L113-117 | YES (models/qwen2_vl_c/define_qwen2_vl_c.py, via `models/__init__.py` safe star-import) |
| `Qwen2VLConfigCompress`, `Qwen2VLCompressionSchemesManager` | idem | YES |
| `Qwen2_5_VLForConditionalGenerationCompress`, `Qwen2_5_VLConfigCompress`, `Qwen2_5_VLCompressionSchemesManager` | L126-130 | YES (models/qwen2_5_vl_c/define_qwen2_5_vl_c.py) |
| `managerClass(model)` ctor | L151 | YES (subclass ctor takes only `model`) |
| `mgr.set("lrd","rank",N,criteria)` | L159+ | YES (utils/manager.py `set(compression, property, value, criteria=None, group=None, verbose=False)`); path-list criteria format `["language_model.layers.17","mlp.up_proj"]` unverified vs. current INDEXING (text path is `model.language_model.layers.{i}`) |
| `mgr.init_vcon(verbose, criteria)`, `mgr.set_vcon_beta(beta, criteria)`, `mgr.cancel_vcon()` | L53,77,90,677-678 | YES (manager.py L479/502/490) |
| `mgr.apply(verbose=True)` | L680 | YES (manager.py L701) |
Note: `models/__init__.py` swallows import errors (`_safe_star_import` returns []), so a missing
optional dep silently removes the Qwen names from the top-level namespace -> ImportError here.
No other transformersurgeon import anywhere in the folder. Nothing in the library or tests/docs
references qwen-vl-finetune (grep of repo outside the two folders: no hits).

## 4. Vendored third-party?
YES, mostly. Evidence:
- train_qwen.py header: "Adopted from https://github.com/lm-sys/FastChat ... Adopted from
  tatsu-lab@stanford_alpaca ... Apache License 2.0" (Copyright 2023 Rohan Taori et al.).
- README.md is the upstream "QwenVL Training Framework" README (structure, version pins
  torch 2.6.0 / transformers 4.50.0.dev0 / deepspeed 0.16.4 / flash_attn 2.7.4).
- Directory layout `qwenvl/{train,data}`, `scripts/sft*.sh`, `tools/pack_data.py`,
  `tools/check_image.py`, demo/ media = upstream QwenLM/Qwen2.5-VL `qwen-vl-finetune`.
- No LICENSE file in the folder (Apache-2.0 notice only in one header; trainer.py,
  argument.py, pack_data.py have no headers). Upstream license obligations not fully met.
- Local (author antonic, KAUST IBEX) additions: train_qwen_c.py, test_messages.py,
  example_image.jpg(?), ibex_train_3b.sbatch, train_3b*.sh.
Git history touching it: 24b99fb "some extra tests and fix problems with QWEN-VL fine-tuning",
0962e92, fadc8f4.

## 5. Recommendation: MOVE OUT (of the library repo); archive the vendored part
- Not part of package `transformersurgeon/`; not imported by library/tests; not installed by
  setup.py. Broken as-is (missing qwenvl/data, hardcoded /ibex paths, tab/space mixed code).
- Large binaries tracked (demo mp4s/images) bloat the repo.
- Keep only what is genuinely ours, ideally as an `examples/` script or separate repo:
  `train_qwen_c.py` logic (compressed-model + VCON recovery fine-tuning). Rewrite it to
  the current manager API: replace the 18 hardcoded `mgr.set` blocks with a config/loop, remove
  sys.path hacks, drop test_messages import, parameterise paths.
- Drop/archive upstream copies (README, scripts/sft*.sh, tools, demo, trainer.py, argument.py,
  train_qwen.py) and instead depend on / point to QwenLM/Qwen2.5-VL; if kept, add LICENSE +
  NOTICE for Apache-2.0.
- If the refactor renames manager API (`set`, `apply`, `init_vcon`, `set_vcon_beta`,
  `cancel_vcon`) or Qwen*Compress exports, this is the only consumer to update.
