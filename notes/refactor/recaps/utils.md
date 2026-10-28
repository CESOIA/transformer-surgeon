# Recap: transformersurgeon/utils/

Snapshot: manager.py and convert.py were modified 2026-10-28 (most recent); all other files Jun/Sep. Line numbers below refer to this snapshot.

## 1. Purpose

Despite the name, `utils/` is NOT a utilities folder. It holds (a) the whole user-facing orchestration layer of compression (`CompressionSchemesManager`, `CompressionScheme`, `SchemeGroup`, block-wise cascade calibration), (b) HF-model construction helpers (`init_compressed_config`, `replace_layers_upon_init`), (c) the HF-model -> `blocks/` converter for export (`convert_for_export`), and (d) only a handful of true generic helpers (`get_submodule`, `flatten_index_paths`, `infer_model_device`, `move_to_device`, progress bar). Only the last group satisfies the target rule "layer handling only".

## 2. File tree (lines)

```
utils/
  __init__.py        7   star-imports grouping, scheme, configuration, manager, modeling, convert, utils (NOT cascade, NOT interface)
  utils.py         105   generic helpers (get_submodule, flatten_index_paths, device helpers)
  interface.py      48   progress bar (calibration-specific text)
  grouping.py       73   SchemeGroup
  configuration.py  74   init_compressed_config
  modeling.py      181   replace_layers_upon_init (+ private helpers)
  scheme.py        627   CompressionScheme
  manager.py      1280   CompressionSchemesManager
  cascade.py       994   apply_cascade + private helpers
  convert.py       424   convert_for_export + private helpers
  (total 3813)
```

## 3. Per-file contents

### utils.py (105)
- `flatten_index_paths(path_list) -> list[str]` (l.6): normalises indexing `path_list` (list/tuple, or dict `{parent: [children]}`; empty children => parent itself) to a flat list of dotted paths. Raises TypeError on bad shape.
- `get_submodule(module, submodule_path) -> Module` (l.34): dotted getattr traversal; raises ValueError if any piece is missing/None. (Note: `getattr(...) is None` also trips on legitimately-None attributes; does not support ModuleList int indexing except through getattr on "0" - works because nn.ModuleList registers children as attrs "0","1"...)
- `infer_model_device(model) -> torch.device` (l.54): model.device -> first param -> first buffer -> cpu.
- `move_to_device(value, device)` (l.77): recursive move of tensors/Mapping/tuple/list/objects with `.to`.
- `__all__` = those four. Pure, no repo imports (only torch). This file is already compliant.

### interface.py (48)
- `FallbackProgressBar(total, enabled)` with `update(n)`, `close()` (l.9): print-based bar; strings hard-code "Calibration".
- `build_progress_bar(*, enabled, total, verbose)` (l.34): returns tqdm (desc "Calibration") or fallback.
- Calibration-specific wording (violates "no compression ref" mildly). Not exported through `__init__`.

### grouping.py (73)
- `SchemeGroup(name, schemes=None)` (l.16): holds list of `CompressionScheme` pointers + `properties` dict. Methods: `add(scheme)`, `remove(scheme)`, `set_property(key, value)`, `get_property(key, default=None)`, `reset()`, `__iter__`, `__contains__`, `__len__`, `__repr__`. Duck-typed (no imports). Docs say "model- and compression-agnostic" but docstring talks about pruning masks.

### configuration.py (74)
- `init_compressed_config(config_instance, indexing, compression_config=None) -> None` (l.7): builds all layer paths from `indexing` (`path_template`, `num_blocks_attr`, `path_list`), then for every path x every compression type in `COMPRESSION_REGISTRY` x every property fills default (`meta["default"]`) via `setdefault`, runs `meta["validator"]`, and sets `config_instance.compression_config`. Mutates the passed-in dict.
- Imports `..compression.COMPRESSION_REGISTRY` (compression-type knowledge by iteration; no names hard-coded).

### modeling.py (181)
- `replace_layers_upon_init(model, indexing, config)` (l.9): during `__init__` of a `*_c` HF model, walks `indexing['path_list']` x blocks, replaces exact `torch.nn.Linear` with `LinearCompressed(rank=lrd rank from config['compression_config'][path]['lrd']['rank'])` (inline kwargs, l.60-81); then for singleton entries `indexing['preprocessing']` (Embedding -> EmbeddingCompressed), `['final_layer']` (Linear -> LinearCompressed), `['preprocessing_conv']` (Conv2d/Conv3d -> Conv2dCompressed/Conv3dCompressed) calls `_replace_singleton_layer`.
- `_CONV_TYPE_MAP` (l.97), `_conv_new_module_kwargs(old, cfg)` (l.103), `_linear_new_module_kwargs` (l.118), `_embedding_new_module_kwargs` (l.130), `_replace_singleton_layer(model, full_path, type_map, get_config, config_dict)` (l.141; warns and skips on type mismatch).
- Imports `..blocks` (LinearCompressed, EmbeddingCompressed, Conv2dCompressed, Conv3dCompressed).

### scheme.py (627) - `CompressionScheme`
Binds one model path to a set of compressors. Constructor (l.51): `(name, block_id, path, compression_config=None, model=None, pruning_indexing=None, path_template=None, path_list=None, num_blocks=None, preprocessing_path=None, final_layer_path=None, final_norm_path=None, preprocessing_conv_extra_params_paths=None, is_norm=False)`. Builds `self.compressors` from `COMPRESSOR_DICT[name](cfg)`, sorts by `APPLICATION_ORDER`, links back-reference `compressor.scheme = self`, derives `position_linked` from `pruning_indexing['position_linked']`. State flags: `soft_applied`, `hard_applied`, `vcon_initialized`, `groups` (dict, max 1), `calibration_data` (dict of summaries), `_weight_original`.
Public/private methods:
- `get_module()` l.147: `get_submodule(model, path)`.
- `set(compression_name, compression_property, value, verbose=False)` l.157: validates via `COMPRESSION_REGISTRY` validator; refuses after apply; lazily creates compressor with registry defaults; norm schemes (`is_norm`) are silent no-op.
- `_link_compressors()` l.209; `add_to_group(group)` l.219 (enforces <=1 group); `remove_from_group(name)` l.229; `_sort_compressors()` l.233.
- `set_module(new_module)` l.246: manual parent traversal + setattr; forbidden while VCON active. (Duplicates get/set submodule; no `set_submodule` helper in utils.py.)
- `get_compression_module()` l.272: module, or `.block_b` if VCON.
- `_is_to_compress()` l.283: any compressor `_to_compress()` and `_is_compatible()`.
- `_is_compatible()` l.299: type is exactly Linear/Embedding/Conv2d/Conv3dCompressed or VCON on.
- `_module_copy(module)` l.320: signature-introspection deep copy of a module (init kwargs from attributes + copy params/buffers/plain attrs).
- `init_vcon(verbose)` l.364 / `cancel_vcon(keep_block_b=True, verbose)` l.398 / `set_vcon_beta(beta, verbose)` l.436 / `freeze_uncompressed_vcon(verbose)` l.466: wrap module in `VCONBlock(block_a=orig, block_b=copy)`, unwrap, set beta, freeze block_a.
- `apply(hard=False, verbose=False)` l.489: skip if not `_is_to_compress`; backup `weight` for soft; for each compressor in application order: `compressor.set_calibration_store(self.calibration_data)`; `compressor.apply(module, hard, soft_applied)`; set flags.
- `restore(topology=False, verbose=False)` l.536: `compressor.restore(module)` each; restore weight unless `topology`; error if hard-applied.
- `reapply_masks()` l.576, `remove_masks()` l.592: delegate `compressor.reapply_mask / remove_mask`.
- `__repr__` l.614.
Imports: `..blocks` (LinearCompressed, EmbeddingCompressed, Conv2dCompressed, Conv3dCompressed, VCONBlock), `..compression` (COMPRESSOR_DICT, COMPRESSION_REGISTRY, APPLICATION_ORDER), `.utils.get_submodule`. Unused imports: `Union` (l.9).

### manager.py (1280) - `CompressionSchemesManager`
Module constant `_SENTINEL_COMPRESSED_TYPE` (l.31): {"preprocessing": (EmbeddingCompressed,), "final_layer": (LinearCompressed,), "preprocessing_conv": (Conv2dCompressed, Conv3dCompressed)}.

`__init__(model, indexing)` l.55: stores model, `model.config` (asserts PretrainedConfig), indexing (dict block_name -> block indexing), `calibration_groups` from indexing, `schemes = _generate_schemes()` (nested {block_name: {full_path: scheme}}), `groups = {}`; also `calibration_data=None`, `calibration_loss_fn=None`, `calibration_mode="standard"`.

Methods (public first):
- `set(compression, property, value, criteria=None, group=None, verbose=False)` l.143: declarative setter; resolves target schemes (by `iter_filtered` or by group; group-only options in `GROUP_OPTIONS` need `group=`), calls `scheme.set`; for group options first `_reset_compression_config`.
- `create_group(schemes, name=None) -> SchemeGroup` l.250: validates single indexing block and single-group-per-scheme; `delete_group(name)` l.294.
- `auto_groups() -> dict` l.319: from `indexing[block]['pruning']['coupled_masks']` (per block) and `['coupled_masks_all']` (cross-block) creates groups.
- `smooth_activations(alpha=0.5, criteria=None, max_batches=None)` l.302: lazily imports `compression.smoothquant.apply_smoothquant`; stores `self.smoothing` (never read elsewhere).
- VCON passthroughs: `init_vcon(criteria, verbose)` l.479, `cancel_vcon(keep_block_b, criteria, verbose)` l.490, `set_vcon_beta(beta, criteria, verbose)` l.502, `freeze_uncompressed_vcon(criteria, verbose)` l.515.
- Calibration config: `set_calibration_data(loader)` l.527 (DataLoader only), `set_calibration_loss(fn)` l.544, `set_calibration_mode("standard"|"cascade")` l.555.
- `run_calibration(shifted_model=None, criteria=None, max_batches=None, device=None, offload_to_cpu=True, verbose=False, show_progress=True, summary_dump_dir=None, summary_dump_names=None)` l.580: builds stages via `collect_calibration_target_stages` and calls `calibration.run_compression_calibration`.
- `extract_summaries(criteria, summaries, shifted_model, max_batches, device, offload_to_cpu, verbose, show_progress)` l.626: runs calibration forcing given summary names, returns `{path: {summary: value}}`.
- `clear_calibration(criteria)` l.691.
- `apply(hard=False, shifted_model=None, criteria=None, verbose=False, max_batches=None, device=None, offload_to_cpu=True, show_progress=True, summary_dump_dir=None, summary_dump_names=None)` l.701: MAIN FLOW: `_validate_grouping` -> `_validate_coupling` -> if mode=="cascade": `apply_cascade(manager=self,...)` and return; else collect calibration targets; if any: require calibration_data, `run_calibration`; then `scheme.apply(hard, verbose)` for each filtered scheme (in `self.schemes` iteration order = block by block, path_list order, then sentinels).
- `collect_calibration_target_stages(criteria)` l.837: standard mode => one stage; cascade mode => stages per `calibration_groups` then per remaining scheme.
- `save_state(path)` l.889 / `load_state(path)` l.905: torch.save/load of `{scheme.path: scheme.calibration_data}` (`weights_only=False`).
- `prepare_for_save(criteria=None)` l.925: dequantizes torchao weights, calls `Quantizer.strip_runtime_state` (lazy import `..compression.quantization.Quantizer`).
- `restore(topology=False, criteria, verbose)` l.978; `reapply_masks(criteria)` l.990; `remove_masks(criteria)` l.1014.
- `print_filtered(criteria)` l.1165; `iter_filtered(criteria)` l.1175 (criteria language: None/"all"; int = block_id; str = substring of name or path; list = OR of items, nested list = AND); `update_config(verbose)` l.1229 (EMPTY body, only docstring); `__iter__` l.1239; `__len__` l.1251; `__repr__` l.1257; `_set_model(model)` l.1270.
- Private: `_get_calibration_groups_from_indexing` l.83 (normalises indexing `calibration_groups` dict/list format), `_reset_compression_config` l.198, `_find_scheme(path)` l.216, `_scheme_block_name` l.223, `_resolve_scheme` l.230, `_next_group_name` l.243, `_resolve_coupled_masks_all_paths` (static) l.373, `_resolve_coupled_group_paths` l.394, `_iter_coupled_groups` l.408, `_validate_coupling(hard, criteria)` l.440 (delegates to `compressor.check_coupling(hard, siblings)`), `_validate_grouping` l.467, `_collect_calibration_targets` l.772, `_collect_calibration_targets_for_schemes` l.787 (asks each compressor `needs_calibration()` for summary names), `_generate_schemes` l.1026 (per indexing block: reads config via `config_attr`/`num_blocks_attr`, creates one `CompressionScheme` per (block, path) plus sentinels `preprocessing`, `preprocessing_conv`, `final_layer`, `final_norm`), `_resolve_sentinel_path` l.1134, `_resolve_final_norm_path` l.1151.
Imports: `..calibration` (run_compression_calibration, SUPPORTED_SUMMARIES), `..compression` (GROUP_OPTIONS, COMPRESSION_REGISTRY; lazy smoothquant, quantization), `..blocks` (4 compressed layer classes), `.scheme`, `.grouping`, `.utils`, `.cascade`. Unused imports: `deepcopy` (l.13), `List` partly, `Any` used. (Not verified with a linter; pyflakes unavailable. `deepcopy` verified by reading: no use.)

### cascade.py (994) - block-wise cascade calibration
Only public symbol: `apply_cascade(manager, hard=False, criteria=None, verbose=False, max_batches=None, device=None, offload_to_cpu=True, show_progress=True, summary_dump_dir=None, summary_dump_names=None)` l.24. Control flow: (1) select schemes; raise if a block family has `no_cascade_calibration=True` in indexing; (2) require `manager.calibration_data`; (3) per indexing block: collect block inputs (from `indexing['preprocessing']` module output via `_collect_preprocessing_outputs`, else raw loader inputs via `_collect_loader_inputs`), deepcopy into "uncompressed" and "compressed" flows, attach position embeddings (`_attach_position_embeddings_if_configured`); (4) per block id: build layer stages from `calibration_groups` (`_build_layer_stages_for_block`), clone block as `reference_block` (`_clone_module`), per stage: `manager._collect_calibration_targets_for_schemes`, localise scheme paths to the block (`_localize_targets_to_block`, uses inner class `_LocalCalibrationScheme`), run `run_compression_calibration(model=reference_block, shifted_model=block_module, ...)`, then `scheme.apply(hard)`; (5) propagate both flows through the block (`_run_block_on_cached_inputs`) with skip-connection residual bookkeeping (`_cache_skip_residuals`, `_apply_skip_residuals`, `_extract_cached_hidden_states`, `_add_residual_to_cached_outputs`), re-attach positional embeddings; (6) clear scheme calibration data; restore train/eval mode.
Private helpers: `_extract_path_entries` l.326, `_build_layer_stages_for_block` l.378, `_criteria_match_layer` l.457, `_criterion_matches_layer` l.472, `_resolve_block_container_module` l.495, `_clone_module` l.518, `_stage_schemes_from_layer_group` l.526, `_localize_targets_to_block` l.541, `_collect_preprocessing_outputs` l.586, `_collect_loader_inputs` l.626, `_run_block_on_cached_inputs` l.662, `_build_block_tokens` l.703, `_cache_skip_residuals` l.713, `_apply_skip_residuals` l.737, `_extract_cached_hidden_states` l.770, `_add_residual_to_cached_outputs` l.783, `_attach_position_embeddings_if_configured` l.819, `_call_preprocessing_module` l.900, `_detach_and_optionally_offload` l.975.
Reaches into manager private API: `manager._collect_calibration_targets_for_schemes` (l.190), and manager attributes `indexing, schemes, calibration_data, calibration_groups, calibration_loss_fn, model`. Imports: `..calibration` (run_compression_calibration, `calibration.raw_data.normalize_calibration_batch`), `.utils`, tqdm directly. Unused imports (by reading): `Any` (l.13), `Union` is used, `Sequence` used.

### convert.py (424) - `convert_for_export`
- `convert_for_export(model, options=None, verbose=False) -> dict[name -> block model or None]` l.168: for each component in `model.indexing` (e.g. "text", "vision"): skip with warning if no `structure`; read `config_attr`/`num_blocks_attr`; branch on `indexing['structure']`: `'transformer_decoder'` => build `CustomDecoderConfigCompress.from_source_config(...)` with `options` {use_sdpa, attn_impl, max_cache_len=2048, cache_impl='mutable', add_batch_dim, rmsnorm_prescale=True, rmsnorm_upcast, masked_softmax} and `TransformerDecoder(config)`; `'transformer_encoder'` => `CustomEncoderConfigCompress` (+use_final_norm, position_embedding_type from indexing) and `TransformerEncoder`; else NotImplementedError. Then per block and per `(path_list, layer_matching)` pair copies weights with `_load_state_dict_dequantized(old_layer, new_layer)`; after each block calls `attn.finalize_head_pruning()`, `finalize_rope_pruning()`, `finalize_cache_geometry()` if present; finally copies `extra_layers` (with `extra_layers_matching`). Uses `get_submodule`, `flatten_index_paths`.
- Private: `_QUANT_BUFFER_SUFFIXES` l.14, `_QUANT_PLAIN_ATTRS` l.19 (torchao/act-quant names hard-coded), `_extract_aqt_scale(w)` l.28, `_replace_param(root, dotted_key, ref)` l.42, `_load_state_dict_dequantized(old, new)` l.54 (dequantize quantized tensors, pre-register quant buffers, adopt pruned shapes, load_state_dict strict, copy in/out_features and `_torchao_*`/`_act_quant_*` attrs, derive `_torchao_per_channel`), `_flatten_layer_matching(indexing, source_paths)` l.133, `_build_bias_required_config(layer_matching, source_bias)` l.166 (hard-codes "attn."/"mlp." prefixes), `_build_converted_compression_config` l.195, `_CONVERTED_BLOCK_INDEXING_TEMPLATE` l.222, `_build_converted_decoder_indexing` l.228, `_build_converted_encoder_indexing` l.241.
- Imports `..blocks` (TransformerDecoder), `..blocks.encoder`, `..blocks.config`, `.utils`. Does NOT import `export/`; but the converter is the bridge to it.

## 4. Dependencies

Intra-folder import graph: `utils.py` <- {configuration, modeling, scheme, manager, cascade, convert}; `scheme` <- manager; `grouping` <- manager; `cascade` <- manager (manager calls apply_cascade; cascade calls back manager privates); `interface`: nobody inside utils.

Imports of other transformersurgeon packages from utils/ (all upward, risk of cycles):
- `..compression`: configuration.py (COMPRESSION_REGISTRY), scheme.py (COMPRESSOR_DICT, COMPRESSION_REGISTRY, APPLICATION_ORDER), manager.py (GROUP_OPTIONS, COMPRESSION_REGISTRY, lazily smoothquant.apply_smoothquant, quantization.Quantizer).
- `..calibration`: manager.py (run_compression_calibration, SUPPORTED_SUMMARIES), cascade.py (+ `calibration.raw_data.normalize_calibration_batch`).
- `..blocks`: modeling.py, scheme.py, manager.py (compressed layer classes, VCONBlock), convert.py (TransformerDecoder, TransformerEncoder, config classes).

Reverse (who imports utils) - grep over repo (transformersurgeon/, test/, scripts/, examples/, benchmarks/):
- `utils.utils` (get_submodule/infer_model_device/move_to_device): `calibration/backbone.py:20`, `compression/smoothquant.py:33`, `export/export.py:10`, lazy in `compression/block_pruning.py:92`, `compression/structured_pruning.py:543,669`. => compression/ and calibration/ depend on utils.utils, while utils imports compression/ and calibration/ -> circularity (hence the "lazy: avoid import cycle" comments).
- `utils.interface.build_progress_bar`: only `calibration/backbone.py:19`.
- `utils.configuration.init_compressed_config`: `blocks/config.py:5`; also via `from ...utils import (init_compressed_config, replace_layers_upon_init, CompressionSchemesManager)` in all `models/*_c/define_*_c.py` (qwen2, llama, qwen2_vl, qwen2_5_vl, bert, distilbert, modernbert, vit).
- `utils.convert_for_export`: `transformersurgeon/__init__.py:9` (try/except), `export/export.py:9`, tests (test_qnn_activation_only, test_qnn_bias_split, test_paged_attention, test_onnx_tensorrt_export, test_export_fixes, test_rope_variants, test_attention_semantics, test_block_pruning, test_gqa_rope_pruning (comment), e2e/test_gqa_attention_pruning, e2e/test_model_families via `from transformersurgeon import`), scripts (compress_infer_vit, compress_infer_qwen, executorch/qnn/full_quant_export, qnn/exporter_function_test, xnnpack/exporter_function_test), examples/04_cache_impl_export_interface.py. Comments in `blocks/mha.py:191,248,676`, `compression/block_pruning.py` mention it.
- `CompressionSchemesManager`: model `define_*_c.py` files, `hf/hf_export.py` (docstring only), README.md:154, docs (`docs/api/utils/manager.md` points to `transformersurgeon.utils.manager.CompressionSchemesManager`), AGENTS.md:575-577 and CLAUDE.md mention `utils/manager.py`, `utils/convert.py`, `utils/configuration.py`. Tests build managers through model factory/`define_*` classes (test/_helpers/model_factory.py, many unit tests).
- `CompressionScheme`: only docs (`docs/api/utils/scheme.md`) and comments in compression/*; no external code constructs it (only manager.py:1086,1113).
- `SchemeGroup`: referenced in `compression/structured_pruning.py:42,199` (docs/comments only, duck-typed), AGENTS.md.
- `apply_cascade`: manager.py:733, test/unit/test_cascade_calibration.py (name mention).
- `replace_layers_upon_init`: model `define_*_c.py`; `models/modernbert_c/indexing_modernbert_c.py` (comment).
- `docs/api/utils/{manager,scheme}.md` and `mkdocs.yml` hard-code the `transformersurgeon.utils.*` paths: update on move.

## 5. Violations of the target rule and proposed destinations

Rule: utils/ = layer get/set/replace/shape helpers only; no model type, no compression type, no export/backend.

| File | Violations (file:line) | Proposed destination |
|---|---|---|
| `utils.py` | none. (`flatten_index_paths` is indexing-metadata aware but generic.) | stays in `utils/` (maybe rename `utils/modules.py`; add `set_submodule`, see Sec.6) |
| `interface.py` | strings "Calibration batches", "Calibration:" l.20-26, desc "Calibration" l.41: calibration-specific. Only consumer is calibration. | `calibration/` (e.g. `calibration/progress.py`) or make generic (accept `desc` arg) and keep in utils |
| `grouping.py` | docstring only mentions pruning mask (l.6-12); code is agnostic. Semantically part of the compression-scheme layer (holds CompressionScheme pointers). | `compression/` (next to structured_pruning which consumes it) or the new `pipeline/`/`core/` with scheme |
| `configuration.py` | imports `COMPRESSION_REGISTRY` (compression-type knowledge, l.4, l.61); uses compression_config. | `compression/config.py` (or `models/` base). `blocks/config.py:5` imports it, so blocks->compression dependency would be introduced; alternative: keep in `models/common` |
| `modeling.py` | compression-type refs: `"lrd"`/`rank` l.64-68, 119, 131 and `compression_config` lookups; comment "pruning config" l.60; "Qwen-VL" in comment l.95. Uses `blocks.*Compressed` classes (blocks = frozen compressed graph; but also the compressed layer types). | split: generic part (walk path, resolve parent, `setattr` replace-with-type-map) -> `utils/`; the Linear->LinearCompressed + lrd-rank kwargs -> `compression/replace.py` (or `models/`) |
| `scheme.py` | compression names: whole class is compression semantics (COMPRESSOR_DICT, APPLICATION_ORDER, `pruning_indexing`, `position_linked` l.82-ish, `preprocessing_conv_extra_params_paths` ViT cls_token comment l.109-116, VCON, `reapply_masks`, `weight_mask`). Mentions "RoPE q/k projections", "llama_c", "Qwen-VL", "ViT" in comments (l.76-78, l.109-116). | `compression/scheme.py` (or new top-level `pipeline/` or `core/`; recommend a single place with manager+scheme+grouping+cascade). VCON bits (`init_vcon`, `cancel_vcon`, `_module_copy`) arguably `compression/vcon.py` |
| `manager.py` | compression: `GROUP_OPTIONS`, `COMPRESSION_REGISTRY`, `share_mask`/`reduce_op` strings in docs and error text (l.158, 202, 275, 475), `pruning.coupled_masks[_all]`, `pruning.norm_layers`, `smooth_activations` -> smoothquant (l.302-317), `prepare_for_save` -> torchao/Quantizer (l.925-976), VCON passthroughs (l.479-525), `reapply_masks`/`remove_masks`, `restore`. Model-name refs only in docstring examples (l.914-915 "Qwen2..."), l.1059 comment ("vit_c/qwen2_5_vl_c"). Export: `prepare_for_save` is HF-export prep; no backend refs. Calibration orchestration (l.527-700, 772-887) is calibration logic. CLAUDE.md says "Don't put model-specific logic in manager". | new `pipeline/manager.py` (recommended; name candidates `core/` or `compression/manager.py`). Candidate splits: calibration scheduling (`_collect_calibration_*`, `collect_calibration_target_stages`, `_get_calibration_groups_from_indexing`, `run_calibration`, `extract_summaries`, `save_state/load_state`) -> `calibration/` (e.g. `calibration/scheduler.py`); `prepare_for_save` -> `hf/` or `export/hf`; `smooth_activations` -> `compression/smoothquant`; scheme generation `_generate_schemes` + sentinel resolution + group builders (`auto_groups`, `_resolve_coupled_*`, `_iter_coupled_groups`) -> pipeline/ helper |
| `cascade.py` | calibration + compression semantics (calls `scheme.apply`); model refs: "Optional model-specific positional embedding injection" l.125 and `_attach_position_embeddings_if_configured` (rotary / `position_ids_ndim` for 3D mRoPE, `position_embeddings` kwarg: transformer-LM-specific) l.819-897; hard-coded input keys `"input_ids","pixel_values","input_values","inputs_embeds"` l.944-949; calls `manager._collect_calibration_targets_for_schemes` (private). | `calibration/cascade.py` (it is calibration scheduling). Generic tensor helpers `_detach_and_optionally_offload` (l.975), `move_to_device` -> utils |
| `convert.py` | Export: whole file is "convert for export" (docstring l.168-176 "ONNX"); `options` keys `use_sdpa`, `attn_impl`, `max_cache_len`, `cache_impl`, `rmsnorm_*`, `masked_softmax` l.200-208; `finalize_rope_pruning`/`finalize_head_pruning`/`finalize_cache_geometry` (RoPE/KV) l.396-410; compression: torchao/act-quant buffer names l.14-26, `_extract_aqt_scale`; `"attn."`/`"mlp."` prefixes l.173-179, 232-251 (architecture assumption); "rope_prune_mask" comment l.113. No explicit model-family names. | `export/convert.py` (or `models/convert.py`/`blocks/convert.py`; since it builds `blocks/` graphs and `export/export.py:9` is its only runtime caller, `export/` is the natural home). The dequantisation helper `_load_state_dict_dequantized` + `_QUANT_*` -> `compression/quantization` or `export/` |
| `__init__.py` | star-exports manager/scheme/convert: makes `utils` import the full pipeline. | after refactor: export only utils.py helpers |

Import-cycle note: utils/ currently imports compression/, calibration/, blocks/, while compression/ + calibration/ + export/ + blocks/ import utils/. After moving manager/scheme/cascade/convert out, `utils` becomes a leaf (only torch) which dissolves the "lazy: avoid import cycle" imports at `compression/block_pruning.py:92` and `compression/structured_pruning.py:543,669`.

## 6. Duplicated code

Within utils/:
1. Path-parent traversal (getattr over `path.split('.')[:-1]`, raise if None) is written 4 times: `modeling.py:50-56` and `modeling.py:149-153` (`_replace_singleton_layer`), `scheme.py:set_module` ~l.262-270, plus `convert.py:_replace_param` (l.42-51, uses `get_submodule` + rpartition). Also in `compression/structured_pruning.py:669-674` (parent via rpartition + `get_submodule`). Unify with new `utils.get_parent_and_name(model, path)` / `set_submodule(model, path, new)` / `replace_submodule` in `utils.py`; use in all.
2. Inline Linear kwargs in `modeling.py:60-81` duplicate `_linear_new_module_kwargs` (l.118-127) -> main loop can call `_replace_singleton_layer`-like helper with the Linear type_map; the two code paths differ only in config key (`full_path` vs block loop).
3. `_extract_path_entries` (`cascade.py:326`) re-implements `flatten_index_paths` (`utils.py:6`) grouped-dict expansion with the same error text and logic, but returns a mapping. Also `_get_calibration_groups_from_indexing` (`manager.py:83-141`) is another normaliser of the same "dict of subblock -> list" shape with the same prefixing rule (`f"{subblock}.{layer}"`). Unify: one `expand_grouped_paths(path_list) -> (OrderedDict subblock->[paths])`, with `flatten_index_paths` = concatenation.
4. `convert.py`: `_build_converted_decoder_indexing` (l.228) and `_build_converted_encoder_indexing` (l.241) are identical except the dict key ("decoder"/"encoder"); the `transformer_decoder` and `transformer_encoder` branches of `convert_for_export` (l.213-285) repeat `converted_indexing / converted_compression_config / source_paths / layer_matching / bias_required_config` computation; and `source_paths = flatten_index_paths(...)` + `_flatten_layer_matching(...)` are recomputed three times per component (l.222-226, 262-264, 293-294) plus inside `_build_*`. Unify: compute once before the branch.
5. `convert.py:87-100` and `convert.py:112-117`: two blocks that register missing buffers on `new_modules` parent (`rsplit(".")` + `register_buffer(torch.empty_like)`); factor `_ensure_buffer(new_modules, key, tensor)`.
6. `cascade.py:_collect_preprocessing_outputs` (l.586) vs `_collect_loader_inputs` (l.626): same loop, differs by running a module and detach/offload; the cached dict format (`__ts_args__`, `__ts_kwargs__`, `__ts_label__`) is also built in 5 places (l.612, 648, 692, 808, 889).
7. `cascade.py:_criteria_match_layer/_criterion_matches_layer` (l.457-492) re-implement `manager.iter_filtered` criteria semantics (`manager.py:1175-1227`) over a layer string; unify into a single `matches_criteria(name, path, block_id, criteria)` used by both. NOTE the two are subtly different (iter_filtered treats top-level list as OR of items where nested list = AND, and ignores `None` inside nested lists differently; cascade's version treats `None` at top-level as no match), so unification must pick one semantic.
8. `cascade._detach_and_optionally_offload` (l.975) vs `utils.move_to_device` (`utils.py:77`): same recursive tensor mapping; cascade's adds `.detach()` and handles dict only (not Mapping). Fold into `move_to_device(value, device, detach=False)`.
9. `cascade.py:17` uses `tqdm` directly while `interface.build_progress_bar` exists; `interface` text hard-codes "Calibration".
10. `manager.py`: `_iter_coupled_groups` (l.408-438) and `auto_groups` (l.319-370) duplicate the same loop (config lookup, `num_blocks`, per-block + cross-block resolution); `auto_groups` could consume `_iter_coupled_groups`. Also `block_config = self.config if config_attr == "" else getattr(self.config, config_attr)` repeated at l.346, 423, 1042-1046, and in `modeling.py:25-28`, `convert.py:179-183`, `configuration.py:50`: unify with `get_block_config(config, indexing)`.
11. `manager.py:1060-1062` + `_generate_schemes` sentinel loop: the two `CompressionScheme(...)` constructions (l.1086 vs l.1113) share 12 identical kwargs; use a small factory.
12. `manager.set` target resolution and `iter_filtered` aside, `manager._reset_compression_config` re-sets defaults already available through `scheme.set` default construction (`scheme.py:set` l.190-200 builds defaults from `COMPRESSION_REGISTRY` too).

With other folders:
- `compression/structured_pruning.py:669-674` parent/leaf traversal (see #1) and `:543-559` sentinel path resolution duplicates `manager._resolve_sentinel_path` + `CompressionScheme.preprocessing_path/final_layer_path` (guess: same intent; not diffed in detail).
- `export/common.py:606` uses `prepared_model.get_submodule(node.target)` (torch native `nn.Module.get_submodule`); `utils.get_submodule` behaves like torch's built-in `Module.get_submodule` but returns ValueError and uses `getattr(..., None)`. Suggest replacing by `model.get_submodule(path)` or keeping one thin wrapper (torch's raises AttributeError, so behaviour differs; check callers).
- `calibration/backbone.py` and `compression/smoothquant.py` import the three device helpers, no duplicate; but `blocks/config.py` + utils duplicate "config_attr == ''" resolution (guess, not exhaustively grepped).

## 7. Stale / unused code (grep evidence)

- `CompressionSchemesManager.update_config` (`manager.py:1229-1237`): body is only a docstring (returns None, does nothing). Only call site: `examples/03_hf_export_roundtrip.py:111` (`manager.update_config()`), so that example calls a no-op. No test.
- `CompressionSchemesManager._set_model` (l.1270): grep over repo = zero callers (outside its definition).
- `CompressionSchemesManager.delete_group` (l.294): no callers anywhere (tests/scripts/examples).
- `CompressionSchemesManager.print_filtered` (l.1165), `clear_calibration` (l.691), `cancel_vcon`, `freeze_uncompressed_vcon`: no callers in tests/scripts/examples (still public API; `init_vcon`/`set_vcon_beta` only used by examples/03).
- `self.smoothing` (`manager.py:316`): assigned, never read (`grep "\.smoothing\b"` only that line).
- Unused imports (verified by reading, no linter available): `deepcopy` in manager.py:13; `Union` in scheme.py:9; `Any` in cascade.py:13 (used? only `Dict, List, Optional, Sequence, Tuple, Union` are used in annotations - `Any` appears unused); `torch.nn` imported lazily in `prepare_for_save` while `torch` already imported.
- `manager.calibration_groups` initialised twice (l.70 and l.77); `config_attr` assigned twice in `_generate_schemes` (l.1035 and l.1039, with comment "it's done twice").
- `interface.FallbackProgressBar`: only used inside `build_progress_bar`; the whole file used by one caller.
- `CompressionScheme._is_to_compress` docstring says compatible = LinearCompressed only (stale, code allows four types). `modeling.py:60` comment "WIP logic for determining features size from pruning config" (stale WIP). `convert.py` docstring (l.168-176) documents non-existent arg `indexing` and omits `options`.
- `CompressionScheme.__init__` argument `model=None` plus `hasattr(self,'model')` checks (scheme.py l.151, l.254) always true (attribute always set) - dead guards.
- `utils/__init__.py` does not export `cascade` or `interface` (intentional, but `cascade.apply_cascade` is reachable only via manager) and exports `convert_for_export` which the top-level `__init__` re-exports under try/except (the `except` masks real import errors: `convert_for_export = None`).
- `docs/api/utils/*.md` reference `transformersurgeon.utils.manager/scheme`; will break on move.
- `test/compression_tests/` and `test/hf_export_tests/` contain only `__pycache__` (no tests).

## 8. Test coverage (grep in test/)

No test file targets utils directly (there is no `test/unit/test_utils*.py`, `test_manager*.py`, `test_scheme*.py`). Coverage is indirect via other features:
- `CompressionSchemesManager`: constructed via `*_c` model classes in nearly all unit/e2e tests (test/_helpers/model_factory.py). Direct manager-API tests: `set` + `apply(hard=...)` (test_attention_semantics, test_block_pruning, test_cascade_calibration, test_export_pipelines, test_gqa_attention_pruning, test_known_bugs, test_lpbq_quantization, test_model_families, test_onnx_tensorrt_export, test_paged_attention, test_qnn_bias_split, test_smoothquant); `auto_groups` (model_factory, test_attention_semantics, test_gqa_attention_pruning, test_known_bugs, test_model_families); `create_group` (test_known_bugs: cross-tower rejection l.117, duplicate-group rejection l.140); `_validate_coupling` (test_known_bugs l.64, 74); `_get_calibration_groups_from_indexing` (test_known_bugs); `set_calibration_data` (4 files); `set_calibration_mode` (test_cascade_calibration); `extract_summaries`, `iter_filtered`, `_find_scheme`, `get_compression_module` (test_extract_summaries); `smooth_activations` (test_smoothquant, test_onnx_tensorrt_export); `restore` (test_block_pruning, test_model_families); `prepare_for_save` (test_gqa_attention_pruning only); `load_state` matches in 3 tests but those are `model.load_state_dict` hits likely (guess, ugrep substring match; not manager.load_state).
- UNTESTED manager APIs (zero hits in test/): `delete_group`, `init_vcon`, `cancel_vcon`, `set_vcon_beta`, `freeze_uncompressed_vcon`, `set_calibration_loss` (used in scripts only), `run_calibration` (direct; indirect through apply), `clear_calibration`, `save_state`/`load_state` (used in benchmarks/nonhom_* only), `reapply_masks`, `remove_masks`, `print_filtered`, `update_config`, `_set_model`, `_validate_grouping`, `collect_calibration_target_stages`, `_resolve_sentinel_path`/`_resolve_final_norm_path` (indirect only).
- `apply_cascade`: test/unit/test_cascade_calibration.py: 3 tests: BERT and ModernBERT raise clear error (`no_cascade_calibration`), Qwen2 cascade still works. Not covered: skip-connections, position embeddings branch with 3D ids, `calibration_groups` overlap/non-consecutive errors, `_call_preprocessing_module` fallbacks, ViT/VL paths.
- `CompressionScheme`: no direct test (never constructed in tests). Indirect: `set`, `apply`, `restore`. Untested: VCON methods, `_module_copy`, `reapply_masks`, `remove_masks`, `add_to_group` conflict (covered via manager.create_group test_known_bugs:140).
- `SchemeGroup`: indirect via auto_groups/structured pruning shared-mask tests only.
- `convert_for_export`: heavily used (test_model_families::test_convert_for_export l.143, test_qnn_*, test_paged_attention, test_onnx_tensorrt_export, test_export_fixes, test_rope_variants, test_attention_semantics, test_block_pruning, test_gqa_rope_pruning, e2e/test_gqa_attention_pruning, test_export_pipelines). Private helpers tested only transitively: `_load_state_dict_dequantized` (quantized paths in qnn/lpbq/export_fixes tests), `_extract_aqt_scale`, `_flatten_layer_matching` (error branches untested), `_build_bias_required_config` (test_qnn_bias_split likely), encoder branch (via BERT/ViT e2e in test_model_families). Untested: unsupported `structure` NotImplementedError, missing `structure` warning path, `extra_layers_matching` default.
- `init_compressed_config` / `replace_layers_upon_init`: exercised at every model construction; no unit tests on validator failure, `preprocessing_conv` Conv3d path (Qwen-VL e2e only), or skip-with-warning branch in `_replace_singleton_layer`.
- `utils.py` helpers (`get_submodule`, `flatten_index_paths`, `infer_model_device`, `move_to_device`): no direct tests; `flatten_index_paths` error branches untested; `get_submodule` None-vs-missing semantics untested.
- `interface.py`: no tests.

## Suggested target layout (guess)

```
utils/            utils.py only: get_submodule, set_submodule/replace_submodule, get_parent, flatten_index_paths (+expand_grouped_paths),
                  infer_model_device, move_to_device(detach=), get_block_config(config, indexing)  [leaf, imports only torch]
calibration/      + progress.py (from interface.py), + cascade.py, + scheduling pieces of manager (stages, targets, groups, state save/load)
compression/      + scheme.py, grouping.py, config.py (init_compressed_config), replace.py (replace_layers_upon_init), vcon.py
pipeline/ (or core/)  manager.py (CompressionSchemesManager: set/apply/restore/groups/filter), thin facade over compression + calibration
export/           + convert.py (convert_for_export + dequantized state-dict transfer)
models/           define_*_c.py keep importing from the new locations
```
Keep a compat shim `transformersurgeon/utils/__init__.py` re-exporting moved names for one release only if external code needs it (README.md:154 and docs/api/*.md use `transformersurgeon.utils` paths).
