# Recap: transformersurgeon/compression/

(All paths relative to transformersurgeon/ unless absolute. Line numbers verified by reading every file.)

## 1. Purpose, registration, invocation

compression/ holds one `Compressor` subclass per compression type (LRD, structured / unstructured / block pruning,
quantization) plus per-method math functions (svd, gptq, hqq, magnitude masks...) and a standalone SmoothQuant
transform. Compressors are config-driven: a dict `{type: {key: value}}` per layer is turned into compressor objects by
`utils/scheme.py::CompressionScheme`; the manager (`utils/manager.py`) drives apply/restore.

Registration (3 hand-edited tables, all in `compression/registry.py`):
- `COMPRESSOR_DICT` (registry.py:44-50): `"name" -> CompressorClass`.
- `COMPRESSION_REGISTRY` (registry.py:68-137): `"name" -> {key: dict(default=..., validator=fn)}`; this is the single
  source for config keys, defaults, validators. Also consumed by `utils/configuration.py:61` (default config building)
  and `utils/manager.py:206-209`.
- `APPLICATION_ORDER` (registry.py:57): `["block_pruning","lrd","structured_pruning","unstructured_pruning","quantization"]`;
  types not listed are appended after (scheme.py:237-246 `_sort_compressors`).
- `GROUP_OPTIONS` (registry.py:63): only `{"structured_pruning": ["share_mask"]}`; keys settable only via a group.
- `compression/__init__.py` re-exports the 6 compressor classes (StructuredPruner,
  CoupledPruner, BlockPruner, UnstructuredPruner, LRDer, Quantizer) + the 4 registry tables + validators.

Abstract base `abstract.py::Compressor(ABC)`: abstract = `set_calibration_store`, `needs_calibration`, `apply`, `restore`,
`_to_compress`, `__repr__`; non-abstract no-op hooks = `needs_grouping`, `check_coupling`, `reapply_mask`, `remove_mask`.
Informal contract not in the ABC: constructor takes the per-layer config dict (stored as `self.config`, mutated in place
by validators/restore); each config key must exist as an attribute of the same name (scheme.py:206-210 does
`setattr(compressor, key, value)` and raises if `hasattr` fails); `self.scheme` back-reference is planted by
scheme.py:213-221 / 202 and is used only by structured and block pruning.

Call path:
1. `CompressionScheme.__init__` (scheme.py:128-137): `COMPRESSOR_DICT[name](config)`; `_link_compressors` sets `.scheme`.
2. `CompressionScheme.set()` (scheme.py:180-211): validator from `COMPRESSION_REGISTRY`, builds default dict from registry
   defaults, instantiates compressor lazily, `setattr`.
3. `Manager.apply` (manager.py ~726-800): for each compressor `set_calibration_store(scheme.calibration_data)`,
   `needs_calibration()` -> tuple of summary names -> `calibration.run_compression_calibration` fills the store;
   `check_coupling`/`needs_grouping` are called around manager.py:445-471.
4. `CompressionScheme.apply` (scheme.py:535-539): per compressor in APPLICATION_ORDER:
   `set_calibration_store(...)`, `apply(module, hard, soft_applied)`.
5. `restore` (scheme.py:567-570), `reapply_mask`/`remove_mask` (scheme.py:599-612, manager reapply_masks/remove_masks).
6. Outside the registry flow: `Manager.smooth_activations` imports `compression.smoothquant.apply_smoothquant` (manager.py:309);
   `Manager.prepare_for_save` does `isinstance(compressor, Quantizer)` + `strip_runtime_state` (manager.py:948-975).

## 2. File tree (line counts)

```
compression/                       (3623 total)
  __init__.py                7
  abstract.py              211   Compressor ABC
  registry.py              147   COMPRESSOR_DICT, COMPRESSION_REGISTRY, APPLICATION_ORDER, GROUP_OPTIONS
  lrd.py                   155   LRDer + 3 validators
  lrd_methods/
    __init__.py             10   METHOD_FUNCTIONS
    svd.py                  16   low_rank_svd
    svd_llm_v2.py           45   low_rank_svd_llm_v2
    aa_svd.py              102   low_rank_aa_svd
  quantization.py          593   Quantizer + hooks + torchao hard path + 11 validators
  quantization_methods/
    __init__.py             15   METHOD_FUNCTIONS, ACT_METHOD_FUNCTIONS
    vanilla.py              91   RTN, LPBQ scales, dequant fns
    gptq.py                120   quantize_gptq
    hqq.py                  57   quantize_hqq
    act_maxmin.py           31   compute_activation_scale_zp_maxmin
  smoothquant.py           166   standalone SmoothQuant transform (not a Compressor)
  structured_pruning.py    798   StructuredPruner + 7 validators
  structured_pruning_methods/
    __init__.py             33   SCORE_FUNCTIONS + re-exports
    magnitude.py             9   score_magnitude
    gradient.py             11   score_gradient
    random.py                9   score_random
    mask_generation.py      91   build_structured_mask, reduce_scores (+re-export from blocks)
    pattern_ops.py         174   reduce_pattern_scores, reduce_member_to_patterns, tile_pattern_mask, _reduce
  coupled_pruning.py       226   CoupledPruner (not registered; helper)
  block_pruning.py         236   BlockPruner + 3 validators
  unstructured_pruning.py  156   UnstructuredPruner + 3 validators
  unstructured_pruning_methods/
    __init__.py             11   METHOD_FUNCTIONS
    magnitude.py            17   mask_magnitude
    gradient.py             20   mask_gradient
    random.py               17   mask_random
    mask_generation.py      49   build_unstructured_mask, _reshape_for_granularity
```

## 3. Per-file contents and config keys

Common: every compressor stores `self.config` (the live dict), `self.calibration_store=None`; `needs_calibration()` returns
`()` when `_to_compress()` is False.

### registry.py
Config schema (default, validator). Keys = exactly what `manager.set(type, key, value)` accepts.
- structured_pruning: ratio=0.0; method="magnitude" (magnitude|gradient|random); granularity="layer" ("layer"|None|positive int);
  repeated_pattern=False (False|True|"max"|"auto"|int>0); coupled_repeated_pattern=False (False|int>0); reduce_op=None
  ("add"|"multiply"|None|""); share_mask=False (bool; group-only).
- block_pruning: ratio=0.0 ([0,1)); method="magnitude" (magnitude|random); blocks=None (list[int] of global block idx).
- unstructured_pruning: ratio=0.0 ([0,1]); method="magnitude" (magnitude|gradient|random); granularity="layer" ("layer"|"neuron"|int>0).
- lrd: rank="full" ("full"|int>0); method="svd" (svd|svd-llm-v2|aa-svd); eps=1e-6 (>0).
- quantization: method="vanilla" (vanilla|gptq|hqq); precision="full" ("full"|"binary"|int 2..16; hard path needs 2..8);
  sparsity=0.0 (reuses unstructured ratio validator); sparse_method="magnitude" (magnitude|random); eps=1e-6;
  granularity="per_tensor" (per_tensor|per_channel|per_group); group_size=128; group_scale_bits=None (None|int 1..8; LPBQ,
  needs per_group+vanilla); precision_activation="full" (same validator as precision); method_activation="maxmin"
  (maxmin|dynamic_per_token); scheme_activation="asymmetric" (symmetric|asymmetric); quantize_output_activation=True (bool).
Note `method_activation`/`scheme_activation`/`group_*`/`quantize_output_activation` are read with `.get(..., default)` in
Quantizer.__init__ only for group_size, group_scale_bits, quantize_output_activation; others use `config[...]` (KeyError if absent).

### abstract.py
`class Compressor(ABC)`: see section 1. Docstrings embed LRD/pruning/quant specifics (e.g. restore() says "weight @ linear_V.weight",
"cancel_lrd") - LRD/LinearCompressed knowledge in the base class docs.

### lrd.py
- `LRDer(config)`: rank, method, eps. `apply(module, hard, soft_applied)`: validates, rejects weight.dim()>2, dispatches
  per method by if/elif (svd / svd-llm-v2 needs `calibration_store["covariance"]` / aa-svd needs `cross_covariance`,
  `shifted_covariance`), then `module.init_lrd(rank)`, writes `module.weight[:, :rank]` (U*S factor) and
  `module.linear_V.weight` (V). `hard` is a no-op. `restore`: `weight @ linear_V.weight`, `module.cancel_lrd()`.
  `_to_compress`: rank != "full". Module API required: `init_lrd`, `cancel_lrd`, `linear_V`, `rank`, `.weight` 2-D.
- module globals: `VALID_METHODS` (3), `CALIBRATED_METHOD_DICT` (method -> summary names).
- `validate_lrd_rank/method/eps`.
### lrd_methods/
- `low_rank_svd(weight, rank) -> (US_r, V_r)`; `low_rank_svd_llm_v2(weight, rank, covariance, eps)` (eig-based whitening);
  `low_rank_aa_svd(weight, rank, cross_covariance, shifted_covariance, eps)` (eig whitening, warnings on ill-conditioning).
  `METHOD_FUNCTIONS` dict maps method -> fn; each has a different signature, so LRDer special-cases each (lrd.py:57-85).

### quantization.py
- `Quantizer(config)`: fields listed above. Methods: `set_calibration_store`; `needs_calibration` (gptq -> covariance;
  plus `activation_range` / `output_activation_range` when precision_activation != full and method_activation != dynamic_per_token);
  `apply` (L54-218): soft = per-param fake-quant (`weight`, `linear_V.weight`) via `METHOD_FUNCTIONS[method]` then `dequantize_fn`,
  optional sparsity mask mixed in, sets `module._soft_quant_precision`; hard = `_apply_hard_quantization` (torchao
  `IntxUnpackedToInt8Tensor` container; sets `_torchao_*` attrs and `_torchao_scale` buffer); activation fake-quant via forward
  pre-hook / post-hook registered on the module + buffers `_act_quant_scale/_zero_point`, `_act_out_quant_*`, attrs `_act_quant_precision/_method/_scheme`;
  LRD-aware (covariance for U factor = V C V^T, L82-... and L355-375); `_calibration_kwargs`; `restore` (resets config to defaults, then calls `strip_runtime_state`; weights are restored by the scheme, not here); `strip_runtime_state(module)` (removes hooks + 19 attrs; used by manager.prepare_for_save);
  `_to_compress` (precision != "full"); `__repr__`.
- `_make_activation_fake_quant_hook(scale, zero_point, precision, scheme)` L290, `_make_output_activation_fake_quant_hook(...)` L317
  (scale/zero_point args unused: hook reads module buffers).
- `_extract_aqt_scale(w)` L341; `_apply_hard_quantization(module, method, precision, granularity, group_size, eps, calibration_kwargs, dynamic_activation, group_scale_bits)` L355;
  `_apply_torchao_hard_quantization(module, precision, granularity, group_size)` L425 (see section 7: dead).
- module globals: `VALID_METHODS`, `VALID_ACT_METHODS`, `VALID_SPARSE_METHODS`, `VALID_ACTIVATION_SCHEMES`, `VALID_GRANULARITIES`, `CALIBRATED_METHOD_DICT`.
- validators: method, activation_method, precision, sparse_method, eps, granularity, group_size, group_scale_bits, quantize_output_activation, activation_scheme.
- Imports `SPARSITY_FUNCTIONS` from unstructured_pruning_methods and `validate_unstructured_pruning_ratio` (cross-compressor coupling).
### quantization_methods/
All return `(qdata int8, scale, dequantize_fn)`.
- vanilla.py: `_dequantize_symmetric(q, scale)`, `_dequantize_grouped(q, scale)`, `_quantize_maxabs(weight, precision, eps, per_channel, group_size, group_scale_bits)`,
  `lpbq_group_scales(scale, bits)`, `_quantize_binarize(weight, per_channel)`, `quantize_vanilla(weight, precision, eps, granularity, group_size, group_scale_bits)`.
  Symmetric range [-qmax, qmax], qmax=2^(b-1)-1; scale = max|w|/qmax; divides by `(scale+eps)`.
- gptq.py: `quantize_gptq(weight, precision, eps, granularity, *, covariance, block_size=128, percdamp=0.01)` (per_channel/per_tensor only; falls back to RTN without covariance); `_cholesky_inv_upper(H, damp)`.
- hqq.py: `quantize_hqq(weight, precision, eps, granularity, group_size, iters=20, early_stop_tol=1e-5)`: scale-only, signed full range [-2^(b-1), 2^(b-1)-1] (differs from vanilla/gptq symmetric [-qmax,qmax]).
- act_maxmin.py: `compute_activation_scale_zp_maxmin(act_min, act_max, precision, scheme) -> (scale, zero_point int32)`.
- `__init__`: `METHOD_FUNCTIONS = {vanilla,gptq,hqq}`; `ACT_METHOD_FUNCTIONS = {"maxmin"}` ("dynamic_per_token" is accepted by validator but handled by branching in Quantizer.apply, not in the dict).
  Method fns receive different kwargs; Quantizer builds them via CALIBRATED_METHOD_DICT keys + group_size/group_scale_bits (quantization.py:126-148).

### smoothquant.py
Not a Compressor; not in the registry; separate entrypoint.
`_input_weight(linear)`, `_check_float(t, what)`, `resolve_smoothing_groups(model, indexing, consumers_filter) -> list[dict]` (walks
indexing[*]['smoothing'], `config_attr`, `num_blocks_attr`, `path_template`), `collect_input_absmax(model, modules, calibration_data, max_batches)` (forward pre-hooks, runs
the model on calibration batches), `apply_smoothquant(model, indexing, calibration_data, alpha=0.5, consumers_filter=None, max_batches=None, min_scale=1e-5) -> summary`.
Keys: alpha, max_batches, min_scale (function args, NOT in COMPRESSION_REGISTRY).

### structured_pruning.py (L73-725)
`StructuredPruner(config)`: ratio, method (fallback key `criterion`), granularity, repeated_pattern, coupled_repeated_pattern, share_mask, reduce_op.
Public/internal methods: `needs_calibration` (gradient->weight_grad), `needs_grouping` (share_mask), `check_coupling(hard, siblings)`, `get_scores(module)`,
`_repeat_n`, `_pattern_keep_mask`, `_own_mask`, `_group`, `_resolve_shared_mask`, `_group_members`, `_compute_group_shared`, `_compute_auto_shared`,
`_build_auto_mask` (position_linked folding), `_resolve_mask`, `apply` (registers `weight_mask`; for position_linked also `rope_prune_mask`; soft zeroes rows+bias; hard: `_hard_remove`, `_cascade_coupled`, `_prune_extra_params`),
`_hard_remove`, static `_hop_block_id`, static `_resolve_named_module`, `_resolve_target_chains`, `_coupled_target_chains`, `_coupled_cascade_mask`,
`_cascade_coupled`, `_prune_extra_params`, `restore` (resets config, reverts coupled input pruning, clears group shared-mask cache; does NOT restore row shapes of module itself), `reapply_mask`, `remove_mask`, `_to_compress`, `__repr__`.
Helpers: `_is_embedding(module)`, `_score_view(weight)`. Constants: `VALID_METHODS`, `VALID_REDUCE_OPS`, `CALIBRATED_METHOD_DICT`, `_SHARED_MASK_KEY`.
Validators (7): ratio, method, granularity, repeated_pattern, coupled_repeated_pattern, share_mask, reduce_op.
### structured_pruning_methods/
- `score_magnitude(weight, norm=2)`, `score_gradient(weight, weight_grad, norm=2)`, `score_random(weight, norm=2)` (row-wise, dim=1); `SCORE_FUNCTIONS`.
- mask_generation: `build_structured_mask(scores, ratio, granularity)`, `reduce_scores(list, op)`; re-exports `effective_num_pruned`, `effective_out_features` from `blocks/pruning_dims.py`.
- pattern_ops: `reduce_pattern_scores`, `reduce_member_to_patterns`, `tile_pattern_mask`, `_reduce`.

### coupled_pruning.py
`CoupledPruner` (stateless; instantiated ad hoc): `_input_target(module)` (LRD -> linear_V), `apply(module, mask, hard=True)` (drops input columns; backup attr `_coupled_input_backup`),
`revert(module)`, static `_is_elementwise(module)` (weight.dim()==1 => norm), `_apply_elementwise`, `_revert_elementwise` (backup `_coupled_elementwise_backup`; adjusts `normalized_shape`), `apply_chain(chain, mask, hard)`, `revert_chain(chain)`.

### block_pruning.py
`BlockPruner(config)`: ratio, method, blocks; `.scheme`. `_block_scores_magnitude(q_weight, dep_weights, block, n_blocks)`. Methods: `needs_calibration` (always `()`), `_spec` (reads `scheme.pruning_indexing["block_pruning"][scheme.name]`),
`_resolve(name)` (lazy `utils.utils.get_submodule`), `_geometry`, `removed_blocks(module)`, `apply` (soft: zero dependent input columns, backup in `_block_pruning_soft_backup`, buffer `block_prune_mask`; hard: slice rows, CoupledPruner on deps, **writes `owner.<blocks_per_unit_attr>`**), `_undo_soft`, `restore` (soft only; hard is irreversible), `_to_compress`.
Validators: ratio, method, blocks.

### unstructured_pruning.py
`UnstructuredPruner(config)`: ratio, method (fallback key `criterion`), granularity. `apply`: mask via `METHOD_FUNCTIONS`, `register_buffer("weight_mask")`, in-place mul; `hard` only logs a warning (no sparse backend). `restore` resets config only (mask kept). `reapply_mask`, `remove_mask`. Validators: ratio, method, granularity.
### unstructured_pruning_methods/
`mask_magnitude/mask_gradient/mask_random(weight, [weight_grad,] pruning_ratio, granularity)`; `build_unstructured_mask(scores, ratio, granularity)`; `_reshape_for_granularity`.

## 4. Dependencies

Imports from elsewhere in transformersurgeon (compression -> outside):
- structured_pruning.py:15 `..blocks.EmbeddingCompressed`; :543,:669 lazy `..utils.utils.get_submodule`.
- structured_pruning_methods/mask_generation.py:8 `...blocks.pruning_dims.{effective_num_pruned, effective_out_features}` (re-exported).
- block_pruning.py:`_resolve` lazy `..utils.utils.get_submodule`.
- smoothquant.py:`..calibration.raw_data.normalize_calibration_batch`, `..utils.utils.{get_submodule, infer_model_device, move_to_device}` (top-level imports).
- lazy torchao imports in quantization.py:`_apply_hard_quantization` and `_apply_torchao_hard_quantization`.
- Implicit (duck-typed, no import): `CompressionScheme` attrs (`model, path, path_template, block_id, num_blocks, path_list, name, groups, pruning_indexing, position_linked, final_layer_path, final_norm_path, preprocessing_conv_extra_params_paths, get_compression_module()`), `SchemeGroup.properties/.schemes`, `LinearCompressed` API (`init_lrd, cancel_lrd, linear_V, rank, out_features, in_features`).

Who imports compression/ (grep across transformersurgeon, test, scripts, examples, benchmarks):
- `utils/scheme.py:14-18`: `COMPRESSOR_DICT, COMPRESSION_REGISTRY, APPLICATION_ORDER`.
- `utils/manager.py:17`: `GROUP_OPTIONS, COMPRESSION_REGISTRY`; manager.py:309 lazy `compression.smoothquant.apply_smoothquant`; manager.py:948 lazy `compression.quantization.Quantizer`.
- `utils/configuration.py:4`: `COMPRESSION_REGISTRY`.
- test/unit/test_repeated_pattern.py:5 `compression.structured_pruning_methods` (build_structured_mask, reduce_pattern_scores, reduce_member_to_patterns, tile_pattern_mask).
- test/unit/test_lpbq_quantization.py:13 `compression.quantization_methods.vanilla` (lpbq_group_scales, quantize_vanilla).
- Nothing in export/, calibration/, hf/, models/, blocks/, examples/, scripts/, benchmarks/ imports compression/ (comments only: models/*/indexing_*.py mention `CoupledPruner.apply_chain`, `compression/smoothquant.py`; blocks/mha.py:221 mentions block_pruning).
- Internal: structured_pruning -> coupled_pruning, structured_pruning_methods; block_pruning -> coupled_pruning; quantization -> unstructured_pruning(+_methods), quantization_methods; lrd -> lrd_methods; unstructured_* methods -> own mask_generation; registry -> every compressor module for validators; gptq/hqq -> vanilla (private `_dequantize_*`).
- Circularity: utils.scheme imports compression (top-level) and compression imports utils lazily (hence the "avoid import cycle" comments).

## 5. Violations vs. target rule

Model-specific names: no Qwen/Llama/ViT identifiers in executable code (grep confirmed). Only comments: structured_pruning.py:25 ("ViT/Qwen-VL patch-embed", "BERT-style"), :536 ("llama_c's mlp.down_proj"), :655-657 (ViT cls_token/position_embeddings).
Model-structure knowledge baked into compression code (leaks through indexing keys, i.e. model-family concepts):
- structured_pruning.py:19-34, :526-559, :596-623: sentinel names `'preprocessing'`, `'final_layer'`, `'final_norm'`; `_hop_block_id` encodes residual-stream/block wrap-around topology (L490-520); `_resolve_named_module` resolves paths via `scheme.path_template`/`final_layer_path`. This is graph/model topology resolution, belongs to the indexing/manager layer (it already exists partly in manager `_resolve_sentinel_path`, per comment L22).
- structured_pruning.py:663 hard-coded scheme name `"preprocessing_conv"` and `scheme.preprocessing_conv_extra_params_paths` (ViT cls_token/pos-embed) in `_prune_extra_params` L648-678.
- structured_pruning.py:418-426 and :440-453 `position_linked` / `rope_prune_mask` (RoPE q/k) buffer + warning; :331-355 `_build_auto_mask` RoPE real/imag folding; block_pruning.py:152 reads `rope_prune_mask`. RoPE is attention-architecture knowledge; consumed by blocks/mha.py:238,294 and blocks/rope.py:316 (compressor -> blocks contract via buffer name).
- block_pruning.py: whole module is attention-head specific (docstring L1-33, `owner`, `block_size_attr`, `blocks_per_unit_attr`; hard apply mutates `owner.num_key_value_groups` via setattr, L186). Spec is read from indexing (`scheme.pruning_indexing["block_pruning"]`), so names are generic, but semantic = heads/GQA; couples to `blocks/mha.py::finalize_head_pruning`.
- structured_pruning.py:15,46-56,146,191,429,464-469: `EmbeddingCompressed` import + `_is_embedding` branches (prunable axis dim 1) - block-type knowledge.
- lrd.py / quantization.py rely on `LinearCompressed` attrs (`init_lrd`, `cancel_lrd`, `linear_V`, `rank`) - acceptable if module interface is declared, but undocumented as an interface.
- structured_pruning.py:482-485 `hasattr(module, "out_channels")` (conv) branching.
- coupled_pruning.py:_is_elementwise uses 1-D weight as "is norm" heuristic (comment says model-agnostic; fine).

Backend/export-specific logic:
- quantization.py:11-14 comment/feature `dynamic_per_token` = "ExecuTorch/XNNPACK 8daXw"; quantization.py:355-423 `_apply_hard_quantization` builds `torchao IntxUnpackedToInt8Tensor` and sets `_torchao_*` attrs (read by export/common.py:368-420, utils/convert.py:14-23). Export-contract attributes live in a compressor.
- registry.py:133-136 `quantize_output_activation` comment names TensorRT W8A8; registry.py:127-131 `group_scale_bits` comment names "Qualcomm HTP LPBQ"; vanilla.py:45-55 `lpbq_group_scales` = Qualcomm LPBQ. `smoothquant.py` header names TensorRT W8A8. No code imports or branches on qnn/xnnpack/tensorrt/onnx names, but the options exist for backends.
- quantization.py:242-276 `strip_runtime_state` lists backend/export attr names (19 attrs). The same list is duplicated in utils/convert.py:14-23.
- Hard-quant torchao container format chosen for exporters (docstring L360-367 "every exporter see the same numbers").

Generic layer-manipulation helpers that belong in utils/ (or blocks/):
- coupled_pruning.py entire `CoupledPruner` (resize in_features / norm weights; `_input_target`, `_apply_elementwise`) - generic module surgery. (Used by structured + block pruning, so can stay as a shared compression-internal helper; but row/column slicing + backup/revert could live in utils.)
- structured_pruning.py:460-488 `_hard_remove` (resize Linear/Embedding/Conv rows), :677 `_prune_extra_params` param slicing, :70 `_score_view`, :46 `_is_embedding`.
- block_pruning.py:176-186 row slicing duplicates `_hard_remove`.
- smoothquant.py:`_input_weight`, producer/consumer scaling (L130-160).
- unstructured/structured `reapply_mask` duplicate broadcast logic (see 6).

Calibration-data collection that belongs in calibration/:
- smoothquant.py:68-92 `collect_input_absmax` (runs the model over calibration batches with pre-hooks; uses `normalize_calibration_batch`, `infer_model_device`, `move_to_device`). It is a calibration summary ("per-input-channel absmax") duplicated with the calibration/summaries machinery (not verified which existing summary overlaps; `activation_range` has per-tensor min/max, not per-channel - GUESS: a new `channel_absmax` summary would be needed).
- smoothquant.py:`resolve_smoothing_groups` walks indexing/model (graph resolution) - belongs with manager/indexing helpers.
- quantization.py:174-217 activation-range consumption is OK (reads store), no collection there.
- Pruning "gradient" method: pure consumer of `weight_grad` summary; OK.

Other leaks:
- Compressors reach into `CompressionScheme`/`SchemeGroup` internals (structured_pruning.py:198-206, 249-258, 563-621, 694-695; block_pruning.py:75-94). `.scheme` is not part of the ABC.
- registry.py hand-lists validators imported from each compressor module (repeated 40 names).
- Quantizer imports UnstructuredPruner validator and `SPARSITY_FUNCTIONS` (sparse-quant mixes two compressors).
- manager.py:948 `isinstance(compressor, Quantizer)` (outside folder, but implies lacking generic `prepare_for_save` hook on ABC).
- scheme.py/manager.py special cases: manager.smooth_activations, `auto_groups`, `GROUP_OPTIONS` handling (manager.py:158-209) hard-wire structured pruning concepts.

## 6. Duplicated code

Within compression/:
1. Validators repeated: `validate_structured_pruning_ratio` (structured_pruning.py:730) == `validate_unstructured_pruning_ratio` (unstructured_pruning.py:~134); `*_method` pairs (structured:734, unstructured, block:223); `validate_quantization_eps` (quantization.py:550) == `validate_lrd_eps` (lrd.py:143). Unify: shared `validators.py` with `positive_float`, `ratio_in_unit_interval`, `one_of(list)` factories.
2. `VALID_METHODS` + `CALIBRATED_METHOD_DICT` + `needs_calibration()` pattern copied in lrd.py:8-15/30-33, quantization.py:11-20, structured_pruning.py:36-40, unstructured_pruning.py:10-13. Unify: base class attribute `CALIBRATION_REQUIREMENTS = {method: (names)}` and a default `needs_calibration`.
3. `set_calibration_store`, `calibration_store=None`, `_to_compress` guard identical in all 5; `self.config["x"] = x` self-assignments after validate (lrd.py:43-45, unstructured_pruning.py:44-46, structured_pruning.py:397-399, quantization.py:57-60,86-97) are no-ops. Unify: base-class `__init__(config)` + `validate()`; drive attributes from `COMPRESSION_REGISTRY` keys.
4. Mask application: structured_pruning.py:427-438 == :702-710 (`apply` vs `reapply_mask`); unstructured_pruning.py:76-77 == :99-101. Unify: helper `apply_row_mask(module, mask)`.
5. Quantizer.apply soft path builds calibration kwargs (quantization.py:126-132) duplicating `_calibration_kwargs` (L220-226). Also sparsity branches L138-146: the `magnitude` and `random` arms are identical code - collapse to `SPARSITY_FUNCTIONS[sparse_method](param, pruning_ratio=sparsity)`.
6. LRD covariance projection `V @ C @ V.T` duplicated: quantization.py:117-123 and :383-386.
7. Lower/upper clamp-of-scale and `qmax = 2**(b-1)-1` appear in vanilla.py:23,51, gptq.py:44, hqq.py:35 (hqq uses a different signed range), act_maxmin.py:16-19, and fake-quant hooks quantization.py:296-299/323-326 (the same `symmetric: qmax=2^(b-1)-1, qmin=-qmax; asym: qmax=2^b-1, qmin=0` rule appears 3x in compression/ and again in export/common.py:733-737). Unify: `quantization_methods/qrange.py: qrange(precision, scheme)`.
8. `_make_activation_fake_quant_hook` vs `_make_output_activation_fake_quant_hook` differ only in buffer names/pre vs post; activation input/output blocks in apply (L174-217) are ~copy-paste. Unify with a parametrized helper.
9. Scores->mask chain: `unstructured_pruning_methods/{magnitude,gradient,random}.py` each define `_score` + call `build_unstructured_mask`; structured methods similar. Fine but could be a single `SCORE_FUNCTIONS` shared by both pruners (magnitude/random scoring is the same abs/rand_like, only reduction dim differs).
10. Row/column slice of `nn.Parameter` with backup: coupled_pruning.py:84-92 vs :146-156 vs structured_pruning.py:466-480 vs block_pruning.py:176-181 vs smoothquant.py:140-158. Unify: `utils/param_ops.py: replace_param(module, attr, tensor)`.
11. Pattern-reduce fns pattern_ops.py: `reduce_pattern_scores` and `reduce_member_to_patterns` share validation; `_reduce` only used here.
12. low_rank_svd_llm_v2 vs low_rank_aa_svd: same eigh-based whitening + sqrt(S) split (svd_llm_v2.py:20-43 vs aa_svd.py:46-100). Unify into `_whitened_lowrank(weight, rank, w_cov_sqrt, w_cov_inv_sqrt)`.

Across compression/ and other folders (quantization math):
- `_extract_aqt_scale`: quantization.py:341-352 == utils/convert.py:27-37 (verbatim). quantization.py copy only used by dead `_apply_torchao_hard_quantization`. Unify into one util.
- Backend re-quantization of weights from stashed scale with the same formula `clamp(round(w / scale), qmin, qmax)`:
  - export/executorch_exporters/xnnpack/xnnpack_export.py:78-84 (also rebuilds `IntxUnpackedToInt8Tensor` - identical construct to quantization.py:410-419). It recomputes `qdata` that `_apply_hard_quantization` already computed (qdata is lost after container build, only scale stashed in `_torchao_scale`). Unify: store `qdata` in the layer-info (or expose `quantize_with_scale(w, scale, precision, block)` in compression/quantization_methods) and have xnnpack call it / reuse `_apply_hard_quantization` container builder.
  - export/onnx/quantization.py:77, 92, 104, 109, 135-137: `np.rint(w/scale).clip(qmin,qmax)` for per-channel, per-group, and a *max-abs scale recomputation* `np.abs(blocks).max(axis=2)/qmax` (L135) that duplicates `_quantize_maxabs` (vanilla.py:23-31) for int4 block mode, plus its own `_SCALE_FLOOR`/`_QRANGE` tables.
  - export/common.py:68-77 `_PRECISION_TO_QRANGE` (int4: -8..7, int8: -127..127) vs vanilla `qmax=2^(b-1)-1` symmetric (-qmax..qmax) vs hqq (-2^(b-1)..2^(b-1)-1) vs xnnpack `qmin=-qmax-1` (xnnpack_export.py:80-81): four different qmin conventions for the same weights - bug risk (export/common.py:973 comment already notes -127 vs -128).
  - export/common.py:733-737, 764-778: activation qrange / zero-point derivation duplicates act_maxmin.py:16-27 and hook rule quantization.py:296-299 (inverse direction: scale/zp -> min/max).
  - export/common.py:897-960 `dequant_from_exported_state_dict`: dequant `(q - zp) * scale` duplicates `_dequantize_symmetric/_grouped` (vanilla.py:5-13).
  - `lpbq_group_scales` (vanilla.py:45) is the only LPBQ implementation; qnn_export.py:509 only detects `group_size` - no duplicate math found there (checked grep ceil/lpbq).
  - strip-attr list: quantization.py:253-272 vs utils/convert.py:14-23 (same 19 attr names, two copies). Unify: a `QUANT_RUNTIME_ATTRS` constant in compression/ (export/convert import it).
- `effective_num_pruned` single source in blocks/pruning_dims.py (OK, re-exported).
- `get_submodule` / hop resolution: structured_pruning.py:523-559 `_resolve_named_module` mirrors manager `_resolve_sentinel_path` (per comment, not verified line-by-line).

## 7. Stale / unused code (grep evidence)

- `quantization.py:425-517 _apply_torchao_hard_quantization` (+ its helper use of `_extract_aqt_scale` L341): no callers anywhere (grep over transformersurgeon, test, scripts, examples, benchmarks: only the comment in blocks/embedding_compressed.py:14). Dead; superseded by `_apply_hard_quantization`. Also removes the `Int8WeightOnlyConfig`/`quantize_` torchao dependency.
- `quantization.py:341 _extract_aqt_scale`: only used by the dead function above (live copy in utils/convert.py:27).
- `quantization.py` args `scale, zero_point` of `_make_activation_fake_quant_hook` / `_make_output_activation_fake_quant_hook` are accepted but never used inside hooks (L290-337).
- `quantization.py:44` `Quantizer.restore` resets config but weights restored by the scheme (`_weight_original`), not here - not stale, but misleading comments.
- `ACT_METHOD_FUNCTIONS` only contains "maxmin"; "dynamic_per_token" in `VALID_ACT_METHODS` is special-cased in apply (L158-164).
- `validate_quantization_group_size` is called only from registry (not in apply; group_size not re-validated); `Quantizer.__init__` doesn't validate eps.
- Legacy key fallback `config.get("criterion")` in structured_pruning.py:77 and unstructured_pruning.py:20 (grep "criterion" outside: only utils/cascade.py unrelated `_criterion_matches_layer`). Likely dead legacy; flagged as probable.
- `structured_pruning_methods/mask_generation.py` re-exports `effective_out_features`; used only in blocks/pruning_dims.py itself (grep) - re-export unused by compression.
- `score_random(weight, norm)`: `norm` param unused. `_reshape_for_granularity(scores, pruning_ratio: int, ...)` wrong annotation.
- `CALIBRATED_METHOD_DICT` in `quantization.py` only has gptq; hqq is calibration-free (OK).
- `UnstructuredPruner` hard path is a documented stub (TODO sparse tensor; unstructured_pruning.py:78-88).
- `registry.py` `GROUP_OPTIONS` single entry; comment mentions reduce_op excluded.
- `compression/__init__.py` omits `smoothquant` exports (intentionally lazy).
- Docstring drift: abstract.py:`_to_compress` says Quantization `bits == 32`; real check is precision != "full".
- `LRDer.apply` re-validates then re-assigns identical config values (no-op); `apply(hard)` branch `pass`.

## 8. Test coverage (grep of test/, scripts/, examples/, benchmarks/)

Test dir: test/unit (19 files), test/e2e (3), test/compression_tests (empty, only __pycache__), no tests directly import any Compressor class.
Direct unit tests of compression internals:
- `structured_pruning_methods.{build_structured_mask, reduce_pattern_scores, reduce_member_to_patterns, tile_pattern_mask}`: test/unit/test_repeated_pattern.py.
- `quantization_methods.vanilla.{lpbq_group_scales, quantize_vanilla}`: test/unit/test_lpbq_quantization.py (per_group+LPBQ, exact reproducibility; also `manager.set("quantization", ...)` hard path L3 refs).
- `smoothquant` via `manager.smooth_activations`: test/unit/test_smoothquant.py (6 refs; incl. "already quantized" RuntimeError); `collect_input_absmax`/`resolve_smoothing_groups` only indirectly.
- `check_coupling`: test/unit/test_known_bugs.py.
Indirect coverage through `manager.set/apply`:
- StructuredPruner (+CoupledPruner, share_mask, repeated_pattern, coupled_repeated_pattern, auto groups, position_linked): test/e2e/test_gqa_attention_pruning.py (10), test/unit/test_attention_semantics.py, test_gqa_rope_pruning.py, test/e2e/test_model_families.py, test_known_bugs.py.
- BlockPruner: test/unit/test_block_pruning.py (6), test_paged_attention.py.
- LRDer: test/unit/test_cascade_calibration.py (svd, aa-svd; 6 refs), e2e test_export_pipelines.py:1, test_model_families.py:1. `svd-llm-v2` only in examples/02_calibrated_lrd_svd_llm_v2.py, scripts, benchmarks (no test).
- Quantizer: test_export_pipelines.py (5), test_onnx_tensorrt_export.py (3), test_qnn_bias_split.py (2), test_known_bugs.py (2), test_lpbq_quantization.py, test_smoothquant.py; activation quant flags only in test_onnx_tensorrt_export.py (quantize_output_activation, precision_activation).
- UnstructuredPruner: only test/e2e/test_model_families.py (1 ref).
Untested (no test references at all):
- `quantize_hqq` (hqq), `quantize_gptq` (only scripts/executorch/qnn/full_quant_export.py and benchmarks/nonhom_quant), `compute_activation_scale_zp_maxmin` directly, `method_activation="dynamic_per_token"`, `Quantizer.strip_runtime_state` (via manager.prepare_for_save, no test), soft vs hard quantization equivalence, `sparsity>0` quantization, binary precision.
- `reapply_mask`, `remove_mask` (all compressors), `needs_grouping`, `_prune_extra_params` (ViT), CoupledPruner.revert / revert_chain / elementwise norm path directly, StructuredPruner.restore, BlockPruner.restore, LRDer.restore, gradient method for structured/unstructured (only in scripts), `low_rank_svd_llm_v2` and `low_rank_svd` (direct), `build_unstructured_mask` granularity="neuron"/int, `reduce_scores`, `validate_*` functions (no direct tests; some value errors reached via manager.set).
- No test enforces registry consistency (key <-> attribute name, validator/default pairing).

## 9. Adding a new compression method today vs. target

Today, touching files:
New *method* of an existing type (e.g. LRD method "foo"): (1) `lrd_methods/foo.py` + add to `lrd_methods/__init__.py` METHOD_FUNCTIONS; (2) `lrd.py` `VALID_METHODS` list, `CALIBRATED_METHOD_DICT`, and the hand-written if/elif dispatch in `LRDer.apply` (different signature per method); (3) if it needs a new calibration summary -> `calibration/summaries/*` and `SUPPORTED_SUMMARIES` (outside compression/). Same shape for quantization (also `VALID_METHODS` and the kwargs plumbing in apply) and pruning (VALID_METHODS in 2 places per type + if/elif in unstructured_pruning.py:52-71; structured uses SCORE_FUNCTIONS dict so only mask_generation/scoring + VALID_METHODS).
New *compressor type* ("foo"): (1) new `foo.py` with `FooCompressor(Compressor)` + validators; (2) `registry.py`: import it + validators, add to `COMPRESSOR_DICT`, `COMPRESSION_REGISTRY`, `APPLICATION_ORDER` (else silently appended last), optional `GROUP_OPTIONS`; (3) `compression/__init__.py` export; (4) config keys must equal attribute names (scheme.py `setattr`). Folder-external edits are normally NOT required for registry-driven types: scheme.py/manager.py/configuration.py loop over the registry. Exceptions that force outside edits: anything needing special manager flow (manager.py:309 smooth_activations, :948 Quantizer isinstance for prepare_for_save, `auto_groups`, `GROUP_OPTIONS` use), anything needing a model-structure key in `models/*/indexing_*.py` (block_pruning / smoothing / output_dependence), anything needing a new calibration summary (calibration/), and anything the exporters must understand (export/common.py reads Quantizer private attrs `_torchao_*`, `_act_quant_*`, `_soft_quant_precision`; utils/convert.py:14-23 duplicates the attr list).

What should change so only compression/ is touched:
1. Replace the three hand-edited tables with self-registration: `@register_compressor("foo", order=N, config={...})` on the class; the registry builds COMPRESSOR_DICT / COMPRESSION_REGISTRY / APPLICATION_ORDER / GROUP_OPTIONS from class attributes (`CONFIG_SCHEMA`, `ORDER`, `GROUP_OPTIONS`) with auto-discovery of modules in compression/.
2. Same for methods: each `*_methods/` package auto-registers functions via decorator carrying `name` and `requires_calibration`; one `METHOD_REGISTRY[type][method]` replaces VALID_METHODS lists, CALIBRATED_METHOD_DICT and the if/elif dispatch (uniform signature: `fn(weight, cfg, calib) -> result`; LRDer/Quantizer pass `calibration_store` and let the method pick what it needs).
3. Promote contract hooks into the ABC so manager/export have no special cases: `prepare_for_save(module)` (replace manager's `isinstance(Quantizer)`), `runtime_state_attrs()` (replace duplicated attr lists), optional `smooth`/pre-pass hook (move smoothquant behind a `Compressor.pre_apply(model, indexing, calib)` or a registered "transform" registry so `manager.smooth_activations` is generic), `export_info(module)` returning a backend-agnostic descriptor so export/common.py stops reading private `_torchao_*` attrs.
4. Move calibration collection out (smoothquant.collect_input_absmax -> a calibration summary `channel_absmax`), move generic module surgery (CoupledPruner internals, row/column replacement, Embedding/conv axis logic, `_prune_extra_params`'s param slicing) to utils/; keep only mask/ratio decision logic in compression/.
5. Move model/graph-topology resolution (`output_dependence` chains, `_hop_block_id`, sentinel names, `preprocessing_conv`, `position_linked`/`rope_prune_mask`) behind a scheme API (`scheme.coupled_targets()` / `scheme.extra_params()`) implemented in utils/ + indexing, so compressors consume resolved module chains and never read `scheme.path_template` / `final_layer_path`.
6. Share validators (`validators.py`), qrange helper, `_extract_aqt_scale`, `strip attr` list; make export call compression's `quantize_with_scale` rather than recomputing `round(w/scale)` (xnnpack_export.py:78-84, onnx/quantization.py:92,109,137).
7. Add a registry-consistency test (every schema key is an attribute, every validator accepts its default, every method function has a declared signature).
