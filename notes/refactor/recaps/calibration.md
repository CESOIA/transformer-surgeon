# Recap: transformersurgeon/calibration/ (backbone.py, raw_data/, summaries/)

## 1. Purpose and call flow
Runs the model over a calibration DataLoader, taps selected `LinearCompressed` modules with hooks (raw streams), reduces
them into per-module "summaries" (covariance, min/max, grads, raw tensors) stored in `scheme.calibration_data` (dict, per
compression scheme). Compressors later read `self.calibration_store[<summary name>]`. Two-level plugin design:
RawDataCollector (what tensor to tap) -> CalibrationSummary (how to reduce it), glued by `SummaryRuntime` (per scheme x summary).

Flow (`backbone.run_compression_calibration`, called from `utils/manager.py:611` (run_calibration), `utils/manager.py:673`
(extract_summaries), `utils/cascade.py:219` (block-wise cascade)):
1. validate DataLoader; per stage in `target_stages` (list of lists of `(scheme, summary_names)`):
2. `get_summary(name)` -> `summary.initialize_store(scheme.calibration_data)`; wrap in `SummaryRuntime`; union of `summary.required_raw_data`.
3. `instantiate_raw_collector(raw_name, offload_to_cpu=)` once per raw name; check `requires_shifted_model` (needs `shifted_model`), `requires_backward` (needs `loss_fn`).
4. register `collector.build_forward_hook(emit_raw=...)` on `scheme.get_compression_module()` (or the same-path module of the shifted model if `collector.uses_shifted_model`).
5. per batch: `runtime.reset_batch()`, `normalize_calibration_batch` -> (args, kwargs, label), `move_to_device`; forward (+loss+`backward()` and `collector.collect_after_backward(module)` when backward needed; shifted model forward under no_grad).
6. hooks emit raw tensors -> `_dispatch_raw` -> `SummaryRuntime.on_raw` (FIFO pairing across required streams) -> `summary.update_runtime` -> `update_from_raw`.
7. end: `summary.finalize_store`; optional `torch.save` dump per (summary, scheme path) to `summary_dump_dir/<summary>/<urlencoded path>.pt`; returns number of schemes.

## 2. File tree (lines)
```
calibration/__init__.py            11  exports run_compression_calibration, SUPPORTED_SUMMARIES
calibration/backbone.py           382
calibration/raw_data/__init__.py   13
calibration/raw_data/base.py      179
calibration/raw_data/activation.py 87
calibration/raw_data/weight_grad.py 33
calibration/raw_data/params.py     52
calibration/raw_data/registry.py   34
calibration/summaries/__init__.py  14
calibration/summaries/base.py     201
calibration/summaries/activation_range.py 72
calibration/summaries/covariance.py 98
calibration/summaries/weight_grad.py 32
calibration/summaries/raw_activation.py 38
calibration/summaries/params.py    26
calibration/summaries/registry.py  44
```
(no `__pycache__`/data files of interest; `raw_data/`, `summaries/` hold code only, no dumped data.)

## 3. Per file: symbols
**backbone.py**
- `_compute_calibration_loss(loss_fn, model_output, label)` -> `loss_fn(...)` or None (trivial wrapper, :23).
- `run_compression_calibration(model, calibration_data, target_stages, shifted_model=None, shifted_calibration_data=None, loss_fn=None, max_batches=None, device=None, offload_to_cpu=False, verbose=False, show_progress=True, summary_dump_dir=None, summary_dump_names=None) -> int`: the whole pass (see 1). Contains nested closures `_update_shifted_sanity_stats` (:158, rel-L2 diff of `activation` vs `activation_shifted`, printed when verbose) and `_dispatch_raw` (:191). Single 350-line function.

**raw_data/base.py**
- `class RawDataCollector(ABC)`: attrs `name`, `requires_backward=False`, `requires_loss=False`, `requires_labels=False`, `uses_shifted_model=False`; `build_forward_hook(*, emit_raw) -> hook|None` (default None); `collect_after_backward(module) -> Tensor|None` (default None). Not actually abstract (no @abstractmethod).
- `normalize_calibration_batch(batch, batch_id) -> (args tuple, kwargs dict, label)`: handles `{"__ts_args__","__ts_kwargs__","__ts_label__"}` cascade fast-path, Mapping, 1-/2-tuple (data,target)/(target,data), positional tuple, single tensor. This is a *batch utility*, not a collector.

**raw_data/activation.py**
- `ActivationCollector(name="activation")` `__init__(*, offload_to_cpu=False, raw_name="activation")`; hook emits `inputs[0]` flattened to [tokens, features], float32, checks last dim == `module.in_features`.
- `ShiftedActivationCollector(ActivationCollector)` name `"activation_shifted"`, `uses_shifted_model=True`.
- `OutputActivationCollector(RawDataCollector)` name `"output_activation"`; hook emits module `output` flattened, checks `out_features`.

**raw_data/weight_grad.py**: `WeightGradCollector` name `"weight_grad"`, `requires_backward=True`, `requires_loss=True`; `collect_after_backward` returns `module.weight.grad.detach().clone().float()` (None if no grad).

**raw_data/params.py**: `WeightCollector` (`"weight"`) and `BiasCollector` (`"bias"`): forward hooks that emit `module.weight`/`module.bias` clone as float32 (parameters, not activations; bias raises ValueError if None).

**raw_data/registry.py**: `RAW_DATA_REGISTRY` (class, not instance) keys: `activation, activation_shifted, output_activation, weight_grad, weight, bias`. `SUPPORTED_RAW_DATA` (tuple of keys). `instantiate_raw_collector(raw_name, *, offload_to_cpu)` (all collectors must accept that kwarg).

**summaries/base.py**
- `CalibrationSummary(ABC)`: attrs `name`, `required_raw_data: tuple[str]`, `requires_shifted_model=False`; `initialize_store(store)` (pops own key), abstract `update_from_raw(store, raw_data)`, `update_runtime(runtime, store, raw_data)` (default -> update_from_raw), `finalize_store(store)` (no-op). Singletons shared across schemes.
- `@dataclass SummaryRuntime(summary, pending_raw={}, num_updates=0, state={})`: `reset_batch()` (raises if leftover unpaired raws), `on_raw(store, raw_name, raw_value)` (FIFO queues, fires when all required streams non-empty). `state` annotated `Dict[str,int]`.
- `unique_summaries(names)` order-preserving dedupe.

**summaries/activation_range.py**: `ActivationRangeSummary` (`"activation_range"`, raw `activation`) and `OutputActivationRangeSummary` (`"output_activation_range"`, raw `output_activation`): store `{"min": 0-dim tensor, "max": ...}` running over batches. Both redundantly override `initialize_store` with the base behaviour.

**summaries/covariance.py**
- `_BaseCovarianceSummary`: template (`_compute_batch_sum_and_tokens` -> `_validated_...` -> token-weighted running mean E[x^T x], `runtime.state["token_count"]`).
- `CovarianceSummary` (`"covariance"`, `activation`: X^T X), `ShiftedCovarianceSummary` (`"shifted_covariance"`, `activation_shifted`, needs shifted model), `CrossCovarianceSummary` (`"cross_covariance"`, `activation`+`activation_shifted`: X^T X_shifted, needs shifted model).

**summaries/weight_grad.py**: `WeightGradSummary` (`"weight_grad"`): running mean of grads over batches (`runtime.state["num_batches"]`).

**summaries/raw_activation.py**: `_BaseRawActivationSummary` (appends every raw tensor to a list in `update_runtime`, `finalize_store` torch.cat dim 0; `update_from_raw` raises NotImplementedError), `InputActivationSummary` (`"input_activation"`, raw `activation`), `OutputActivationSummary` (`"output_activation"`, raw `output_activation`). NOTE: summary key `output_activation` == raw key `output_activation` (separate registries, no clash).

**summaries/params.py**: `WeightSummary` (`"weight"`), `BiasSummary` (`"bias"`): last-batch overwrite.

**summaries/registry.py**: `SUMMARY_REGISTRY` (instances) keys: `covariance, shifted_covariance, cross_covariance, weight_grad, activation_range, output_activation_range, weight, bias, input_activation, output_activation`. `SUPPORTED_SUMMARIES` tuple; `get_summary(name)` (ValueError if unknown).

## 4. Dependencies
Imports (outside calibration): `backbone.py:20-21` -> `..utils.interface.build_progress_bar`, `..utils.utils.{get_submodule, infer_model_device, move_to_device}`. Everything else only torch + intra-calibration. No import of compression/, models/, export/.
Implicit (duck-typed) coupling to the "scheme" object from compression/utils: `scheme.get_compression_module()`, `scheme.path`, `scheme.calibration_data`, `scheme.debug_full_path` (optional), and the `target_stages` structure built by `utils/manager.py::collect_calibration_target_stages`.

Imported by (whole repo):
- `calibration/__init__` <- `utils/manager.py:16` (`run_compression_calibration, SUPPORTED_SUMMARIES`), `utils/cascade.py:19` (`run_compression_calibration`).
- `calibration.raw_data.normalize_calibration_batch` <- `utils/cascade.py:20` (used :600, :639), `compression/smoothquant.py:32` (used :97). (compression importing calibration = upward-ish dependency.)
- `utils/manager.py:662` validates summary names against `SUPPORTED_SUMMARIES` (extract_summaries).
- `compression/{lrd,quantization,structured_pruning,unstructured_pruning}.py` reference summary names as bare strings (`needs_calibration()`), no import: `"covariance"` (lrd.py:10, quantization.py:19 gptq), `"cross_covariance"/"shifted_covariance"` (lrd.py:11), `"weight_grad"` (structured_pruning.py:39, unstructured_pruning.py:12), `"activation_range"/"output_activation_range"` (quantization.py:48-51,162-198). `compression/abstract.py:38-60` docs.
- Nothing in test/, benchmarks/, scripts/, examples/ imports `transformersurgeon.calibration` directly (they go through the manager: `set_calibration_data`, `run_calibration`, `extract_summaries`). `scripts/summaries/extract_summaries_qwen.py` mentions SUPPORTED_SUMMARIES only in a comment.
- Within calibration: backbone -> raw_data, summaries; registries -> concrete classes; concrete -> `.base`.

## 5. Violations of the target rule
Model names: none in code (grep qwen/llama/bert: zero hits in calibration/).
Compression-application / compression-name leakage:
- `summaries/activation_range.py:16,50` docstrings cite `Quantizer.apply()`, PT2E observer injection, "XNNPACK" (also `raw_data/activation.py:63`); `summaries/registry.py:13` "svd-llm-v2", `:19`; `summaries/base.py:57` "AA-SVD", `:29,40,47` "Compressor.needs_calibration". Comment-level only, but the framework documents consumers.
- Names encode consumers: `weight_grad` "gradient-aware pruning" (raw_data/registry.py:17).
- `raw_data/base.py:20,32,110` assume `LinearCompressed` modules; collectors assume `in_features/out_features/weight/bias` attributes (activation.py:25,69; weight_grad.py:27; params.py:25,39) -> linear-only, fine as scope but not documented as contract.
- `backbone.py:103,141,325,340-343`: depends on compression "scheme" API (`get_compression_module`, `.path`, `.calibration_data`, `.debug_full_path`). Acceptable if a tiny Protocol is defined; currently implicit.
- `backbone.py:158-187` hard-codes stream names `"activation"`/`"activation_shifted"` (summary-specific knowledge inside the generic runner; the shifted sanity check belongs in a summary/collector or a diagnostic hook).
- Error messages mention `manager.set_calibration_data/set_calibration_loss/run_calibration/apply` (backbone.py:61,135,149) -> backbone knows the manager API.
Export logic: none.
Generic helpers that belong in utils/ (not calibration):
- `normalize_calibration_batch` (raw_data/base.py:120-179) is a dataloader-batch helper used by cascade and smoothquant; the `__ts_*__` keys (:122-148) are a protocol owned by `utils/cascade.py` (:614-616,650-652,683-685) -> move both to `utils/` (e.g. `utils/batches.py`) and have cascade+calibration import it.
- Summary dump-to-disk (backbone.py:330-358, `torch.save`, url-quoted path) is a debug utility; could be a separate `calibration/dump.py` or utils.
- Device/progress helpers are already in utils (good).

## 6. Duplicated code
Within calibration:
- Flatten-to-[tokens,features]+offload+float: `raw_data/activation.py:27-43` vs `:75-85` (OutputActivationCollector) -> one `_flatten_tokens(t, offload)` helper or make Output a param of a single `ModuleTensorCollector(which="input"|"output")`.
- `offload_to_cpu` `__init__` repeated in WeightGrad/Weight/Bias/Output collectors (weight_grad.py:22, params.py:19,34, activation.py:67) -> put in base `RawDataCollector.__init__`.
- `WeightCollector`/`BiasCollector` (params.py:15-52) differ only by attribute name -> one `ParamCollector(attr)`.
- `ActivationRangeSummary` vs `OutputActivationRangeSummary` (activation_range.py:25-39 vs 58-72) identical except stream key -> one base with `required_raw_data[0]`; likewise raw_activation already does it correctly.
- Three covariance classes share structure (covariance.py:51-98): Cov/ShiftedCov identical modulo key; can be a parametrised `_Gram(a_key, b_key=None)`.
- `WeightSummary`/`BiasSummary` (params.py:14-26) differ by key only. `initialize_store` overrides in activation_range.py:23,54 duplicate base (:63-77 of base.py).
- Running mean idiom in covariance.py:36-45 and weight_grad.py:22-30 (weighted by tokens vs batches) -> shared `running_mean(store,state,key,value,weight)` helper in summaries/base.
Cross-folder:
- `compression/smoothquant.py:78-103 collect_input_absmax` re-implements a calibration pass (hook on inputs, batch normalize, device move, loop over loader) outside calibration; could be an `input_absmax` summary + collector (per-channel max |x|) executed by the backbone (it only needs the `ActivationCollector` stream and a summary reducing `amax(0)`). Also `utils/cascade.py:600-700` builds the `__ts_*` batches again.
- Batch loop / device-move pattern also in `utils/cascade.py` (:600-700); unify via shared batch helper.

## 7. Stale / unused (grep-verified)
- `RawDataCollector.requires_loss` / `requires_labels` (raw_data/base.py:47,52): only set in base and weight_grad.py:17,19, never read anywhere (backbone uses only `requires_backward`, `uses_shifted_model`). Docstring admits "reserved".
- `unique_summaries` (summaries/base.py:195): only defined and re-exported in summaries/__init__.py; zero callers (manager has its own dedupe? not checked -> unknown).
- `SUPPORTED_RAW_DATA`: only used inside registry error message + re-export.
- `RawDataCollector` is an ABC with no abstract methods; `_compute_calibration_loss` (backbone.py:23) is a pointless wrapper over `loss_fn(...)` (None check duplicated at :276).
- `_BaseRawActivationSummary.update_from_raw` raises NotImplementedError (abstract method forced by ABC; `update_from_raw` should be non-abstract or `update_runtime` the abstract one).
- `ActivationCollector.name = "activation"` class attr then overwritten by `raw_name` in `__init__` (activation.py:15,20); `ShiftedActivationCollector.__init__` passes `self.name` (works via class attr).
- `SummaryRuntime.state: Dict[str,int]` annotation wrong for general use (guess: harmless). `from collections.abc import Mapping` + `typing.Mapping` mix in base.py (cosmetic).
- `quantization.py` has `quantize_output_activation` using `output_activation_range`; `output_activation_range` also used in extract flow only through that. Not stale.

## 8. Test coverage (grep test/)
No test imports `transformersurgeon.calibration` directly; coverage is indirect via manager.
- `test/unit/test_extract_summaries.py` (CPU, tiny Qwen2): covers `weight, bias, input_activation, output_activation, covariance` summaries and their raw collectors (`weight`, `bias`, `activation`, `output_activation`), the dispatcher/backbone forward-only path, unknown-summary and empty-list errors, bias-missing error.
- `test/unit/test_cascade_calibration.py`: aa-svd on qwen2 with cascade mode (exercises `shifted_covariance`, `cross_covariance`, `activation_shifted`, `shifted_model`, `normalize_calibration_batch` `__ts_*` path) -> only one positive test (`test_qwen2_cascade_calibration_still_works`) plus BERT error tests.
- `test/unit/test_onnx_tensorrt_export.py:166-200` (needs CUDA/TensorRT): static act quantization through `manager.apply` -> `activation_range` (not `output_activation_range`, `quantize_output_activation=False`).
- `test/unit/test_smoothquant.py`: `normalize_calibration_batch` indirectly via `collect_input_absmax`.
- `test/unit/test_qnn_*`, `test_known_bugs.py`: use `calibration_token_ids` for export (unrelated to this package).
UNTESTED: `weight_grad` raw+summary and the backward path (`requires_backward`, `loss_fn`, `collect_after_backward`) — no test sets `set_calibration_loss`; `output_activation_range`; `ShiftedActivationCollector`/`OutputActivationCollector` only via the summaries above; `activation_range` only in a CUDA test; GPTQ/svd-llm-v2 (`covariance` via compressor) calibration; `summary_dump_dir/names`; `offload_to_cpu`; `max_batches`; `shifted_calibration_data`; the shifted sanity stats; `SummaryRuntime.reset_batch` mismatch error and FIFO multi-update; `unique_summaries`; `normalize_calibration_batch` branches (tuple/(target, data)/single tensor) have no direct unit test; error paths (non-DataLoader, no batches, no raw collected).

## Refactor hints (guesses)
- Make `target_stages` entries a small Protocol/dataclass (`CalibrationTarget(module, store, path, summaries)`) so backbone stops touching scheme internals.
- Adding a technique = new file in `raw_data/` (optional) + `summaries/` + one line in each registry; today that already holds, except backbone sanity check, batch protocol and compressors using string keys.
- Split `run_compression_calibration` (stage setup / batch loop / finalize+dump).
