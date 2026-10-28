# Recap: benchmarks/

Two self-contained research scripts (non-homogeneous LRD / quantization on Qwen2-1.5B, PPL on WikiText-2)
plus one pure-torch microbenchmark and some artifacts. No package, no tests, nothing in the repo imports
or references these (grep: only self + docs mentions of "benchmarks/" unrelated).

## File tree (lines; binaries shown with size)
```
benchmarks/context_increase_benchmark.py       173
benchmarks/attention_benchmark_cpu.png         PNG 1000x600, 89 KB   (wc -l: 231, meaningless)
benchmarks/attention_benchmark_cuda.png        PNG 1000x600, 82 KB   (wc -l: 172)
benchmarks/example_image.jpg                   JPEG 1147x816, 152 KB (wc -l: 338)
benchmarks/example_image2.png                  PNG 1600x1200, 3.7 MB (wc -l: 13203)
benchmarks/nonhom_lrd/run_benchmark.py         748
benchmarks/nonhom_quant/run_benchmark.py       700
```
No results/ dirs are checked in (scripts write to `<script dir>/results`).

## Per file

### context_increase_benchmark.py (173) - attention vs SDPA vs KV concat timing
- CLI: none. `__main__` runs `benchmark("cpu")` then `"cuda"` if available. Writes `attention_benchmark_cpu.csv`
  (bug: hardcoded to cpu name even for cuda run, :150 -> overwritten; printed message also says "CPU") and
  `attention_benchmark_{device}.png` to CWD.
- transformersurgeon API: NONE (imports only torch, time, math, pandas, matplotlib, os).
- Bug: `torch.cuda.synchronize()` is called unconditionally in all three timing fns (:67,74,82) -> crashes on the CPU run
  on CUDA-less machines (and for the CPU case measures nothing device-specific). The checked-in cpu.png implies it was run
  on a CUDA machine.
- Hardcoded: heads=60, kv_heads=4, head_dim=128, seq_max=2**16, 50 reps.
- Duplication: `attention()` (:7-61) is a copy of the library's explicit GQA attention
  (`transformersurgeon/blocks/mha.py`, MHACausal/MHABase `use_sdpa=False` path). Not imported, so it is a frozen
  snapshot that can drift. Docstring even says "needed for ONNX export" - same rationale as the library block.
- Stale: one-off exploration of "do we cat KV cache or use scatter" (motivation for `cache_impl`
  mutable/io_scatter/io_concat). Output PNGs are its artifacts.

### nonhom_lrd/run_benchmark.py (748) - iso-MACs non-homogeneous LRD
CLI: `--model-name` (Qwen/Qwen2-1.5B), `--lrd-ratio` (0.4), `--num-calibration-samples` (32), `--num-test-samples` (50),
`--seq-len` (2048), `--decompress-steps N...` (default 0 2 4 8 12 16), `--skip-calibration`, `--skip-baseline`,
`--output-dir` (<dir>/results), `--device`.
transformersurgeon API:
- `from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress` (:56)
- DEEP: `from transformersurgeon.blocks import LinearCompressed` (:57); used for `isinstance` and attrs
  `.rank` (int vs "full"), `.in_features`, `.out_features`, `.bias` in `compute_model_ops_params` (:192-225)
- `Qwen2ForCausalLMCompress.from_pretrained(name, torch_dtype=torch.float16)` (:129,250,~385-395, calibration ~358)
- `Qwen2CompressionSchemesManager(model)`
- `manager.set("lrd","method","svd-llm-v2",criteria="all")`
- `manager.set("lrd","rank", int | "full", criteria=<full dotted path e.g. "model.layers.3.mlp.up_proj">)` per layer
  (:360-363, :402-403)  -> relies on path-string criteria matching full module names
- `manager.set_calibration_data(loader)`, `manager.run_calibration(criteria="all", device=, offload_to_cpu=True, show_progress=True)` (:372-376)
- `manager.save_state(path)` / `manager.load_state(path)` (:380,:407) - calib state reuse across ranks
- `manager.apply(hard=False, criteria="all", device=, offload_to_cpu=True)` (:412)
- `model.save_pretrained(exp_dir)` per experiment (:418) of a soft-LRD model (HF save, not export_to_hf)
Flow: layer_info -> `compute_lrd_errors` (SVD of each W, relative Frobenius truncation error, :243) ->
`sort_units_by_lrd_error` -> `compute_experiment_ranks` (scale ranks so MACs == homogeneous budget, :298) ->
calibrate once -> per-n fresh model/apply/PPL/save -> results.json + ppl_vs_budget.png.
Docstring says error is "||W-W_lrd||_F" absolute; code uses relative (:243-273) - docstring stale vs code.

### nonhom_quant/run_benchmark.py (700) - iso-FLOPs INT8/INT4 mixed precision
CLI: same as lrd minus `--lrd-ratio` (default steps 0 2 4 8 12 16); defaults in module constants :69-74.
API:
- `from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress` (:65)
- `manager.set("quantization", <prop>, <value>, criteria=...)` for props: `precision` ("full"|8|4), `method` ("gptq"),
  `granularity` ("per_channel"), `sparsity` (0.0), `sparse_method` ("magnitude"), `eps` (1e-6),
  `method_activation` ("maxmin"), `scheme_activation` ("asymmetric") (:317-336). `precision_activation` lines are commented out.
  Criteria are `["self_attn","mlp"]` (OR substrings) and per-unit `[[i,"self_attn"]]`/`[[i,"mlp"]]` (AND lists).
- `set_calibration_data`, `run_calibration(criteria=PROJ_CRITERIA, ...)` (:339-370), `save_state`, `load_state`, `apply(hard=False, criteria=PROJ_CRITERIA, ...)` (:372-408)
- Private attr read: `mod._soft_quant_precision` (:206-230) to detect INT4/INT8 layers after soft apply
  (set in compression/quantization.py:151, listed in :267 cleanup list).
Assumption: INT8 calibration stats (GPTQ XtX + act min/max) are reused for INT4 - correctness claim in docstring, not enforced.
Docstring/CLI help inconsistent: "Numbers of fp16 units" ; docstring "--decompress-steps 0 2 4 8 12 16".

## Verified imports
All resolve: top-level Qwen2 classes (models/qwen2_c/define_qwen2_c.py:22,30); `transformersurgeon.blocks.LinearCompressed`
(blocks/linear_compressed.py via blocks/__init__.py:1; `.rank` at :45,65). Manager methods exist at
utils/manager.py: set :143, set_calibration_data :527, run_calibration :580, apply :701, save_state :889,
load_state :905. Quantization property names all appear in compression/quantization.py (sparse_method, method_activation,
scheme_activation, granularity, eps, sparsity, gptq, maxmin). LRD method "svd-llm-v2" valid (compression/lrd.py:8).
No stale imports. Not executed (needs GPU + model download).

## Duplicated library logic
- `build_loaders` WikiText-2 tokenize/chunk: lrd:99-139 == quant:111-149 (identical), and same pattern in
  examples/02:47-73, scripts/executorch/{qnn/full_quant_export.py:245, qnn/exporter_function_test.py:227, xnnpack/exporter_function_test.py:154}.
- `evaluate_perplexity`: lrd:142 == quant:152 (identical); other PPL evaluators likely under scripts/.
- `build_decompression_units`, `unit_full_paths`, `plot_results`, `main` skeleton, `run_calibration_and_save`, `run_experiment`,
  args: ~50% of the two files is copy-paste; should share a common module (data, ppl, units, plotting, results.json).
- MAC/param counting (`compute_model_ops_params` lrd:192, `compute_baseline_flops_params`/`compute_effective_flops_params`
  quant:187-230): reimplements a model-wide cost model; the library has `export/common.py` layer-info introspection and
  per-module `rank`/`_soft_quant_precision`, but no FLOPs/params counter. Candidate for a library utility.
- `compute_lrd_errors` lrd:243 re-does SVD that `compression/lrd.py` also performs.
- context_increase_benchmark.attention(): see above (copy of mha.py explicit attention).

## Model-specific hardcoding
- Qwen2 only: both import Qwen2 classes; default `Qwen/Qwen2-1.5B`; `ALL_PROJECTIONS`/`MHA_PROJECTIONS`/`MLP_PROJECTIONS`
  suffix lists (`self_attn.q_proj,k_proj,v_proj,o_proj`, `mlp.gate/up/down_proj`) are Qwen/Llama naming.
- lrd:295 `f"model.layers.{i}.{s}"` hardcodes HF Qwen2 module paths (breaks if layer naming/indexing changes).
- lrd:252 comment and :627 "1536" assume Qwen2-1.5B dims; `SKIP_LAYERS={"k_proj","v_proj"}` (GQA-specific rationale).
- Budget constants: FLOPS INT8=0.5, INT4=0.25 (quant:70-74); "Max feasible units" calc assumes uniform unit cost.
- Title strings in plots hardcode "Qwen2-1.5B" (lrd:445, quant:428).
- context_increase: GQA 60/4/128 shape (looks like Qwen-72B/large-model-like), seq 65536.

## Stale / superseded assessment
- `example_image.jpg`, `example_image2.png` (3.7 MB): no reference anywhere in the repo (grep for the names finds nothing
  in code/docs); no VLM script in benchmarks uses them. Probably leftovers from a VLM test -> candidates to delete or move
  to test assets.
- `attention_benchmark_{cpu,cuda}.png`: generated outputs of context_increase_benchmark.py (dated Jun 29), not source.
  The script can't regenerate cpu.png on a CPU-only machine (unconditional cuda.synchronize).
- `context_increase_benchmark.py`: one-off, superseded by the `cache_impl` work; no library deps.
- nonhom_lrd/nonhom_quant: current-looking (Jul 2) and API-compatible, but not connected to docs/tests; results not stored.
  Both are useful as end-to-end regression tests of manager.set/calibrate/save_state/load_state/apply.
- Refactor-sensitive surface (what to update if APIs move): Qwen2 class names, `CompressionSchemesManager`
  methods (`set`, `set_calibration_data`, `run_calibration`, `save_state`, `load_state`, `apply`), criteria syntax
  (list-of-AND-lists / OR-list / "all" / full path string), quantization property names, `blocks.LinearCompressed.rank`,
  `_soft_quant_precision`.
