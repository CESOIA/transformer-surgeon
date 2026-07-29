# Closing the XNNPACK decode-speed gap vs. ExecuTorch's own Llama export

Branch: `debug-xnnpackspeed` (based on `draft-attnkernel` @ `71e7da7`).
Date: 2026-07-29.
Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Llama2 architecture, GQA, 22 layers), XNNPACK backend, CPU. Same setup as `FRAMEWORK_PROBLEMS.md` / `/workspace/executorch_llama/TSURGEON_VS_EXECUTORCH.md`.

## Starting point

`FRAMEWORK_PROBLEMS.md` (P1) had already fixed a memory-planning bug but left a
**1.5x (fp32) / 2.5x (w4) decode-speed gap** against ExecuTorch's own llama
export "unexplained pending ETDump profiling" — the environment's ExecuTorch
build lacks `EXECUTORCH_ENABLE_EVENT_TRACER`, so no per-op timing was
available. That report's strongest remaining lead was that tsurgeon's graph
runs everything unbatched ((seq, dim), no leading batch dim) end to end,
unlike ExecuTorch's own (1, seq, dim) convention.

Rather than chase the batch-dimension theory (which would require reworking
every `blocks/` module's shape contract — a large, high-risk change), this
pass re-read `blocks/mha.py` / `blocks/decoder.py` line by line for concrete,
provable inefficiencies. Two were found, fixed, and verified to give a real,
reproducible speedup — see below.

## Root cause: two computations redundantly repeated once per layer instead of once per token

`TransformerDecoder.forward()` calls each of the 22 `TransformerDecoderBlock`s
per decode step, threading the same `pos_id`, `(q_pos, k_pos)`, `mask_penalty`,
and `(rope_cos, rope_sin)` buffers to every one of them unchanged. Inside
`MHACausal.forward()` (called once per layer), two lines derived per-step
values **from those same shared inputs, independently in every layer**:

```python
# blocks/mha.py, MHACausal.forward, before this fix
q_pos, k_pos = pos_id_list
attn_mask = torch.where((q_pos < k_pos), mask_penalty, torch.zeros_like(mask_penalty))[pos_id].unsqueeze(0)
...
cos = rope[0][pos_id]
sin = rope[1][pos_id]
```

Both `attn_mask` and the base `(cos, sin)` depend **only** on decoder-level
state (`pos_id`, the two position-grid buffers, `mask_penalty`, and the
`rope_cos`/`rope_sin` buffers) — none of which vary by layer. So the exact
same `(max_cache_len, max_cache_len)` compare + `where` + index, and the same
`rope_cos`/`rope_sin` index-by-`pos_id`, were being recomputed **22 times per
generated token** instead of once.

This matters more than the FLOP count suggests: `torch.where`, tensor
comparison, and advanced indexing are not part of XNNPACK's operator set, so
these ops run through ExecuTorch's non-delegated ("portable") interpreter path
— each occurrence pays real per-op dispatch/tensor-allocation overhead that
XNNPACK-delegated matmuls don't. Repeating that 22x per token, every token, is
pure waste with no effect on the result.

## Fix

Hoisted both computations up to `TransformerDecoder.forward()` (computed once
per token) and threaded the already-computed values down to every block/attn
call instead of the raw ingredients:

- `transformersurgeon/blocks/decoder.py`: `TransformerDecoder.forward` now
  computes `attn_mask` and `rope_pos = (rope_cos[pos_id], rope_sin[pos_id])`
  once, before the per-block loop, and passes them to
  `TransformerDecoderBlock.forward(..., attn_mask=..., rope=rope_pos)`.
- `transformersurgeon/blocks/mha.py`: `MHACausal.forward` signature changed
  from `(x, pos_id, pos_id_list, mask_penalty, ...)` to
  `(x, pos_id, attn_mask, ...)`; it now consumes the precomputed mask and an
  already-pos_id-indexed `rope` tuple directly, instead of re-deriving them.
  Each layer still applies its own (possibly pruning-projected) split via
  `_project_rope` — only the shared, layer-invariant base lookup was hoisted.
- Updated the two unit tests that called `MHACausal.forward` directly with the
  old `(pos_id_list, mask_penalty)` calling convention
  (`test_mha_causal_custom_sdpa.py`, `test_gqa_rope_pruning.py`) to precompute
  the equivalent mask/rope values the same way `TransformerDecoder` now does.

This is a pure dead-computation-elimination change: it does not alter what is
computed, only when/how many times.

## Correctness verification

- `pytest test/unit` — **35/35 pass**, unchanged.
- `pytest test/e2e` — **57/57 pass, 4 skipped** (TensorRT/QNN unavailable in
  this environment, pre-existing), including `test_export_xnnpack` and
  `test_export_xnnpack_custom_sdpa` (both attention-kernel paths this change
  touches).
- Re-exported both benchmark variants and reran tsurgeon's own inference-stats
  check against the float model:
  - `fp32`: `max_abs_err=0.000144`, `mean_abs_err=0.000027` — matches the
    pre-fix numbers (`FRAMEWORK_PROBLEMS.md`'s P1 section reported
    `max_abs_err≈1.6e-4`).
  - `w4`: `max_abs_err=2.074`, `mean_abs_err=0.372` — matches the expected
    4-bit weight-quantization noise floor (pre-fix: `max_abs_err≈2.0`), i.e.
    unchanged, not a new regression.

## Benchmark methodology note (read before trusting any single number here)

This machine is shared and was under heavy, variable load while benchmarking
(`uptime` load average ~40 on a 112-core box). A naive sequential
multi-model benchmark (load model A, run N trials; load model B, run N
trials; ...) produced an internally inconsistent result for the fp32 variant
across repeated runs (sometimes showing the fix as faster, once showing it as
slower) — the smaller fp32 delta was within the noise floor of drifting
system load between when each model happened to run.

Switching to an **interleaved A/B script** (both `.pte` variants loaded once
in the same process, trials alternated `A, B, A, B, ...` so both variants see
the same system-load conditions at the same wall-clock times, with 8 discarded
warm-up iterations per variant before measuring) gave clean, low-variance,
reproducible results — stdev dropped from run-to-run swings of several tok/s
down to ~0.1–0.3 tok/s. **All numbers below are from that interleaved
methodology** (script: see Reproduce section).

One incidental finding worth flagging for anyone benchmarking this export
path in the future: with only 1–3 warm-up iterations (this repo's default —
see `benchmark_tsurgeon.py --warmup`), decode throughput can still be in a
transient ramp-up state for several more iterations before reaching steady
state. Under-warmed measurements are a plausible partial explanation for some
of the run-to-run variance seen historically in this comparison; **at least
8 warm-up iterations are recommended** before trusting a decode-speed number
from this export path.

## Results

| Variant | Baseline decode (tok/s) | Fixed decode (tok/s) | Speedup | stdev (baseline / fixed) |
|---|---:|---:|---:|---:|
| fp32 | 18.87 | 20.23 | **+7.2%** (1.072x) | 0.10 / 0.09 |
| w4 (attn+mlp INT4) | 33.31 | 38.45 | **+15.9%** (1.159x) | 0.32 / 0.76 |

(15 interleaved trials each, 8 discarded warm-up iterations, same prompt/greedy-decode/max-new-tokens=64 setup as `TSURGEON_VS_EXECUTORCH.md`.)

Remaining gap vs. ExecuTorch's own export (reference numbers unchanged by this
work — see `TSURGEON_VS_EXECUTORCH.md`: `xnnpack_fp32` 27.1 tok/s,
`xnnpack_q8da4w` 80.4 tok/s decode):

| Variant | Gap before this fix | Gap after this fix |
|---|---:|---:|
| fp32 | 1.44x (27.1 / 18.87) | **1.34x** (27.1 / 20.23) |
| w4 | 2.41x (80.4 / 33.31) | **2.09x** (80.4 / 38.45) |

## Why w4 benefits more than fp32

Quantized (w4) linears run through fast fused INT4/INT8 XNNPACK kernels, so
the model's total per-token compute time is much shorter. The fixed, per-layer
dispatch overhead this change removes is the same in absolute terms for both
variants, but it's a **larger fraction of a shorter total** in the w4 case —
hence the bigger percentage win (+15.9% vs +7.2%). This is consistent with
(and a nice confirmation of) the mechanism claimed above: the win comes from
removing non-delegated per-op dispatch overhead, which doesn't scale with the
delegated matmul FLOP count.

## What this does *not* fix

The remaining ~1.34x/2.09x gaps are not attributed to any single newly
identified cause in this pass. `FRAMEWORK_PROBLEMS.md`'s other findings are
untouched and still open:
- **P2** — batched/dynamic-shape prefill and multi-token KV-cache writes are
  still unsupported (prefill is still a token-by-token loop).
- **P3** — embedding-table and `lm_head` hard quantization export bugs are
  still open, so `w4` here still leaves those two layers in fp32 (unlike
  ExecuTorch's `xnnpack_q8da4w`, which quantizes both).

Fully explaining what's left of the gap still needs ETDump per-op profiling
(requires an ExecuTorch build with `EXECUTORCH_ENABLE_EVENT_TRACER`,
unavailable in this environment) to see inside the XNNPACK delegate blobs
themselves — the fixes in this pass only addressed the non-delegated
"glue" computation around them.

## Files changed

- `transformersurgeon/blocks/decoder.py` — hoist mask/RoPE-index computation.
- `transformersurgeon/blocks/mha.py` — `MHACausal.forward` consumes precomputed values.
- `test/unit/test_mha_causal_custom_sdpa.py`, `test/unit/test_gqa_rope_pruning.py` — updated calling convention.

## Reproduce

```bash
conda activate py312_executorch_xnnpack
cd /workspace/transformer-surgeon && pytest test/unit test/e2e -q   # correctness

cd /workspace/executorch_llama
python tsurgeon_export_llama.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --variant fp32 --out-dir pte_fixed
python tsurgeon_export_llama.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --variant w4   --out-dir pte_fixed

# interleaved A/B (script written for this investigation, kept alongside the
# other benchmark scripts in this sibling dir for provenance -- see FRAMEWORK_PROBLEMS.md's
# convention of keeping repro scripts in /workspace/executorch_llama/, outside this repo)
python interleaved_ab.py pte/tsurgeon_llama_fp32.pte pte_fixed/tsurgeon_llama_fp32.pte 15
python interleaved_ab.py pte/tsurgeon_llama_w4.pte   pte_fixed/tsurgeon_llama_w4.pte   15
```
