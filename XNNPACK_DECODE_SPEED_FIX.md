# Closing the XNNPACK decode-speed gap vs. ExecuTorch's own Llama export

Branch: `debug-xnnpackspeed` (based on `draft-attnkernel` @ `71e7da7`).
Date: 2026-07-29, updated 2026-09-10.
Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Llama2 architecture, GQA, 22 layers), XNNPACK backend, CPU. Same setup as `FRAMEWORK_PROBLEMS.md` / `/workspace/executorch_llama/TSURGEON_VS_EXECUTORCH.md`.

---

## TL;DR — gap closed

Two independent head-to-head runs against Meta's actual `.pte` files (same process,
interleaved A/B, run on different days of this investigation — see "Independent
re-verification" below for why the two runs' precise numbers differ):

| Variant | Run 1 (2026-09-10, 15 trials) | Run 2 (2026-09-10, 20 trials, re-verification) |
|---|---:|---:|
| fp32 | tsurgeon 1.14x *faster* (34.5 vs 30.3 tok/s) | tsurgeon 1.03x *faster* (34.1 vs 33.0 tok/s) |
| quantized (`q8da4w` vs `w4`) | 0.98x — parity (86.2 vs 87.7 tok/s) | 0.94x (85.5 vs 90.9 tok/s) |

Starting point was 1.44x / 2.41x *slower*. **Both runs agree tsurgeon fp32 now matches or
slightly beats Meta's own export, and quantized is within ~2-6% of it** — a dramatic
reversal from the starting gap, achieved via three findings, in order of impact:

1. **`custom_sdpa` was scanning the whole KV cache every token** (`start_pos=0` + explicit
   full-length mask instead of `is_causal=True` + real `start_pos`) — **the dominant cost**.
   fp32 1.51x, w4 2.35x over the fix-2 baseline. This alone closed almost the entire gap.
2. **Mask/RoPE recomputed once per layer instead of once per token** — fp32 1.07x, w4 1.16x.
3. **Batch dimension theory — tested and refuted** (batched was *slower*); kept as an
   opt-in, off-by-default flag.

The residual few percent on the quantized row is not a mystery: tsurgeon's `w4` is
weight-only INT4 over attn+mlp, leaving the embedding table and `lm_head` in fp32 (blocked
by `FRAMEWORK_PROBLEMS.md` P3), while Meta's `q8da4w` additionally quantizes both of those
to 4-bit — a known, already-documented scope difference, not new engineering work.

A latent test bug was also found and fixed: the `custom_sdpa`-vs-`manual` parity test had
been passing against an accidentally all-zeros mask, so it never actually exercised causal
masking (details in finding 1 below).

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

## Second experiment: `add_batch_dim` — testing (and refuting) the batch-dimension theory

`FRAMEWORK_PROBLEMS.md`'s strongest *remaining* lead for the leftover 1.34x/2.09x
gap was that tsurgeon's graph runs everything unbatched `(seq, dim)` end to end,
unlike ExecuTorch's own `(1, seq, dim)` convention — hypothesizing that XNNPACK
might select faster GEMM/FC microkernels or weight-packing for batched shapes.
This was tested directly rather than left as a guess.

**Design, scoped to minimize risk.** A naive fix would thread a batch dim
through every module's shape contract. Instead: `LinearCompressed` already
flattens arbitrary leading dims internally and restores the caller's original
shape on output, and `RMSNorm`/`MLP`/`AtomicSum` all reduce only over the last
dim — so they're already shape-agnostic. That means a batch dim can be threaded
through the **residual stream only** (embedding → q/k/v/out_proj → mlp → next
layer) while leaving attention math, KV-cache read/write, and `custom_sdpa`
**completely untouched**: `MHACausal.forward` absorbs the batch dim immediately
after q/k/v projection (`.view(in_seq_len, heads, head_dim)` on a leading-1-dim
tensor is a no-op reshape — total element count is unchanged) and re-attaches
it only after `out_proj`. This tests the hypothesis for the bulk of the
XNNPACK-delegated Linear ops without touching cache/attention internals at all.

**Implementation:** new `add_batch_dim` option (default `False`), threaded
through `blocks/config.py` → `utils/convert.py` → `blocks/decoder.py` →
`blocks/mha.py`, exactly like the existing `attn_impl`/`cache_impl` options
(see AGENTS.md's "`add_batch_dim`" section for the full writeup). `export/common.py`'s
`LLMWrapper`/`build_example_inputs`/`calibrate_pt2e_observers` were updated to
build/consume a `(1, 1)`-shaped `input_ids` instead of `(1,)` when set, and to
keep the batch dim on the final logits (matching ExecuTorch's own `(1, vocab)`
output convention) instead of indexing it away.

**Opt-in and off by default specifically because of QNN.** This has not been
validated against QNN's or TensorRT's op converters — the extra leading dim may
not be supported by every converter on those backends. Because it's a
convert_options flag defaulting to `False`, every existing QNN/TensorRT/XNNPACK
export is provably unaffected (confirmed: `pytest test/unit test/e2e` — 40/40
unit incl. 5 new tests, 57/57 e2e — unaffected before/after).

**Correctness:** new `test/unit/test_add_batch_dim.py` (5 tests) checks
numeric parity against the default unbatched path for all three `attn_impl`
values, under per-kv-group pruned RoPE specifically (confirming `add_batch_dim`
doesn't interfere with `_project_rope`'s per-layer pruning projection — see
below), and at the full `TransformerDecoder` level. Re-exported both TinyLlama
variants with `add_batch_dim=True`: `fp32` `max_abs_err=6.0e-5` (tighter than
the unbatched export's `1.4e-4`), `w4` `max_abs_err=2.15` (in the same 4-bit
noise range as unbatched's `2.07`) — both consistent with a correct, working
export.

**Aside, since it came up while discussing this experiment:** `add_batch_dim`
only affects the *shared*, pruning-independent RoPE lookup
(`rope_cos[pos_id]`/`rope_sin[pos_id]`, identical for every layer regardless of
pruning) hoisted in the first fix above — it does not touch `_project_rope`,
which still runs once per layer using that layer's own `rope_freq_proj` buffer
and still produces different projected `cos`/`sin` per layer when different
layers keep different rotary frequencies under structured pruning. The new
pruned-RoPE parametrized test above is exactly what verifies this.

**Result: the hypothesis does not hold on this CPU/environment — batched is
slightly *slower*, not faster:**

| Variant | Unbatched decode (tok/s) | Batched decode (tok/s) | Speedup | stdev (unbatched / batched) |
|---|---:|---:|---:|---:|
| fp32 | 20.82 | 20.58 | **-1.1%** (0.989x) | 0.15 / 0.12 |
| w4 | 39.93 | 38.66 | **-3.2%** (0.968x) | 0.32 / 0.40 |

(Interleaved A/B methodology as above, 15 trials, 8 discarded warm-up
iterations each — low variance on both sides, so this is a real, reproducible
regression, not noise.) The extra `.view`/`.reshape` needed to re-attach the
batch dim after `out_proj` is a real (if tiny) cost with no offsetting
kernel-selection benefit observed — this specific CPU/XNNPACK build does not
appear to pick a faster microkernel or packing routine for batched vs.
unbatched Linear inputs at this scale. **The option is kept in the codebase
(fully tested, documented, opt-in, zero effect on default behavior) since the
answer could differ on other hardware (e.g. ARM NEON) or larger batch sizes,
but it should not be enabled for this model/CPU combination.**

This rules out the leading candidate explanation for the remaining gap.

## Third experiment (2026-09-10): `custom_sdpa` was scanning the entire KV cache every token — THE root cause

### How it was found

Two measurements pointed the same way. First, a fresh instruction-stream diff of both
`.pte` files (`deserialize_pte_binary`, counting `KernelCall`/`DelegateCall` per op)
contradicted the earlier "the graphs are identical" claim and showed tsurgeon was *leaner*
in the glue, not heavier:

| | Meta | tsurgeon |
|---|---:|---:|
| total instructions | 739 | 474 |
| XNNPACK delegate calls | 224 | 136 |
| `aten::max` (tsurgeon's extra RMSNorm stability step) | 0 | 45 |
| `llama::custom_sdpa` | 22 | 22 |

So the remaining cost could not be non-delegated glue op *count*. Second, converting the
gap to absolute time per token:

- fp32: 48.0 ms vs 36.9 ms → **11.1 ms/token** gap
- w4: 25.6 ms vs 12.4 ms → **13.2 ms/token** gap

The absolute gap is essentially the *same* in both precisions. That rules out the weight
GEMMs (which get ~2x faster under w4 and would shrink the gap proportionally) and points
at something precision-independent. `custom_sdpa` is exactly that: its cache is always
fp32 and, as it turned out, always full-length. A back-of-envelope FLOP count for scoring
32 heads × 128 slots × 64 dim across 22 layers lands right at ~11 ms — matching the gap.

### Root cause

`blocks/mha.py::_forward_custom_sdpa` called:

```python
torch.ops.llama.custom_sdpa(
    q_b, key_cache, value_cache,
    0,             # start_pos
    attn_mask_2d,  # explicit (1, max_cache_len) mask
    0.0, False,    # is_causal=False
    scale,
)
```

With `start_pos=0` the kernel has no way to know which cache slots are live, so it scores
the query against **all `max_cache_len` slots** and then masks the invalid ones away
afterwards — O(`max_cache_len`) per layer per token, no matter how far into the sequence
we actually are. At decode step 5 of a 128-slot cache that is ~25x more attention work
than needed.

ExecuTorch's own exporter does not do this. Its default path
(`SDPACustom.forward`, `use_attention_mask=False`, in
`examples/models/llama/source_transformation/sdpa.py`) passes the **real** `start_pos`
with `is_causal=True` and **no** mask, so the kernel scans only
`[0, start_pos + seqlen)` — O(pos).

### Fix

```python
torch.ops.llama.custom_sdpa(
    q_b, key_cache, value_cache,
    start_pos,   # real position -- bounds the kernel's scan
    None,        # no mask; derived from is_causal
    0.0, True,   # is_causal=True
    scale,
)
```

Semantically identical: the mask `TransformerDecoder` builds is plain causal
(mask out `k_pos > q_pos`), which is precisely what `is_causal=True` generates. The only
numeric difference is a true `-inf`/skip rather than a finite `-10000.0` penalty, i.e. the
kernel result is marginally *more* exact. Verified directly by calling the op both ways
against a hand-written reference — both agree to 1.2e-7.

`attn_mask` is now intentionally unused on this path; it is still built for, and used by,
the `"manual"` and `"sdpa"` kernels.

### Latent test bug this uncovered

The change initially failed `test_causal_custom_sdpa_matches_manual` with **100% of
elements** differing. The fix was correct; the test was wrong:

```python
q_pos = k_pos = torch.arange(16)                     # both 1-D!
attn_mask = torch.where((q_pos < k_pos), penalty, zeros)[pid].unsqueeze(0)
```

Two *identical 1-D* tensors compare elementwise to **all-False**, so the mask was silently
**all zeros** — no masking at all. Both the `manual` and old `custom_sdpa` paths were
attending over every slot including empty ones, so they agreed by being equally wrong. The
real `TransformerDecoder` uses `arange(L)[:, None]` / `arange(L)[None, :]`, which broadcast
into a genuine `(L, L)` causal matrix.

Fixed in all three affected files (`test_mha_causal_custom_sdpa.py`,
`test_gqa_rope_pruning.py`, `test_add_batch_dim.py`) to build the mask the way the decoder
does. The parity test now passes against a *real* causal mask, which makes it a strictly
stronger check than before.

### Correctness

- `pytest test/unit` — **40/40 pass**, including the now-genuinely-causal parity test.
- `pytest test/e2e` — **57/57 pass, 4 skipped** (TensorRT/QNN unavailable, pre-existing),
  covering `test_export_xnnpack` and `test_export_xnnpack_custom_sdpa`.
- Export-time inference stats on the final build: fp32 `max_abs_err=8.0e-5`
  (vs `1.4e-4` before the change — slightly *tighter*, consistent with `-inf` vs
  `-10000.0` masking), w4 `max_abs_err=1.34` (within the 4-bit noise band).
- The op itself was probed in isolation both ways (`start_pos=0`+mask vs
  `start_pos=p`+`is_causal`) against a hand-written reference: **both agree to 1.2e-7**.

### Results

| Variant | Before (fix 1 only) | After | Speedup |
|---|---:|---:|---:|
| fp32 | 19.98 | **30.14** | **1.51x** |
| w4 | 38.56 | **90.48** | **2.35x** |

Head-to-head against Meta's own `.pte`, same process, interleaved:

| Variant | Meta | tsurgeon | Ratio |
|---|---:|---:|---:|
| fp32 | 30.31 (stdev 0.39) | **34.52** (stdev 0.65) | **1.14x faster** |
| quantized | 87.73 (stdev 0.75) | **86.19** (stdev 1.69) | 0.98x (parity) |

Caveat on the quantized row, unchanged from `TSURGEON_VS_EXECUTORCH.md`: the recipes are
not mechanism-identical. Meta's `q8da4w` is 4-bit weights **+ dynamic 8-bit activations +
4-bit embedding**; tsurgeon's `w4` is weight-only INT4 over attn+mlp with the embedding and
`lm_head` left in fp32 (blocked by `FRAMEWORK_PROBLEMS.md` P3). The fp32-vs-fp32 row is the
directly comparable one.

### Independent re-verification (same day, later session)

Re-ran both head-to-head comparisons from a fresh session to confirm the result holds up,
not just a one-off measurement — 20 interleaved trials each, same methodology:

| Variant | Meta | tsurgeon | Ratio |
|---|---:|---:|---:|
| fp32 | 33.01 (stdev 2.16) | **34.12** (stdev 1.78) | **1.03x faster** |
| quantized | 90.85 (stdev 1.83) | 85.50 (stdev 2.99) | 0.94x |

Same qualitative conclusion (fp32 at/above parity, quantized within single digits of
parity), but the precise ratios shifted a bit from the first run's 1.14x/0.98x — notably,
**Meta's own fp32 number itself varied across sessions** (27.1 in the original
`TSURGEON_VS_EXECUTORCH.md` → 30.3 in the first run above → 33.0 in this one), all
measured from the identical, unmodified `.pte` file. That variance is larger than
anything tsurgeon-side changes could produce and confirms what the "Benchmark methodology"
section above already flagged: this is a shared, variable-load machine, and Meta's export
is exactly as exposed to that noise as tsurgeon's. Treat "roughly at parity, sometimes
slightly ahead, sometimes slightly behind, depending on the moment measured" as the honest
takeaway rather than either single run's precise ratio.

## Rejected: precomputing the causal mask as a constant buffer

Also tried hoisting the mask one step further — building the whole `(max_cache_len,
max_cache_len)` matrix once in `__init__` as a buffer and indexing it in `forward()`, the
way Meta bakes in `torch.tril(...)`. Measured **0.95x (slower)** on w4, and it adds an
`(L, L)` fp32 constant to every `.pte` — 64 KB at the benchmark's `max_seq_len=128`, but
**16 MB** at the framework default `max_cache_len=2048`. No benefit plus a real size cost,
so this was reverted rather than committed. (After the `is_causal` fix the mask is not even
consumed on the `custom_sdpa` path, so there was nothing left to win.)

## What this does *not* fix

`FRAMEWORK_PROBLEMS.md`'s other findings are untouched and still open:
- **P2** — batched/dynamic-shape prefill and multi-token KV-cache writes are
  still unsupported (prefill is still a token-by-token loop).
- **P3** — embedding-table and `lm_head` hard quantization export bugs are
  still open, so `w4` here still leaves those two layers in fp32 (unlike
  ExecuTorch's `xnnpack_q8da4w`, which quantizes both).

Since tsurgeon is now at/above Meta's throughput on this model+CPU, there is no
longer a "gap" to explain. Remaining known-but-unexploited headroom, if someone
wants to push further:
- `RMSNorm` (`blocks/norm.py`) does a non-standard extra
  max-abs-normalize-then-clamp step before the usual variance/rsqrt (this is a
  deliberate numerical-stability choice — commit `4f73649`, "Improved tsurgeon
  graph numerical stability" — not an accident, so do not just delete it). It
  shows up as 45 non-delegated `aten::max` calls per token that Meta's graph
  does not have. Isolated micro-benchmark put it at only ~375 µs/token
  (<1% of a ~36 ms token), so it was left alone; it would only be worth
  revisiting as an opt-in "fast norm" if that 1% matters.
- Prefill is still token-by-token (`FRAMEWORK_PROBLEMS.md` P2). This does not
  affect the decode numbers above, but it is the largest remaining
  *feature* gap and would dominate any long-prompt workload.
- ETDump per-op profiling (needs an ExecuTorch build with
  `EXECUTORCH_ENABLE_EVENT_TRACER`) would be the way to look inside the XNNPACK
  delegate blobs themselves; still unavailable in this environment, but no
  longer blocking, since the previously-unexplained gap turned out to be in the
  `custom_sdpa` call arguments rather than inside the delegate.

Two more candidates were checked (by reading the installed ExecuTorch source directly, no
ETDump needed) and ruled out as differentiators, closing off the "what else could it be"
list from the independent re-verification pass:
- **Thread-pool sizing** isn't an exporter-side setting at all — both `.pte` files run
  through the same `_load_for_executorch` runtime in the same process/environment, so
  they share whatever default thread pool ExecuTorch's C++ runtime picks. Nothing for
  either export to configure differently.
- **XNNPACK partitioner config**: Meta's `--xnnpack-extended-ops` flag
  (`export_llama_lib.py::_to_edge_and_lower_llama_xnnpack`) only controls whether a
  second, greedy `XnnpackPartitioner()` runs *in addition to* a dynamic-quant-only
  partitioner — needed on Meta's side because without it, an unquantized fp32 model
  would get zero delegation (the dynamic-quant-only partitioner has nothing to claim).
  tsurgeon's exporter (`xnnpack_export.py`) already calls the full greedy
  `XnnpackPartitioner()` directly and unconditionally — functionally equivalent, not a gap.

## Files changed

- `transformersurgeon/blocks/mha.py` — **`custom_sdpa` `is_causal`+`start_pos` fix (the big one)**; `MHACausal.forward` consumes precomputed mask/RoPE; `add_batch_dim` support.
- `transformersurgeon/blocks/decoder.py` — hoist mask/RoPE-index computation; `add_batch_dim` plumbing.
- `transformersurgeon/blocks/config.py`, `transformersurgeon/utils/convert.py` — `add_batch_dim` config option.
- `transformersurgeon/export/common.py` — `LLMWrapper`/`build_example_inputs`/`calibrate_pt2e_observers` batch-dim awareness.
- `test/unit/test_mha_causal_custom_sdpa.py`, `test/unit/test_gqa_rope_pruning.py` — updated calling convention; **fixed the degenerate all-zeros causal mask**.
- `test/unit/test_add_batch_dim.py` — new, `add_batch_dim` correctness tests.
- `AGENTS.md` — documents the new `add_batch_dim` option and its QNN/TensorRT caveat.

## Commits on `debug-xnnpackspeed`

| Commit | What |
|---|---|
| `2d8afad` | Hoist per-layer mask/RoPE computation to once-per-token (fp32 +7%, w4 +16%) |
| `25712fa` | Add opt-in `add_batch_dim`; test and **refute** the batch-dim theory |
| `3242f6c` | **`custom_sdpa` `is_causal`+`start_pos`** (fp32 1.51x, w4 2.35x) + causal-mask test fix |

## Reproduce

```bash
conda activate py312_executorch_xnnpack
cd /workspace/transformer-surgeon && pytest test/unit test/e2e -q   # correctness

cd /workspace/executorch_llama
python tsurgeon_export_llama.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --variant fp32 --out-dir pte_final
python tsurgeon_export_llama.py --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --variant w4   --out-dir pte_final

# the headline head-to-head vs Meta's own export
python interleaved_ab.py pte/llama2_xnnpack_fp32.pte   pte_final/tsurgeon_llama_fp32.pte 12
python interleaved_ab.py pte/llama2_xnnpack_q8da4w.pte pte_final/tsurgeon_llama_w4.pte   12

# interleaved A/B (script written for this investigation, kept alongside the
# other benchmark scripts in this sibling dir for provenance -- see FRAMEWORK_PROBLEMS.md's
# convention of keeping repro scripts in /workspace/executorch_llama/, outside this repo)
python interleaved_ab.py pte/tsurgeon_llama_fp32.pte pte_fixed/tsurgeon_llama_fp32.pte 15
python interleaved_ab.py pte/tsurgeon_llama_w4.pte   pte_fixed/tsurgeon_llama_w4.pte   15
python interleaved_ab.py pte_fixed/tsurgeon_llama_fp32.pte pte_fixed/tsurgeon_llama_fp32_batched.pte 15
python interleaved_ab.py pte_fixed/tsurgeon_llama_w4.pte   pte_fixed/tsurgeon_llama_w4_batched.pte   15
```
