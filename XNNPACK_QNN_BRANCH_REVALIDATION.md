# Does the QNN fix disturb the XNNPACK result?

Short answer: **no.** The quantized case actually improved slightly, into parity
with Meta's export.

`debug-qnnspeed` fast-forwards cleanly on top of `debug-xnnpackspeed` — all 11
XNNPACK commits are intact, nothing was rebased or dropped — and adds exactly
one commit, `a709790` ("Fix GQA expand-before-transpose bug causing QNN/HTP
cache spill"). That commit touches `blocks/mha.py`, `blocks/decoder.py`,
`blocks/norm.py` and `export/common.py`, all of which are on the XNNPACK decode
path too, so the overlap needed checking rather than assuming.

## What actually overlaps

| change | reaches XNNPACK decode? | why |
|---|---|---|
| GQA broadcast rewrite in `attention()` | **no** | `attention()` is called only from the `attn_impl="manual"` branch of `MHACausal.forward` (`mha.py:766`). The benchmarked config uses `custom_sdpa`, whose `_forward_custom_sdpa` (`mha.py:627`) never calls it. |
| `max_seq_len` now sizes the KV cache | **no** | The new fallback in `export/export.py` fires only when `max_cache_len` is absent from `convert_options`. Both the benchmark script and `xnnpack-models` pass it explicitly (128). |
| `rmsnorm_prescale` | **not by default** | Defaults to `True`, i.e. the previous behaviour. Now an available lever — the prescale was measured at ~375 µs/token, under 1% — but nothing changes unless it is set. |
| `_head` rank-2 slice in `export/common.py` | **yes** | `hidden[-1, :]` became `hidden[-1:, :]` + `[0]`, changing the exported graph on every backend, XNNPACK included. This is the only unconditional change. |

So the expectation from reading the code was "no material XNNPACK impact", and
that is what measurement showed.

## Measured

Interleaved A/B (`executorch_llama/interleaved_ab.py`, 8 discarded warmup
trials + 12 timed, both models alternating inside one process), TinyLlama-1.1B,
128-slot cache, decode-only:

| | `debug-xnnpackspeed` (2 runs) | `debug-qnnspeed` |
|---|---|---|
| fp32 vs Meta | 1.139x / 1.03x | **1.099x** (31.98 → 35.14 tok/s median) |
| quantized vs Meta | 0.979x / 0.94x | **1.005x** (90.62 → 91.06 tok/s median) |

fp32 stays in the same band across all three runs. The quantized case moved from
slightly behind to parity — plausibly the `_head` change, but the shift is close
enough to run-to-run spread on a shared machine that it should not be claimed as
a win from any specific edit.

Tests on `debug-qnnspeed`: **unit 40/40 pass**, **e2e 57 passed / 4 skipped**,
the same four pre-existing TensorRT/QNN-SDK/dual-tower-VL skips as before.

## The export-time self-check is run-to-run noise

An earlier draft of this document flagged the export-time numerical self-check
as having loosened on this branch, and named the `_head` change as the likely
cause. **That was wrong**, and the reasoning behind it was wrong too.

A second export of the same model on the same commit, with the same config,
settles it:

| | `debug-xnnpackspeed` | `debug-qnnspeed` run 1 | `debug-qnnspeed` run 2 |
|---|---|---|---|
| fp32 `max_abs_err` | 8.0e-5 | 2.16e-4 | 2.38e-5 |
| w4 `max_abs_err` | 1.34 | 2.21 | 1.34 |

Run 2 is *tighter* than the old branch at fp32 and identical to it at w4. Two
runs of identical code differ by roughly 10x at fp32, so this statistic is
sampling-dependent and cannot resolve a difference of this size.

The original inference — "it moved in the same direction at both precisions,
which argues against noise" — does not hold with one sample per branch. Two
measurements moving together is exactly what uncorrelated noise looks like half
the time. The `_head` change is not implicated.

Practical consequence: **do not use `inference_stats.max_abs_err` from a single
export to compare two builds.** It is a smoke test for "did this export come out
catastrophically wrong", not a regression metric. Comparing builds needs either
a fixed input set or several repeats per build.
