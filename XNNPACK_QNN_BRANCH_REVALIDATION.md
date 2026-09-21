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

## One thing that did move, and is not explained

Export-time numerical self-check is looser on this branch:

| | `debug-xnnpackspeed` | `debug-qnnspeed` |
|---|---|---|
| fp32 `max_abs_err` | 8.0e-5 | 2.16e-4 |
| w4 `max_abs_err` | 1.34 | 2.21 |

Both still export with `mismatch_count: 0`, both pass the full test suite, and
the fp32 figure is small in absolute terms. But it moved in the same direction
at both precisions, which argues against it being noise. The `_head` change is
the obvious suspect, since it is the one unconditional graph edit. This has not
been run down and is worth a look before these numbers are quoted as a
correctness baseline.
