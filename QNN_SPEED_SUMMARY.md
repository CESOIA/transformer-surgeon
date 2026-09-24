# Making the QNN export look like Qualcomm's own

*A plain-language companion to `QNN_DECODE_SPEED_FIX.md` (the detailed engineering
log). Same investigation, same result, with every non-obvious term explained.*

## The one-sentence result

The exported model was shuffling **537 MB of data to and from main memory per word
generated**, purely because two operations were in the wrong order — reordering them
(a change already written in the codebase, but commented out) removed all of it, cut
memory writes by 74% and reads by 21%, and produced bit-comparable output.

---

## Background: who is who

- **transformer-surgeon** — this repo. Takes a HuggingFace language model, optionally
  compresses it, exports it to run on a device.
- **QNN / HTP** — Qualcomm's toolkit for their phone chips. **HTP** is the actual NPU
  (the AI accelerator) inside a Snapdragon.
- **The vendor reference** — Qualcomm ships its own hand-tuned example of exactly this
  kind of model, in `executorch/examples/qualcomm/oss_scripts/llama/`. Written by the
  people who built the chip, so it's the gold standard to compare against.
- **VTCM** — a small, very fast scratchpad memory on the NPU. Roughly: the NPU's
  workbench. Anything that fits on the workbench is fast. Anything too big has to keep
  being carried back and forth from the warehouse (main memory / **DDR**), which is slow.

## The constraint

This is an x86 Linux container. There is no phone here, so **the model cannot be run**.
Every number below comes from compiling the model, not executing it.

That turns out to be fine, because Qualcomm's compiler reports how much memory traffic
the model will generate — including a `spill` figure that means exactly *"this didn't
fit on the workbench and had to go back to the warehouse."* For generating text, which
is bottlenecked by memory rather than arithmetic, that figure is the thing that
predicts speed.

---

## What was wrong

To generate one word, the model looks back over everything it has seen so far. That
history is the **KV cache**. Modern models share one cache entry across several
"attention heads" to save memory — so before the maths, the cache has to be lined up
with the heads that use it.

Two steps are needed: **duplicate** the cache for each head that shares it (7× here),
and **rotate** it into the orientation the matrix multiply expects.

The code did them in this order:

```
duplicate (cache becomes 7× bigger)  →  rotate the now-7×-bigger thing
```

Rotating a 7.3 MB tensor does not fit on the workbench. So for every layer, for every
word, the NPU carried it out to main memory and back.

Doing the same two steps the other way round —

```
rotate the small original  →  duplicate
```

— is the same result, and the rotation now happens on something 7× smaller. Better
still, Qualcomm's matrix multiply can do the duplication implicitly, so the duplicate
step disappears entirely.

**The punchline:** this exact alternative was already sitting in `blocks/mha.py`,
written out, labelled `ALTERNATIVE 2 — broadcasting GQA`, and commented out. It just
needed switching on.

Measured for a single attention operation:

| | main-memory traffic |
|---|---|
| before (duplicate, then rotate) | 9.2 MB in, 9.2 MB out, **plus 18.4 MB of spill** |
| after (rotate, then duplicate) | 0.04 MB in, 0.06 MB out, **no spill** |

---

## The results on a real model

Qwen2-0.5B, targeting a Snapdragon SM8850, same settings before and after:

| per word generated | before | after |
|---|---:|---:|
| carried out to main memory because it didn't fit | 267.6 MB | **0** |
| carried back in | 269.6 MB | **0** |
| total written | 360.2 MB | **92.6 MB** |
| total read | 1294.1 MB | **1024.2 MB** |

The 1024 MB still being read is just the model's own weights — every parameter has to
be read once per word, and there is no way around that short of compressing them. In
other words the export is now at the floor for its precision; before, it was doing
~1.5× more work than necessary.

The model also got simpler: **1676 → 1384 operations**, a 17% reduction, and it still
compiles into a single block that runs entirely on the NPU with nothing falling back
to the CPU.

**Correctness check:** generating 8 words with the real model, old code vs new, the
difference in the output scores is ~0.000002 — ordinary floating-point rounding — and
**the model picks the identical word every single time**.

---

## Three smaller things fixed along the way

**The sequence-length setting did nothing.** `--max-sequence-length` was documented as
controlling the cache size and silently didn't — every export shipped a 2048-slot
cache regardless. The tell was that asking for 128 and asking for 1024 produced
byte-identical compiler output. Now wired up; with it working, memory writes drop
further to 12.9 MB per word.

**The normalisation step was 6 operations instead of 1.** Each of the 49 normalisation
layers does a safety rescale first, which stops ExecuTorch recognising the pattern and
folding it into one operation. That's ~245 extra NPU dispatches per word. The rescale
is a genuine safeguard against numerical overflow in half-precision, so it is now a
**switch** (`--no-rmsnorm-prescale`), off-by-request rather than removed.

**Two ExecuTorch bugs, now caught early.** Qualcomm's examples all convert their
linear layers to 1×1 convolutions, which suits the NPU better. That conversion is
available in ExecuTorch — but in version 1.3.1 it breaks on half-precision weights,
and breaks again on models that share weights between the input and output layers
(very common). Both now produce a clear warning up front instead of an error 200
lines deep that mentions neither linear layers nor precision. The feature is wired up
but left off by default, because it showed no measurable gain here and a third,
unexplained failure mode remains.

---

## A correction: quantization

The measurements above all use 16-bit floating point on purpose, so the numbers cleanly
show just the effect of this fix. **Quantization (4-bit / 8-bit) is not a gap** — the
framework already supports quantizing any layer, attention included, not only the MLP;
the CLI script's `--quant-mlp` flag is just a convenience shortcut, not a ceiling. If
quantization is already applied in your workflow, the DDR numbers above would be
smaller still on top of it; that comparison wasn't run here.

## What's left, and what's just a TODO

1. **TODO — a prefill path.** Right now the model reads your prompt one word at a
   time. Qualcomm's version exports a second graph that reads 128 at a time, sharing
   the same weights. Long prompts are unnecessarily slow. Not attempted in this
   branch, by design — noted for later.
2. **Cache writes are a different design, not a missing optimization.** Every word
   currently does 48 scattered writes into the cache, computed on the NPU. Qualcomm's
   graph does **no scatter at all** — it hands back only the newest word's cache
   entry as a small extra output, and its C++ program copies that into place for the
   next call by moving a pointer forward (a plain memory copy, off the NPU entirely).
   Getting that requires the model to change what it hands back, and requires a C++
   program on the device that knows to do the copy — an export-side change alone
   can't get there, and this repo's export doesn't yet produce or expect that shape
   of output either way.

The full technical detail, including what was already correct and needed no change,
is in `QNN_DECODE_SPEED_FIX.md`.
