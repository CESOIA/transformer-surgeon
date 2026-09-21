# tsurgeon vs. Meta's ExecuTorch export: how we closed the speed gap

*A plain-language companion to `XNNPACK_DECODE_SPEED_FIX.md` (the detailed
engineering log). This version explains the same investigation and result,
but defines every non-obvious term along the way.*

## The one-sentence result

We started with transformer-surgeon's on-device export running **1.4x to
2.4x slower** than Meta's own official export of the same model on the same
hardware, and ended with it running **at parity** — by fixing one specific,
high-impact bug in how the model reads its own attention cache.

> **Corrected 2026-09-21.** This page previously said "at parity or slightly
> faster". Repeating the comparison five times instead of once showed the
> "slightly faster" readings were luck: fp32 lands at **1.02x** (parity) and
> quantized at **0.95x** (about 5% *behind* Meta). The fix was still a large
> genuine win — it closed a gap that started at 1.4–2.4x slower — but tsurgeon
> matches Meta's export rather than beating it. Details and the reason single
> runs mislead here: `../xnnpack-models/AGENTS.md` §4.

---

## Background: what are we even comparing?

- **transformer-surgeon ("tsurgeon")** — this repo. It takes a HuggingFace
  language model, optionally compresses it (pruning, quantization, low-rank
  decomposition), and exports it to run on-device (phones, embedded chips)
  via a backend like XNNPACK.
- **ExecuTorch** — Meta's own runtime/toolchain for running PyTorch models
  on-device. It ships its own official exporter for Llama-family models,
  which we treat as the reference/gold-standard implementation to compare
  against, since it's written and tuned by the people who built the runtime.
- **XNNPACK** — a CPU-optimized neural-network operator library (matrix
  multiplies, convolutions, etc.) that ExecuTorch can "delegate" parts of a
  model's computation to for speed, instead of running everything through
  its own generic interpreter. A "delegate call" hands a chunk of the
  computation graph to XNNPACK's fast, hand-tuned kernels; anything *not*
  delegated runs through ExecuTorch's slower, generic ("portable") fallback
  interpreter, one operation at a time.
- **`.pte` file** — the exported, ready-to-run model artifact that
  ExecuTorch's runtime loads and executes.

We used **TinyLlama-1.1B** (a small real Llama-architecture model, ~1.1
billion parameters) as the test case, exported through both tsurgeon and
Meta's own exporter, and benchmarked them head-to-head on the same CPU.

## Background: how does text generation work, and what are we measuring?

A language model generates text one word-piece ("token") at a time:

- **Prefill** — feeding the initial prompt through the model. This can be
  done for many tokens at once (fast) if the exporter supports it.
- **Decode** — generating each *new* token one at a time, where each new
  token depends on everything generated before it. This is inherently
  sequential (you can't generate token 5 before token 4 exists), so it's
  usually the speed bottleneck for chat/generation use cases. **All the
  numbers in this report are decode speed**, measured in **tok/s (tokens
  generated per second)** — higher is better.
- **KV cache** ("key/value cache") — to avoid recomputing attention over
  the entire conversation from scratch on every new token, the model
  stores intermediate "key" and "value" tensors for every position it has
  already processed, and each new token's attention step only needs to
  look up against that stored cache. `max_cache_len` (128 in our tests) is
  how many positions of cache the model reserves room for.

## The starting problem

An earlier investigation (`FRAMEWORK_PROBLEMS.md`) had already fixed one bug
and found that tsurgeon's export was still **1.44x slower in full precision
("fp32", i.e. no compression) and 2.41x slower with 4-bit weight
quantization ("w4", a compression technique that shrinks each weight from a
32-bit number to a 4-bit number to save memory and, often, time)** compared
to Meta's own export. *Why* was left an open question — profiling tools
that could see inside XNNPACK's delegate calls (`ETDump`) weren't available
in this environment, so the cause had to be found by reasoning about the
code directly instead.

## What we found and fixed, in order of how much each one mattered

### 1. The big one: the attention kernel was scanning the *entire* cache on every single token, instead of only the part that mattered

This was the fix that closed almost the entire gap.

**The setup:** tsurgeon's fastest attention path uses `custom_sdpa`
("scaled dot-product attention", the actual attention math — the step
where a token figures out which earlier tokens are relevant to it), a
fused, hand-optimized operation that ExecuTorch itself provides. Calling
it correctly requires telling it two things:

- `start_pos` — which position in the KV cache we're currently at (i.e.,
  how many tokens have been generated so far).
- `is_causal` — a flag meaning "a token may only attend to itself and
  earlier tokens, never later ones" (the standard rule for generating text
  left-to-right — "causal" as in cause-and-effect, earlier tokens can
  influence later ones but not vice versa).

**The bug:** tsurgeon's code was calling this operation with `start_pos=0`
and `is_causal=False`, and instead manually supplying an explicit **mask**
(a same-size array marking which cache positions are "real" vs. "empty/not
yet written") covering the *entire* `max_cache_len`-sized cache (128
slots), every single time. Because the operation was told `start_pos=0`
("we're at the very beginning") no matter which token we were actually
generating, it had no way to know which of those 128 slots actually held
real data — so on *every* decode step, regardless of whether we were
generating token #2 or token #100, it scored the current token against
**all 128 cache slots** and only afterward used the mask to throw away the
scores for slots that weren't real yet. That's wasted work that grows
directly with `max_cache_len`, regardless of how far into the conversation
you actually are.

**The fix:** pass the real `start_pos` and set `is_causal=True`, with no
manual mask. This tells the operation "just look at positions 0 through
`start_pos`, causally" — so at decode step 5, it does roughly 5x less
attention work instead of 128x, and the amount of work grows naturally as
the conversation gets longer, instead of always doing the maximum possible
amount. This is exactly what Meta's own exporter does by default — we
found the matching code in ExecuTorch's own source and confirmed it uses
the identical pattern.

**Impact:** roughly **1.5x faster in fp32, 2.35x faster in the quantized
(w4) variant** — this single fix accounts for almost the entire
improvement.

**A side discovery while fixing this:** one of the existing correctness
tests for this code path had a subtle bug of its own — it built its causal
mask by comparing two *identical* 1-dimensional lists of positions, which
(due to how array comparison/broadcasting works) silently produced an
all-zeros mask, i.e. no masking at all. The test had been passing for the
wrong reason: it was comparing two *equally broken* code paths against
each other, not verifying either one was actually correct. Fixed alongside
the main change.

### 2. A smaller fix: repeating identical work 22 times per token instead of once

**The setup:** TinyLlama has 22 transformer layers stacked on top of each
other; every generated token passes through all 22 in sequence.

**The bug:** two small pieces of setup work — building the causal mask
(see above) and looking up the current position's **RoPE** values
("Rotary Position Embedding," a technique for encoding *where* in the
sequence a token is, baked directly into its numbers rather than added as
a separate input) — were being recomputed independently inside *each* of
the 22 layers, even though the inputs to that computation never differ
between layers. It's the same math producing the exact same answer, done
22 times over instead of once.

**Why it mattered more than the raw amount of math suggests:** these
particular operations (comparing numbers, selecting values) aren't the
kind XNNPACK's fast delegate path handles — they fall back to ExecuTorch's
slower generic interpreter, which has real per-operation overhead. Doing
that 22 times instead of once, on every single generated token, adds up.

**The fix:** compute both values once per token, at the top level, and
pass the already-computed result down into each of the 22 layers instead
of having every layer redo the work. (Each layer still applies its own
extra, layer-specific step on top for models that use *structured
pruning* — a compression technique that can remove different amounts of
detail from different layers — since that part genuinely does differ per
layer; only the *shared, identical-across-layers* part was hoisted out.)

**Impact:** roughly **+7% in fp32, +16% in the w4 variant**.

### 3. A tested idea that *didn't* help: giving tensors an explicit "batch" dimension

**The idea:** Meta's own export represents its data with an extra leading
"batch" dimension (`(1, sequence_length, ...)`, where the `1` means "batch
size of one" — a placeholder dimension for cases where you might process
multiple sequences at once, even though here there's always just one),
while tsurgeon's export didn't carry that extra dimension. The hypothesis
was that XNNPACK's fast kernels might be specifically tuned for
batch-shaped inputs and pick slower code paths without it.

**What we did:** implemented this as an optional, off-by-default setting
and tested it directly rather than guessing.

**Result:** it made things **slightly slower** (about 1-3%), not faster —
so the hypothesis was wrong, at least on this CPU. We kept the option in
the code (fully tested, harmless, and switched off by default) in case it
behaves differently on other hardware, but it's not something to turn on
here.

## The result, measured directly against Meta's own export

We ran both tsurgeon's export and Meta's official export side-by-side, in
the same process, alternating between them trial-by-trial (to make sure
both experienced the exact same momentary conditions on this shared
machine, rather than one running while the machine happened to be busier).

| Variant | Meta's own speed | tsurgeon's speed | Result |
|---|---:|---:|---:|
| Full precision (fp32) | ~33 tok/s | ~34 tok/s | tsurgeon slightly **faster** |
| 4-bit quantized | ~91 tok/s | ~86 tok/s | tsurgeon **within ~6%** |

Starting point, for comparison: tsurgeon was **1.44x slower** (fp32) and
**2.41x slower** (quantized).

**Why is the quantized number not also at 100%+?** Not a mystery — it's a
known, pre-existing scope difference, not something newly discovered here.
Meta's quantized recipe shrinks the embedding table (the lookup table
converting words into numbers) and the final output layer down to 4 bits
too; tsurgeon's quantization doesn't support that yet for those two
specific parts of the model (a separate, already-documented limitation),
so those two pieces stay at full size in tsurgeon's version. Everything
that *is* quantized in both is quantized the same way.

**A note on trusting these numbers:** this benchmark ran on a shared
machine with fluctuating background load, and even *Meta's own,
completely unmodified* export measured meaningfully different speeds
across different measurement sessions (27, then 30, then 33 tok/s, same
file, same code, different moments). So treat "roughly at parity, give or
take a few percent depending on the moment measured" as the honest
takeaway, rather than fixating on any single decimal point.

## What's left (not fixed here, and not quick fixes)

- **Prefill is still slow.** Feeding in the initial prompt is still done
  one token at a time instead of all at once (Meta's export processes the
  whole prompt in a single batched step). This doesn't affect the decode
  numbers above, but it matters for long prompts. This is a bigger,
  separate piece of work (tracked as "P2" in `FRAMEWORK_PROBLEMS.md`).
- **Quantization doesn't yet cover the embedding table / output layer**
  (mentioned above, tracked as "P3"). Closing this would likely close most
  of the remaining ~6% quantized-mode gap.
- A couple of smaller, already-ruled-out ideas (an unusual extra
  numerical-stability step in the normalization code, CPU thread-pool
  settings, and XNNPACK's internal configuration options) were checked and
  confirmed *not* to be worth pursuing further — see the detailed report
  for specifics.

## Where to look for more detail

- `XNNPACK_DECODE_SPEED_FIX.md` — the full engineering log: exact code
  diffs, every benchmark run's raw numbers, and the reasoning trail for
  each fix and rejected idea.
- `FRAMEWORK_PROBLEMS.md` — the original investigation that first
  identified the speed gap and the still-open P2/P3 items above.
- Branch: `debug-xnnpackspeed`, based off `draft-attnkernel`.
