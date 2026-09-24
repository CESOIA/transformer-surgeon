# transformer-surgeon — Framework Problems Report

Date: 2026-07-28
Method: comparing `export_to_backend(xnnpack)`'s output against ExecuTorch's own
official `examples/models/llama/export_llama` for the *same* checkpoint
(`TinyLlama/TinyLlama-1.1B-Chat-v1.0` — real trained weights, Llama2 architecture,
GQA), on CPU, to answer "is tsurgeon's XNNPACK export state-of-the-art vs.
ExecuTorch's own." Every finding below was reproduced and traced to a specific
file/line, not inferred from symptoms alone (P3b is the one exception — flagged
explicitly). Reproduction lives in a sibling working directory,
`/workspace/executorch_llama/` (`tsurgeon_export_llama.py`, `benchmark_tsurgeon.py`,
`export.py`, `benchmark.py`, `convert_hf_llama.py`, `TSURGEON_VS_EXECUTORCH.md`) —
outside this repo, kept here only for provenance/reproduce commands.

This file follows the same convention as the historical `FRAMEWORK_PROBLEMS.md`
(removed in `a577f0e` once P1–P6 were resolved — see `git log -- FRAMEWORK_PROBLEMS.md`).
Numbering restarts at P1 since the old set is gone.

## Environment used

| Component | Value |
|---|---|
| Python | 3.12 (conda env `py312_executorch_xnnpack`) |
| torch | 2.12.0+cu132 |
| executorch | 1.3.1 (pip, prebuilt — **not** built with `EXECUTORCH_ENABLE_EVENT_TRACER`, so ETDump/per-op profiling is unavailable in this env; see P2/P1 notes) |
| torchao | 0.17.0 |
| transformers | 5.12.0 |
| transformer-surgeon | 0.7.3 (editable install) |
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (dim=2048, 22 layers, 32 heads / 4 KV heads, vocab=32000) |
| Backend | XNNPACK, CPU |

---

## Severity summary

| # | Severity | Area | One-liner |
|---|---|---|---|
| P1 | Medium | `export/executorch_exporters/xnnpack` | **Fixed.** Exported `.pte` inputs were memory-planned (copied into an internal buffer every `forward()` call) because `to_executorch()` was called with no config; ExecuTorch's own llama exporter explicitly disables this. Correctness/parity fix — re-benchmarking measured negligible throughput impact, so it does **not** explain the decode-speed gap vs. ExecuTorch's own export (see full section). |
| P2 | Medium | `blocks/mha.py` (KV cache) | Multi-token (`in_seq_len>1`) cache writes are unsupported or broken in most code paths, blocking batched/dynamic-shape prefill. **Only matters when a forward() call must process more than one new token at once** (prompt prefill, speculative decoding) — plain one-token-at-a-time decode is unaffected. Not fixed — future work. |
| P3 | Medium | `compression/quantization.py` + `export/common.py` | Two independent bugs prevent quantizing an `EmbeddingCompressed` table or the model's `lm_head`/`final_layer` through the XNNPACK export path. Not fixed — future work. |

---

## P1 — XNNPACK export doesn't disable input memory-planning — FIXED

**Symptom:** benchmarking the same checkpoint through both exporters (fp32,
XNNPACK-delegated, otherwise identical settings) showed tsurgeon's decode
throughput 1.5–2.6x slower than ExecuTorch's own llama export (17.9–31.3 tok/s
vs. 27.1–80.4 tok/s on TinyLlama-1.1B, CPU, greedy decode).

**Confirmed root cause:** `transformersurgeon/export/executorch_exporters/xnnpack/xnnpack_export.py:78`:

```python
et_program = edge.to_executorch()
```

called with **no config argument**, so it falls back to
`ExecutorchBackendConfig()`'s default field —
`memory_planning_pass: ... = MemoryPlanningPass()` — whose own default is
`alloc_graph_input=True` (`executorch/exir/passes/memory_planning_pass.py:151`).
With `alloc_graph_input=True`, every graph input gets planned into an
internally-owned buffer, and the runtime copies the caller-supplied tensor into
it on **every** `forward()` call.

ExecuTorch's own llama exporter does not hit this default —
`executorch/extension/llm/export/builder.py:506-516` explicitly passes:

```python
ExecutorchBackendConfig(
    ...,
    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, ...),
)
```

so its exported inputs are used directly (zero-copy, caller-owned).

**Verified via** (not inferred): deserializing both `.pte` files with
`executorch.exir._serialize._program.deserialize_pte_binary` showed byte-identical
non-delegated instruction streams (624 instructions, 136 XNNPACK delegate calls,
same 14-op table, same op histogram) on both exports — ruling out a
graph-structure difference — and then `PyModule.method_meta('forward').input_tensor_meta(i).is_memory_planned()`
showed `True` for tsurgeon's export vs. `False` for ExecuTorch's, isolating the
one real config difference between two otherwise-identical graphs.

**Fix applied:** pass a matching config at the same call site:

```python
from executorch.exir import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass

et_program = edge.to_executorch(
    config=ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
    )
)
```

**Measured impact (re-benchmarked after applying the fix, TinyLlama-1.1B,
10 trials):** negligible — decode went from 17.9→18.2 tok/s (fp32) and
31.3→31.6 tok/s (w4), both within run-to-run noise. This matches the caveat
raised when the fix was proposed: the copied input tensors in the LLM
decode-step case are tiny (a 1-element token id + a 1-element position id), so
removing that copy was never going to move a whole-model forward pass
noticeably. **The fix is still correct and worth keeping** — it removes a
real inefficiency and matches ExecuTorch's own convention — but it is
confirmed **not** the explanation for the 1.5–2.6x decode-speed gap seen
against ExecuTorch's own export. That gap remains unexplained; a complete
explanation would need ETDump per-op timing, which requires an ExecuTorch
build with `EXECUTORCH_ENABLE_EVENT_TRACER` (unavailable in the environment
this was investigated in). See `TSURGEON_VS_EXECUTORCH.md` (sibling
`/workspace/executorch_llama/` directory) for the full before/after numbers.

**Reproduce (before fix):**
```bash
python3 -c "
from executorch.extension.pybindings.portable_lib import _load_for_executorch
m = _load_for_executorch('<tsurgeon-exported>.pte')
meta = m.method_meta('forward')
print(meta.input_tensor_meta(0).is_memory_planned())  # True (bug) vs False (executorch's own export)
"
```

---

## P2 — KV-cache writes assume exactly one new token per forward() call

**Scope note up front:** this is a batched-prefill / multi-token-forward problem
only. Ordinary one-token-at-a-time autoregressive decode (the common on-device
chat inference pattern) is unaffected and works correctly today. This is why it
was deprioritized as future work rather than fixed now.

**Symptom:** `LLMWrapper` (`export/common.py`) traces with a static
`seq_len=1` example input, and `resolve_components_and_wrapper` explicitly
discards any `dynamic_shapes` passed in config ("dynamic_shapes is ignored;
exporter uses a static seq_len=1 contract" warning). So today, feeding a whole
prompt through in one batched call isn't just untraced — parts of the
underlying cache-write code would give wrong or crashing results if you tried.
Measured impact: our benchmark had to fall back to a token-by-token loop even
for prefill, at 17.9–31.2 tok/s vs. ExecuTorch's real batched-prefill
93–262 tok/s on the same model — the single largest gap in the whole
comparison (see `TSURGEON_VS_EXECUTORCH.md`).

**Per-path detail** (`transformersurgeon/blocks/mha.py`, `MHACausal`), from
reading each of the four cache-write mechanisms individually rather than
assuming they're uniformly broken:

- **`_write_mutable`** (`self.key_cache.index_copy_(0, write_index, k)`) and
  **`_write_scatter`** (`key_cache.index_put((write_index,), k)`) — both
  underlying PyTorch ops natively support `write_index` as a range of
  positions with `k`'s leading dim sized to match. Nothing here is hardcoded
  to a single token. **Likely already compatible with multi-token writes,
  untested** — the blocker for these two is purely the export-tracing layer
  (static seq_len=1 + ignored `dynamic_shapes`), not the cache math itself.

- **`_write_concat`** — genuinely, functionally broken for `in_seq_len>1`,
  not just "assumed" to be single-token:
  ```python
  def _write_concat(self, k, v, write_index, key_cache, value_cache):
      idx = torch.arange(self.max_cache_length, device=k.device)
      sel = (idx == write_index).view(self.max_cache_length, 1, 1)
      k_row = k.reshape(1, self.kv_num_heads, self.key_head_dim)   # <- hardcoded literal 1
      ...
  ```
  `k_row`'s reshape hardcodes a leading `1`; with `in_seq_len>1`, `k` holds
  more elements than that shape allows, so this raises a shape-mismatch error
  immediately rather than silently misbehaving. A real fix needs restructuring
  into a proper `(in_seq_len, max_cache_length)` one-hot matrix + batched
  write (or a slice-and-concat), not a one-line change.

- **`custom_sdpa`** (`_forward_custom_sdpa`, the path used in the benchmark
  that surfaced this report) — the most promising path, and worth
  prioritizing over the other three if multi-token support becomes a
  priority: `torch.ops.llama.update_cache` / `torch.ops.llama.custom_sdpa`
  are ExecuTorch's **own** ops, apparently the same ones ExecuTorch's official
  llama transformer uses for its own batched prefill (writing a contiguous
  multi-token block in one call is exactly their designed purpose) — so this
  path is plausibly close to already working for `in_seq_len>1`. One
  concrete, precisely-located bug was found here though:
  ```python
  attn_mask_2d = attn_mask.reshape(1, -1).to(torch.float32)
  ```
  with the code's own comment noting "custom_sdpa requires *exactly* rank 2."
  For `in_seq_len==1` this is harmless. For `in_seq_len>1`, `reshape(1, -1)`
  **flattens the `(in_seq_len, max_cache_len)` causal mask into one row**,
  mixing different query positions' masks together — a silent correctness
  bug (wrong attention output), not a crash. Fix: keep it 2D as
  `(in_seq_len, max_cache_len)` rather than collapsing it.

**Suggested priority if this becomes active work:** start with `custom_sdpa`
(smallest, most localized fix, and it's ExecuTorch's own batched-capable op
underneath) before touching `_write_concat` (needs a real rewrite) or wiring
`dynamic_shapes` through `xnnpack_export.py`'s `torch_export(...)` call and
`resolve_components_and_wrapper`'s example-input builder.

**Not fixed in this pass — future work.**

---

## P3 — Two quantization-export bugs (embedding table, `lm_head`)

Surfaced when trying to quantize every Linear + the embedding table in a Llama
model (`criteria="all"`) to match ExecuTorch's whole-model `8da4w` recipe as
closely as possible for the benchmark comparison. These are **two independent
bugs**, not two symptoms of one root cause — do not conflate them.

### P3a — `EmbeddingCompressed` hard quantization silently no-ops, then raises

**Location:** `transformersurgeon/compression/quantization.py`,
`_apply_torchao_hard_quantization`.

**Mechanism:** the function calls torchao's `quantize_(module, config)`. In
torchao 0.17.0, `quantize_()` only knows how to patch `nn.Linear` submodules
in place. `EmbeddingCompressed` is a plain `nn.Module` (not an `nn.Linear`, and
not an `nn.Embedding` subclass either — it holds the table as a `Parameter` in
`nn.Embedding`'s native `[num_embeddings, embedding_dim]` layout), so
`quantize_()` does nothing to it. The function detects that nothing changed
(`quantized_any` stays `False`) and raises loudly instead of returning a model
that looks quantized but isn't:

```python
raise NotImplementedError(
    f"Hard (torchao) quantization is not supported for {type(module).__name__} "
    "in this torchao version (quantize_() only targets nn.Linear and made no "
    "change here). Use soft quantization (hard=False) instead."
)
```

This guard is *correct behavior*, not the bug — the actual gap is that there's
no alternative code path for embeddings at all. Marked as a TODO on the class
itself in `blocks/embedding_compressed.py`.

**Impact, measured** (`../xnnpack-models/`, 2026-09-21): the embedding table and
the tied/untied `lm_head` are the only large tensors a tsurgeon `w4` export still
leaves in fp32, and the cost scales with vocabulary size. Qwen2.5-0.5B (~151k
vocab) exports to **1211 MB at w4 for a 0.49B-parameter model** — only ~2x
smaller than its own fp32 build, where TinyLlama-1.1B (32k vocab) gets ~4.4x.
This is also the leading explanation for tsurgeon's remaining **~5% decode
deficit** against ExecuTorch's `q8da4w` (median 0.952x over 5 runs), which does
quantize both via `--embedding-quantize 4,32`. At fp32, where neither side
quantizes these tensors, the two exports are at parity (1.025x). So P3 is now
the single highest-value item for closing the quantized gap.

**Fix would be:** add a
separate embedding-quantization path — either a torchao config that actually
targets `nn.Embedding` (check newer torchao versions for an
`IntxWeightOnlyConfig`-equivalent embedding path) or a hand-rolled int-pack +
scale, mirroring what ExecuTorch's own `--embedding-quantize` flag does
(`executorch/examples/models/llama/source_transformation/quantize.py`).

**Reproduce:**
```python
manager.set("quantization", "precision", 4, criteria="all")  # "all" reaches embed_tokens
manager.set("quantization", "method", "vanilla", criteria="all")
manager.apply(hard=True, criteria="all")
# NotImplementedError: Hard (torchao) quantization is not supported for EmbeddingCompressed ...
```

### P3b — `lm_head`/`final_layer` fails during PT2E scale injection at export time — root cause NOT fully diagnosed

**Location:** `transformersurgeon/export/common.py`,
`inject_scales_into_pt2e_observers` / `_layer_names_from_observer`.

**Symptom:** after excluding the embedding table (P3a) and quantizing
attention + MLP + `lm_head` Linear layers (`criteria=["self_attn", "mlp",
"lm_head"]`), `manager.apply(hard=True, ...)` succeeds, but
`export_to_backend(..., XNNPACKExportConfig(...))` later raises:

```
RuntimeError: Could not inject weight scales into PT2E observers for layers: ['final_layer']. The observer graph structure may have changed — check _layer_name_from_observer.
```

**What's confirmed:** the failure is in the weight-observer matching step —
`inject_scales_into_pt2e_observers` walks the traced FX graph from each PT2E
observer to the nearest `linear`/`mm`/`addmm` node and reads that node's
`nn_module_stack` to recover a qualified layer name, which it then looks up in
`layer_info` (built by `extract_layer_quant_info`). For every attention/MLP
layer this succeeds; for `final_layer` (the wrapper's name for `lm_head`) it
doesn't, and `final_layer` ends up in `unmatched_hard`.

**What's NOT confirmed — this is the one finding in this report that's not
fully root-caused:** *why* the qualified-name lookup fails for `final_layer`
specifically. `final_layer` is a direct top-level attribute of `LLMWrapper`
(not nested under `decoder`), so its `nn_module_stack` qname should plausibly
just be `"final_layer"` — matching `layer_info`'s key exactly, per the error
message — which makes the mismatch not obvious from static reading alone. The
code already has a built-in diagnostic for exactly this situation
(`_diag` list, printed via `pprint.pprint(_diag[0])` right before the
`RuntimeError`) — re-running the reproduce steps below and capturing that
printed dict (truncated in the logs available when this report was written)
is the fastest way to actually pin this down, rather than guessing further.

**Reproduce:**
```python
manager.set("quantization", "precision", 4, criteria=["self_attn", "mlp", "lm_head"])
manager.set("quantization", "method", "vanilla", criteria=["self_attn", "mlp", "lm_head"])
manager.apply(hard=True, criteria=["self_attn", "mlp", "lm_head"], verbose=True)  # succeeds
export_to_backend(model, config=XNNPACKExportConfig(output_path=..., backend="xnnpack", ...))
# RuntimeError: ... for layers: ['final_layer'] — capture the pprint'd _diag[0] dict printed just above this
```

**Not fixed in this pass — future work.** Next step should be capturing the
`_diag[0]` dump above before attempting a fix, rather than guessing at the
string-matching failure.

---

## Workaround used for the benchmark comparison (not a framework fix)

Given P3a/P3b, the benchmark in `TSURGEON_VS_EXECUTORCH.md` quantizes only
`criteria=["self_attn", "mlp"]` (excludes the embedding table and `lm_head`),
which is why `tsurgeon_w4`'s exported size is larger than ExecuTorch's
`xnnpack_q8da4w` (which quantizes the embedding table too, via a separate
`--embedding-quantize` mechanism) — see that file's "Caveats" section.
