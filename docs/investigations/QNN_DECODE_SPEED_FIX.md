# QNN/HTP decode-speed investigation

Branch: `debug-qnnspeed` (branched from `debug-xnnpackspeed`).

Goal: bring transformer-surgeon's QNN export closer to the reference exports
Qualcomm ships in `executorch/examples/qualcomm/`. Constraint: this is an x86
host, so nothing can be *run*. Every conclusion below is derived at ahead-of-time
compile time, from the QNN HTP compiler's own output plus the lowered graph.

---

## 0. Environment integrity

| Component | Status |
|---|---|
| QNN / QAIRT SDK | `2.37.0.250724` at `$QNN_SDK_ROOT`, x86_64-linux-clang libs present (`libQnnHtp.so`, `libHtpPrepare.so`, `libQnnSystem.so`, …), hexagon-v68…v79 stubs present |
| ExecuTorch | `v1.3.1` (`e2f18eb`), sourced from `/workspace/executorch` via `PYTHONPATH` |
| QNN AOT bindings | `PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so` imports and runs |
| QNN partitioner / quantizer | `QnnPartitioner`, `QnnQuantizer`, `generate_htp_compiler_spec` all import |
| End-to-end AOT lowering | **Works** — a smoke-test Linear+GELU compiles through HTP and emits a DDR bandwidth summary |
| ExecuTorch **runtime** pybindings | **Not built** (`executorch.extension.pybindings._portable_lib` missing) |

Two notes, neither blocking:

- `PyQnnWrapperAdaptor` no longer exists as a separate module in ExecuTorch 1.3.1
  (it was merged into the manager adaptor); nothing in-tree references it.
- QNN logs `BackendOpInfo API not available` — that API needs SDK ≥ 2.41, and the
  backend falls back to its abstract implementation. Harmless for AOT lowering.

The missing runtime pybindings are why three tests fail in this container
(`test_mha_causal_custom_sdpa.py` ×2, `test_known_bugs.py::test_qnn_unavailable…`).
**They fail identically on the unmodified branch** — verified by stashing — so they
are environmental, not regressions.

---

## 1. The vendor reference

`executorch/examples/qualcomm/oss_scripts/llama/` is a complete Qualcomm-authored
static transformer decoder for HTP. The parts that matter:

| File | What it establishes |
|---|---|
| `model/static_llama.py` | KV-cache layout, attention formulation, GQA repeat, cache write strategy |
| `model/apply_rope.py` | half-split (HF-style) RoPE — explicitly because interleaved RoPE needs a stride-2 StridedSlice, "not friendly for HTP backend" |
| `model/layernorm.py` | norms are plain `torch.nn.RMSNorm` / `torch.nn.LayerNorm`, nothing more |
| `model/feed_forward.py` | `prepare_feedfoward_conv()` — Linear → 1×1 Conv2d; also notes "Gelu is a fused op in QNN and can run faster" |
| `wrappers/llm_wrappers.py` | calls `convert_linear_to_conv2d(decoder)` over the whole model |
| `backends/qualcomm/utils/utils.py` | `to_edge_transform_and_lower_to_qnn(..., convert_linear_to_conv2d=True)` |

The decisive one is `static_llama.py::LlamaAttention.forward`:

```python
k = k.transpose(2, 3)                       # transpose the *new token*, once
kh = torch.cat([k_caches, k], dim=-1)       # cache is already (B, kv, head_dim, L)
vh = torch.cat([v_caches, v], dim=2)        # cache is already (B, kv, L, head_dim)
kh = repeat_kv(kh, self.num_key_value_groups)
attn = q @ kh                               # no transpose of a cache-sized tensor
```

The cache is **stored in the layout the matmul wants**. A cache-sized tensor is
never permuted.

---

## 2. Measurement method

`to_edge_transform_and_lower_to_qnn` runs the real HTP compiler, which prints:

```
====== DDR bandwidth summary ======
spill_bytes=… fill_bytes=… write_total_bytes=… read_total_bytes=…
```

`spill`/`fill` are the giveaway: they are non-zero exactly when an intermediate
tensor does not fit in VTCM and has to round-trip through DDR. On a decode step —
which is memory-bound, not compute-bound — this is the thing that costs time.

A probe harness (captured at fd level, since QNN logs from native code) records
that summary, the histogram of ops the QNN backend visited, and how many nodes
stayed outside the delegate.

---

## 3. Root cause: GQA expand-then-transpose

`blocks/mha.py::attention()` did:

```python
key = key.unsqueeze(2).expand(-1, -1, group_size, -1).reshape(-1, q_head_num, k_head_dim)
key = key.transpose(0, 1)
scores = torch.matmul(query * scale, key.transpose(-2, -1))
```

It **expands first, transposes second**. The expand materialises the whole KV cache
`group_size` times over (14/2 = 7× for Qwen2-0.5B), and the transpose then runs over
that 7×-larger tensor. For a 2048-slot cache that intermediate is 7.3 MB — far past
VTCM — so HTP spills it.

Isolated micro-benchmark, one attention op, 2048-slot cache, 14 q heads / 2 kv heads,
SM8650:

| variant | spill | fill | write | read |
|---|---|---|---|---|
| expand-then-transpose (previous) | 9.18 MB | 9.18 MB | 9.24 MB | 9.22 MB |
| transpose-then-expand | 0 | 0 | 0.06 MB | 0.04 MB |
| vendor cache layout (K stored transposed) | 0 | 0 | 0.06 MB | 0.05 MB |
| broadcast, no materialised repeat | 0 | 0 | 0.01 MB | 0.05 MB |

All four are numerically identical (max |diff| 9.3e-10 in fp32).

Two things worth noting. First, simply reordering the two operations captures
essentially the entire win — changing the *stored* cache layout to the vendor's adds
almost nothing on top, which matters because changing the storage layout would ripple
into `cache_impl`, `build_zero_caches` and the runtime contract. Second, the
broadcasting form removes the `expand_copy` node altogether: QNN's MatMul broadcasts,
so the GQA repeat never needs to be materialised.

The fix was already in the repo — as a commented-out block labelled
`ALTERNATIVE 2 — broadcasting GQA`. It was written and left switched off.

---

## 4. Changes made

### 4.1 `blocks/mha.py::attention()` — broadcasting GQA *(the fix)*

Group query heads under their kv head and let matmul broadcast:

```python
query = query.transpose(0, 1).reshape(kv_head_num, group_size, q_seq_len, q_head_dim)
key   = key.permute(1, 2, 0).unsqueeze(1)   # (kv_head_num, 1, k_head_dim, kv_len)
value = value.transpose(0, 1).unsqueeze(1)  # (kv_head_num, 1, kv_len, v_head_dim)
scores = torch.matmul(query * scale, key)
```

`key.permute(1, 2, 0)` replaces the old transpose-pair and operates on the
*unexpanded* cache. The pruning-aware separation of `q_head_dim` / `k_head_dim` /
`v_head_dim` is preserved, so RoPE-linked structured pruning still works.

Verified on the real model: 8 decode steps of Qwen2-0.5B-Instruct, new attention vs
the previous formulation monkeypatched back in — max |Δlogit| 3.3e-05, relative
1.9e-06, **identical argmax token at every step**.

Verified not to regress XNNPACK: nodes outside the delegate go 20 → 18, with one
fewer `expand_copy.out`.

### 4.2 `blocks/norm.py::RMSNorm` — `prescale` option

The max-abs prescale (`abs` → `max.dim` → `clamp` → `div`) turns what ExecuTorch
would pattern-match into a single `aten.rms_norm` into six ops. Across 49 norms that
is ~245 extra HTP dispatches per token. It is a real fp16 overflow guard, and
disabling it is not bit-identical (rescaling changes how `eps` enters the variance),
so it is **opt-out, default unchanged**, threaded as `rmsnorm_prescale` through
`convert_options` → `blocks/config.py` → `decoder.py`, with a
`--no-rmsnorm-prescale` CLI flag.

### 4.3 `export/common.py::LLMWrapper` — rank-2 lm_head input

`hidden[-1, :]` fed the lm_head a **rank-1** tensor. `_head()` now slices
`hidden[-1:, :]` and drops the rank after the projection — same output contract,
but rank-2 into the linear. Rank-1 breaks ExecuTorch's `ConvertLinearToConv2d`
(it reshapes rank-1 to rank-2 then applies a 4-element permutation:
*"input.dim() = 2 is not equal to len(dims) = 4"*), and rank-2 is what QNN's
FullyConnected builder handles most robustly anyway.

### 4.4 `export/export.py` — `max_seq_len` now sizes the KV cache *(bug fix)*

`--max-sequence-length` was documented as "used for cache size" and **did nothing**.
`convert_for_export` reads the cache size from `convert_options["max_cache_len"]`,
which nothing populated, so every export silently shipped a 2048-slot cache. This is
why the baseline produced byte-identical DDR summaries at `--max-sequence-length 128`
and `1024`. `export_to_backend` now defaults `max_cache_len` from `config.max_seq_len`;
an explicit `max_cache_len` still wins.

### 4.5 `QNNExportConfig.convert_linear_to_conv2d` — plumbed, opt-in

The vendor-recommended conv2d lowering is now reachable
(`--linear-to-conv2d`), but **defaults off**. Two reasons. It produced no
measurable DDR change here — it swaps 14 linears for 14 convolutions plus 28
reshape/permute pairs that LayoutTransform is expected to fold, so any win is in
kernel selection, unmeasurable without a device. And ExecuTorch 1.3.1's pass has
real gaps, two of which `_resolve_linear_to_conv2d` now detects up front and warns
about rather than failing deep in lowering:

- **fp16 weights** — `_reshape_weight` only re-wraps in `nn.Parameter` when
  `dtype == torch.float`, so fp16 hits *"expected torch.nn.Parameter for PARAMETER
  attr_kind, got <class 'torch.Tensor'>"*.
- **tied weights** — for a weight with >1 user it registers a reshaped *fake*
  tensor as a buffer; the later submodule `deepcopy` then raises *"Only Tensors
  created explicitly by the user (graph leaves) support the deepcopy protocol"*.
  Minimal repro: two `nn.Linear` sharing one weight. Ironically the pass's own
  comment claims to handle exactly this case.

A third failure — the same deepcopy error on the **full fp32 decoder pipeline**,
with no shared storage present (169 distinct weights scanned) — I could not
attribute. It does not reproduce on a synthetic decoder of identical shape, with or
without qkv bias, with or without embedding + lm_head. Time-boxed and left as an
upstream issue; the flag stays off so it cannot bite.

---

## 5. Results

Qwen2-0.5B-Instruct, SM8850, fp16, `to_edge_transform_and_lower_to_qnn`, identical
settings. Both exports are **fully delegated** — a single `QnnBackend` delegate,
zero non-delegated operators — so the difference is entirely inside the delegate.

### DDR bandwidth per decode step (2048-slot cache, apples-to-apples)

| | baseline | fixed | change |
|---|---:|---:|---:|
| `spill_bytes` | 267.6 MB | **0** | −100% |
| `fill_bytes` | 269.6 MB | **0** | −100% |
| `write_total_bytes` | 360.2 MB | 92.6 MB | **−74%** |
| `read_total_bytes` | 1294.1 MB | 1024.2 MB | **−21%** |

537 MB of spill/fill round-trip per token is gone. The residual 1024 MB read is
essentially the fp16 weight set itself (~988 MB) — i.e. at the floor for an
unquantized fp16 decode.

With §4.4 also in effect, so the cache is genuinely 1024 slots as requested:
`write_total_bytes` 12.9 MB, `read_total_bytes` 1002.7 MB, spill/fill 0.

### Lowered graph

`aten` ops the QNN backend visited, baseline vs final (`--no-rmsnorm-prescale`):

| op | baseline | final | Δ |
|---|---:|---:|---:|
| `aten.expand_copy.default` | 48 | **0** | −48 |
| `aten.abs.default` | 49 | **0** | −49 |
| `aten.max.dim` | 49 | **0** | −49 |
| `aten.div.Tensor` | 49 | **0** | −49 |
| `aten.clamp.default` | 73 | 24 | −49 |
| `getitem` | 145 | 96 | −49 |
| *(all others unchanged)* | | | |
| **total delegated nodes** | **1676** | **1384** | **−292 (−17%)** |

---

## 6. Remaining gaps vs the vendor reference

Ordered by my estimate of impact. Items marked TODO were deliberately not attempted
here; each is a larger change than what this branch covers.

1. **Quantization was intentionally excluded from these measurements, not
   unaddressed.** `extract_layer_quant_info` (`export/common.py`) scans every
   `nn.Linear` in the wrapper by compression tag — it is not MLP-specific — and the
   QNN quantizer path (`qnn_export.py`) picks up whatever the compression manager was
   applied to, attention included (`indexing_qwen2_c.py` exposes `self_attn` /
   `q_proj` / `k_proj` / `v_proj` / `o_proj` as valid criteria alongside `mlp`). The
   CLI test script's `--quant-mlp` flag only wires up MLP criteria for convenience;
   that is a script default, not a framework limitation. All measurements in §5 use
   fp16, unquantized, on purpose — to isolate the effect of the attention/norm/
   cache-size changes without conflating them with a quantization delta. A DDR
   comparison against an already-quantized config is a natural follow-up but wasn't
   run here.
2. **TODO: prefill.** The vendor exports two graphs sharing weights — prefill
   (`ar_len` = 128) and decode (`ar_len` = 1). transformer-surgeon exports decode-only
   (`in_seq_len` = 1), so prompt processing runs one token at a time. Not attempted in
   this branch.
3. **KV write is `index_put` (ScatterNd), 48 per token — a different I/O contract
   from the vendor's, not just a missing optimization.** The vendor's graph never
   scatters: each layer sets `output_new_cache_only=True`
   (`static_llama.py` `LlamaAttention.forward`), so it `cat`s the existing cache
   (read as plain input) with the fresh token for the attention math, and returns
   *only* the fresh token's K/V as output — nothing about "where that goes next" is
   compiled into the graph at all. Placement into the right slot for the next call is
   done entirely on the CPU, **between** calls: `kv_manager.cpp::update_key/
   update_value` `memcpy`s from the previous call's output buffer into the next
   call's input buffer at an offset that advances by `n_past` each step (the
   "shift-pointer" in the name) — no NPU cycles and no compiled op are spent moving
   cache data.

   This means the vendor's runner (`runner.cpp` + `kv_manager.cpp`) is **not a drop-in
   replacement for exports from this repo**, in either KV mode: `cache_impl="mutable"`
   keeps the cache as internal QNN-mutable-buffer state with no cache I/O at all
   (nothing for a memcpy-based runner to place), and `cache_impl="io_scatter"`/
   `"io_concat"` return the **whole** updated cache as output (the write already
   happened on-NPU), not a delta. Reproducing the vendor's zero-scatter behavior would
   require a new `cache_impl` that emits `output_new_cache_only`-style deltas *and* a
   matching custom runner that knows to memcpy them into place — an export-side change
   is not sufficient on its own. Not attempted here.
4. **Cache stored sequence-major** `(L, kv, hd)` rather than the vendor's
   `(B, kv, hd, L)` / `(B, kv, L, hd)`. After §4.1 the remaining permutes are on the
   *unexpanded* cache (96 `permute_copy`), and §3 measured the delta from going all
   the way to vendor layout as negligible — but it is a real structural difference.
5. **Unbatched rank-2/rank-3 tensors** throughout, vs the vendor's rank-4
   `(B, …)`, which lines up with HTP's NHWC LayoutTransform. `add_batch_dim` exists
   but is XNNPACK-scoped and untested on QNN.
6. **Linear, not Conv2d** — blocked upstream, see §4.5.

## 7. What is already aligned

Worth recording, since these were checked and needed no change: RoPE is the half-split
HF form the vendor requires (not interleaved); the causal mask is additive and built
once per token in `TransformerDecoder`, not per layer; SDPA is never emitted (QNN has
no such op) — the decomposition is explicit; the scale is applied to `q` before the
matmul rather than to the scores after, which is strictly cheaper at decode.
