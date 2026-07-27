# Add opt-in `attn_impl="custom_sdpa"` for XNNPACK-optimized causal attention

## Context

Prior exploration (this session) established that ExecuTorch's XNNPACK backend has two candidate paths for a fused/optimized attention kernel:

1. A native XNNPACK-delegate fused SDPA node — dead in the installed ExecuTorch 1.3.1: the schema (`XNNScaledDotProductAttention`) and pattern-matching pass exist, but no `NodeVisitor` is registered to serialize it. Not usable.
2. `torch.ops.llama.custom_sdpa` + `torch.ops.llama.update_cache` — ExecuTorch's own hand-optimized CPU custom ops, which are the actual mechanism behind Meta's official on-device Llama export recipe (`--use_sdpa_with_kv_cache` in `executorch/examples/models/llama/export_llama_lib.py`). This is real and importable in the installed environment.

ExecuTorch's own helper functions for wiring path 2 in (`replace_sdpa_with_custom_op`, `replace_kv_cache_with_custom_kv_cache`) don't apply here: they do `isinstance` checks against ExecuTorch's own `SDPA`/`KVCache` module classes, and this repo's `MHACausal` calls `F.scaled_dot_product_attention` inline rather than being built from those classes. So the ops must be invoked directly inside `MHACausal`, as a new branch alongside the existing `use_sdpa` toggle — the same place that toggle already lives.

Goal: add this as a third, fully opt-in attention execution mode for the causal decoder path (`MHACausal`, used by the repo's actual LLM/Qwen2 XNNPACK export), leaving today's default behavior (`manual` attention, and the existing `use_sdpa=True` plain-SDPA path) completely unchanged and available. Scope is deliberately limited to the decoder/causal path — `MHAEncoder`/`MHAEncoderFusedProj` (non-causal, no KV cache) are out of scope; `custom_sdpa`/`update_cache` are cache-oriented ops with no clear encoder-side benefit here, and touching them would ripple into unrelated model families.

**Important correctness constraint discovered during planning**: `torch.ops.llama.custom_sdpa`'s registered fake/meta kernel returns `torch.empty_like(query)` — i.e. its output is shaped like `query`, not `value`. This repo's `attention()`/`MHACausal` deliberately support `q_head_dim != value_head_dim` (RoPE-linked structured pruning shrinks q/k's head_dim while `v_proj` stays unpruned — exercised by `test/unit/test_gqa_rope_pruning.py`). So `attn_impl="custom_sdpa"` is only numerically valid when `q_head_dim == value_head_dim`; this must be validated with a clear error at forward time, not silently produce wrong output.

## Design

### 1. Config plumbing — new `attn_impl` on the decoder path only

Add a tri-state `attn_impl` (`"manual" | "sdpa" | "custom_sdpa"`, default `"manual"`) resolved inside `MHACausal.__init__` only — **do not touch `MHABase.use_sdpa` or `MHAEncoder`/`MHAEncoderFusedProj`**, which stay exactly as-is:

```python
ATTN_IMPLS = ("manual", "sdpa", "custom_sdpa")
# in MHACausal.__init__, after existing cache_impl/max_cache_length setup:
self.attn_impl = kwargs.get("attn_impl", None) or ("sdpa" if self.use_sdpa else "manual")
if self.attn_impl not in ATTN_IMPLS:
    raise ValueError(f"Unsupported attn_impl {self.attn_impl!r}; expected one of {ATTN_IMPLS}")
if self.attn_impl == "custom_sdpa" and self.cache_impl != "mutable":
    raise ValueError(
        "attn_impl='custom_sdpa' requires cache_impl='mutable': "
        "torch.ops.llama.update_cache in-place-mutates a dedicated batched "
        "fp32 cache buffer with no io_scatter/io_concat equivalent."
    )
```

`self.use_sdpa` stays live (derives the default when `attn_impl` isn't explicitly passed); `forward()` switches to branching on `self.attn_impl`.

Call sites to update, following the exact existing pattern used for `cache_impl`/`max_cache_len` (decoder-only branch throughout):

- `transformersurgeon/blocks/mha.py` — `MHACausal.__init__` (resolve/validate `attn_impl`, register custom_sdpa buffers, import guard), `_ensure_cache_geometry` (resize custom_sdpa buffers + validate `q_head_dim == value_head_dim`), `forward` (branch on `attn_impl`, new `_forward_custom_sdpa` helper). Add `ATTN_IMPLS` to `__all__`.
- `transformersurgeon/blocks/config.py` — `CustomDecoderConfigCompress.__init__` and `.from_source_config`: add `attn_impl=None` param, store it, add `"attn_impl"` to the `passthrough_kwargs` exclusion set (alongside existing `"use_sdpa"`, `"cache_impl"`). `CustomEncoderConfigCompress` untouched.
- `transformersurgeon/blocks/decoder.py` — `TransformerDecoderBlock.__init__`: `self.attn_impl = getattr(config, "attn_impl", None)`, pass `attn_impl=self.attn_impl` into the `MHACausal(...)` constructor call.
- `transformersurgeon/utils/convert.py` — decoder branch only (`options.get('attn_impl', None)` → `CustomDecoderConfigCompress.from_source_config(..., attn_impl=attn_impl)`). Encoder branch untouched.
- `transformersurgeon/export/config.py` / `transformersurgeon/export/export.py` — no change needed; `attn_impl` absent from `convert_options` just means "derive from `use_sdpa`" (default `"manual"`), so existing defaults stay valid as-is.
- `scripts/executorch/xnnpack/exporter_function_test.py` — new `--attn-impl {manual,sdpa,custom_sdpa}` CLI flag (default `manual`), following the `--cache-impl` precedent exactly. Thread `"attn_impl": args.attn_impl` into the `convert_options`/`options` dict literals used for `mode=="hf"` and `mode=="direct"`, and into the `XNNPACKExportConfig(convert_options=...)` call. Also add it to the KV-cache sidecar JSON for debugging convenience. **Note: this file currently has an unrelated uncommitted local diff (renaming `_write_cache_metadata`→`_save_kv_cache_config`, `.cache_meta.json`→`.kvconfig.json`, and duplicating the `convert_options`/`options` dict literals inline) — read the file fresh before editing rather than assuming it matches what was read earlier in this session.**

Document (docstring, not enforced in code) that `attn_impl="custom_sdpa"` is XNNPACK/portable-CPU-specific — using it with the TensorRT or QNN exporters is untested/unsupported.

### 2. `MHACausal` implementation

- **`__init__`**: after existing cache setup, resolve/validate `attn_impl` as above. If `"custom_sdpa"`: lazily import `executorch.extension.llm.custom_ops.custom_ops` (registers `torch.ops.llama.custom_sdpa`/`update_cache` as a side effect of the module's own `torch.ops.load_library(...)`), raising a clear `RuntimeError` if unavailable (mirror `is_qnn_available()` in `transformersurgeon/export/executorch_exporters/qnn/qnn_export.py`). Register two new **dedicated, batched, fp32** buffers (`persistent=False`, matching the existing cache buffers' pattern): `custom_sdpa_key_cache`/`custom_sdpa_value_cache`, shape `(1, max_cache_length, kv_num_heads, key_head_dim/value_head_dim)` — separate from the existing unbatched buffers since `update_cache` requires `(bsz, seqlen, heads, dim)` layout.
- **`_ensure_cache_geometry`**: extend the existing lazy self-heal to also resize the two new buffers if pruning changed head_dim, and to raise a `RuntimeError` if `q_head_dim != value_head_dim` when `attn_impl == "custom_sdpa"` (the correctness constraint above).
- **`forward`**: branch before the existing cache-write dispatch — when `attn_impl == "custom_sdpa"`, skip `_write_mutable`/`_write_scatter`/`_write_concat` entirely and call a new `_forward_custom_sdpa(q, k, v, pos_id, attn_mask)` helper instead. That helper: unsqueezes a batch dim onto q/k/v, casts to float32, calls `torch.ops.llama.update_cache` for k and v against the dedicated caches (`start_pos = pos_id[0].item()`), computes `scale = 1/sqrt(q_head_dim)` (matching the existing `attention()` helper's convention, not the op's default), calls `torch.ops.llama.custom_sdpa(q, key_cache, value_cache, 0, attn_mask.float(), 0.0, False, scale)` (explicit mask ⇒ `start_pos=0`/`is_causal=False`, mirroring ExecuTorch's own `SDPACustom` pattern), then squeezes the batch dim and casts back to the original dtype. Output shape matches what the shared tail of `forward()` already expects (`(num_heads, in_seq_len, value_head_dim)`) — no change needed there.

### 3. `xnnpack_export.py`

No speculative code changes. `MHACausal` construction (and the custom-ops import) happens upstream in `convert_for_export`, well before `export_with_xnnpack` calls `torch_export`, so ordering is already correct. Custom ops with no registered decomposition and no matching `XnnpackPartitioner` config should pass through `to_edge_transform_and_lower` untouched as portable ops (linears delegate to XNNPACK, attention runs as the custom op — same shape as ExecuTorch's own llama export). **Verify this empirically first** (step 5); only if it errors (e.g. `EdgeCompileConfig(_check_ir_validity=...)` rejecting the mutating custom op) should this file change, most likely by conditionally relaxing `_check_ir_validity` when `attn_impl=="custom_sdpa"` is in play. Capture the exact exception before making any change here.

### 4. Tests

- **New unit test** `test/unit/test_mha_causal_custom_sdpa.py`: numeric-parity check comparing `MHACausal(attn_impl="manual")` vs `MHACausal(attn_impl="custom_sdpa")` on identical weights/inputs (fp32, to isolate kernel-math differences from fp16 rounding), across a few decode steps, `torch.testing.assert_close` with a tight tolerance. Also a negative test asserting `ValueError` for `attn_impl="custom_sdpa", cache_impl="io_scatter"`, and a test that RoPE-pruned q/k (unequal to value head_dim) raises the new `RuntimeError` under `attn_impl="custom_sdpa"` (mirrors the pruning setup already in `test/unit/test_gqa_rope_pruning.py`).
- **New capability marker** in `test/_helpers/capabilities.py`: `requires_custom_sdpa` gated on `_module_available("executorch.extension.llm.custom_ops.custom_ops")`, following the existing `HAS_EXECUTORCH` pattern — narrower than plain `requires_executorch` since a minimal ExecuTorch install may lack the LLM custom-ops extension.
- **New e2e test** in `test/e2e/test_export_pipelines.py`, mirroring `test_export_xnnpack`: same Qwen2 setup, `convert_options={"use_sdpa": False, "attn_impl": "custom_sdpa"}`, gated with both `@caps.requires_executorch` and `@caps.requires_custom_sdpa`, asserting the `.pte` is written and non-empty.
- Run full `pytest` (per `CLAUDE.md`) plus a targeted `pytest test/e2e/test_export_pipelines.py -k xnnpack` and the new unit test file.

### 5. Manual verification (export + inference scripts, then timing comparison)

Using `/opt/conda/envs/py312_executorch_xnnpack/bin/python` (the environment where `executorch`/`torch.ops.llama.*` are confirmed importable):

1. Run the new unit test in isolation first for fast feedback.
2. Export twice with `scripts/executorch/xnnpack/exporter_function_test.py --model-name Qwen/Qwen2-0.5B-Instruct --cache-impl mutable --max-sequence-length 256 --out-dir <dir>`, once with `--attn-impl manual` (baseline, unchanged) and once with `--attn-impl custom_sdpa`, to separate output directories.
3. If step 2's `custom_sdpa` export fails inside `to_edge_transform_and_lower`, capture the exception and only then revisit `xnnpack_export.py` per section 3 — don't pre-fix speculatively.
4. Run `scripts/executorch/xnnpack/inference_exported_test.py` against both resulting `.pte` files (no script changes needed — it auto-detects KV-cache geometry from the cache sidecar and just runs whatever graph it's given).
5. Compare the already-existing `tokens_per_s` / `total_inference_s` / `avg_token_time_ms` output between the two runs — this is the speed signal the whole feature is meant to produce. Report the comparison back.

## Critical files

- `transformersurgeon/blocks/mha.py`
- `transformersurgeon/blocks/config.py`
- `transformersurgeon/blocks/decoder.py`
- `transformersurgeon/utils/convert.py`
- `scripts/executorch/xnnpack/exporter_function_test.py`
- `test/_helpers/capabilities.py`
- `test/e2e/test_export_pipelines.py`
- `test/unit/test_mha_causal_custom_sdpa.py` (new)
- `transformersurgeon/export/executorch_exporters/xnnpack/xnnpack_export.py` (only if step 5.3 surfaces a concrete failure)
