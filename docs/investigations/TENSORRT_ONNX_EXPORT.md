# TensorRT via ONNX: export path for Jetson Orin, benchmarked against TensorRT Edge-LLM

Branch: `debug-tensorrtspeed` (branched from `draft-prolucio` at `5b3d0e7`, v0.10.0).

Goal: an LLM export to **plain TensorRT** suitable for a Jetson Orin Nano
(JetPack 7.2 → TensorRT 10.16), measured against a vendor-level reference, as
was done for XNNPACK (Meta's `export_llama`) and QNN (Qualcomm's llama example).
Development and all numbers here are on an x86 host with NVIDIA L4 GPUs; the
Orin run is future work (see §7).

---

## 1. Decisions and why

- **Not TensorRT-LLM.** Since 1.2 it has no TensorRT backend at all (PyTorch is
  its only runtime), and it never properly supported Orin Nano.
- **Not the previous torch-tensorrt `tensorrt` backend** (since removed; the
  ONNX path now owns the `tensorrt` name). Its artifact was an
  engine compiled for the build GPU (did not run on a Jetson), it kept the
  mutable KV cache in PyTorch on the CPU (graph splits + host↔device copies
  per token), and it relied on weak typing (removed in TensorRT 11).
- **ONNX + engine built on the target.** An engine is tied to GPU arch +
  TensorRT version, so the portable artifact is ONNX; the engine is built on
  the device. This is also exactly what NVIDIA's own edge stack does.
- **Vendor reference: TensorRT Edge-LLM 0.10.1** (Apache-2.0, NVIDIA): HF →
  ONNX → plain TensorRT engine + plugins + C++ runtime; supports Qwen2-0.5B and
  Jetson Orin (FP16/INT8/INT4).
- **"ONNX can't do KV caches" is not true.** ONNX has no *mutable state*, but
  the standard pattern is the cache as graph I/O, updated in place by the
  runtime binding input and output to the same buffer (ONNX opset 24 added
  `TensorScatter` for precisely this).

## 2. Environment

Conda env `py312_tensorrt` (host): Python 3.12, torch 2.13.0+cu132,
transformers 5.14.1, onnx 1.19.0, onnxscript 0.7.1, **tensorrt-cu13 10.16.1**
(pinned to JetPack 7.2's major/minor), torchao 0.18, TensorRT Edge-LLM 0.10.1
(Python export tools + C++ runtime built for sm_89 with every CuTe DSL kernel
group, matching the Orin build's `ENABLE_CUTE_DSL=ALL`), CUDA toolkit 13.2.
TensorRT C++ headers + `trtexec` extracted from NVIDIA's apt repo into
`$CONDA_ENV/opt/tensorrt-10.16.1` (no sudo). Full setup: `../trt-models/AGENTS.md`.

The existing tsurgeon suite passes unchanged on this stack (torch 2.13 /
transformers 5.14).

## 3. Phase 2: how TensorRT 10.16 treats a KV-cache decode step from ONNX

Micro-benchmark (`trt-models/experiments/kv_attention_microbench.py`): 24
attention-only layers, Qwen2-0.5B geometry, one token, fixed-size cache.

| Finding | Evidence |
|---|---|
| `TensorScatter` is executed **in place** (TensorRT aliases output to input) | `engine.get_aliased_input_tensor(present) == past`; correct results |
| In-place write is 12% cheaper than `ScatterND` (today's `io_scatter`) | 0.266 vs 0.304 ms / 24 layers @ 1024 slots |
| Full-cache attention costs O(max_cache_len) | 0.30 → 0.98 → 4.08 ms at 1k → 4k → 16k slots, pos fixed |
| ONNX `Attention` op is unusable on 10.16 | every masked form fails to build (`wrap_attention_op_in_kgen` CHECK) or crashes at run time (`nonpad_kv_seqlen`) |
| Slicing attention to the used span (host shape tensor) only pays off when cache ≫ context | span = full cache is *slower* (dynamic-shape kernels): 0.48 vs 0.27 ms |
| CUDA graphs save ~0.1 ms / step of host launch cost | graph vs no-graph rows |

Conclusion: go — in-place KV from standard ONNX works; attention stays explicit
MatMul/Softmax over the (Orin-sized, 1024-slot) cache.

## 4. What was built

- **`cache_impl="io_inplace"`** (`blocks/mha.py`, `blocks/kv_cache_ops.py`,
  `blocks/decoder.py`): BHSD cache I/O written by `tsurgeon::kv_cache_update`
  (functional in eager, opset-24 `TensorScatter` in ONNX); the only mode that
  accepts `in_seq_len > 1`, so a prompt is prefilled in one call. The decoder
  builds the `(seq, max_cache_len)` causal mask directly instead of indexing a
  `max_cache_len²` matrix per step. `MHACausal.cache_shape()` is now the single
  source of cache geometry (buffers and exporters).
- **`rmsnorm_upcast`** (`blocks/norm.py`): float32 RMSNorm (HF / Edge-LLM
  semantics). Needed because `rmsnorm_prescale=False` in fp16 overflows
  (logit error 10.2 vs HF, measured); upcast costs nothing measurable on GPU.
- **`onnx` backend** (`export/onnx/`): portable `model.onnx` + manifest (I/O
  names, cache shapes/layout/dtype, eos ids, quantized layers, plugin needs);
  weight-only Q/DQ (`quantization.py`); extension points `custom_translations`
  and `graph_passes`.
- **`tensorrt` backend** (`export/tensorrt/tensorrt_export.py`): onnx export
  + optional local engine build (strongly typed; decode and prefill
  optimization profiles). CLI for on-device builds:
  `python -m transformersurgeon.export.tensorrt.tensorrt_export model.manifest.json`.
- **Runtime** (`export/tensorrt/engine.py`, `session.py`): engine build/run on
  torch CUDA tensors, profile-sharing runners, CUDA-graph capture;
  `TensorRTLLMSession` does chunked prefill + CUDA-graph greedy decode (engine
  + argmax + position update in one graph).
- **Opt-in Edge-LLM INT4 plugin** (`export/tensorrt/edgellm_int4.py`,
  `int4_backend="edgellm_plugin"`): see §5.3.

## 5. Results (L4, Qwen2-0.5B-Instruct, KV 1024, batch 1)

Median of 5 separate processes (between-process spread ≤ 0.1% for decode,
≤ 4.7% for prefill). Decode = one step at the given past-KV length (ms, tok/s);
prefill = one call over N prompt tokens (ms). Raw rows: `trt-models/results/`.

| impl | model | decode 16 | decode 256 | decode 900 | prefill 128 | prefill 512 |
|---|---|---:|---:|---:|---:|---:|
| Edge-LLM | FP16 | 4.738 (211) | 4.803 (208) | 4.924 (203) | 5.86 | 12.01 |
| **tsurgeon** | **FP16** | **4.702 (213)** | **4.712 (212)** | **4.718 (212)** | 8.17 | 21.43 |
| tsurgeon | FP16, prescale RMSNorm | 4.719 (212) | 4.731 (211) | 4.730 (211) | 8.20 | 21.59 |
| Edge-LLM | INT4 AWQ | 2.635 (379) | 2.698 (371) | 2.804 (357) | 4.93 | 10.66 |
| Edge-LLM | INT4 GPTQ | 2.661 (376) | 2.719 (368) | 2.825 (354) | 5.18 | 11.34 |
| **tsurgeon** | **W4 + Edge-LLM plugin** | **2.586 (387)** | **2.592 (386)** | **2.589 (386)** | 5.09 | 14.98 |
| tsurgeon | W4, plain TensorRT | 4.655 (215) | 4.666 (214) | 4.674 (214) | 5.96 | 15.94 |
| tsurgeon | W4 block-128, plain TensorRT | 4.671 (214) | 4.689 (213) | 4.691 (213) | 5.96 | 16.25 |
| tsurgeon | W8, plain TensorRT | 4.670 (214) | 4.681 (214) | 4.685 (214) | 8.07 | 21.20 |

For reference, the unmodified `io_scatter` graph exported through ONNX ran a
decode step in ~5.5 ms (trtexec GPU time).

### 5.0 Confirmation run (after the fixes and cleanup)

All six engines rebuilt from `3ad9eb3` (after the eps/RoPE fixes and the
removal of the torch-tensorrt backend), benchmarked on an idle L4 with vendor
and tsurgeon **interleaved** round by round (5 rounds, one process per
measurement), so both see the same thermal/power state. Median between-process
spread 1.0%, worst 6.0%. Raw data: `trt-models/results/final_l4_gpu1.json`.

| impl | model | decode 16 | decode 256 | decode 900 | prefill 128 | prefill 512 |
|---|---|---:|---:|---:|---:|---:|
| Edge-LLM | FP16 | 4.708 (212) | 4.766 (210) | 4.879 (205) | 5.63 | 11.48 |
| **tsurgeon** | **FP16** | **4.664 (214)** | **4.679 (214)** | **4.676 (214)** | 8.01 | 20.81 |
| Edge-LLM | INT4 AWQ | 2.617 (382) | 2.680 (373) | 2.793 (358) | 4.76 | 10.13 |
| Edge-LLM | INT4 GPTQ | 2.636 (379) | 2.702 (370) | 2.814 (355) | 4.99 | 10.85 |
| **tsurgeon** | **W4 + Edge-LLM plugin** | **2.588 (386)** | **2.609 (383)** | **2.595 (385)** | 5.12 | 14.95 |
| tsurgeon | W4, plain TensorRT | 4.638 (216) | 4.668 (214) | 4.669 (214) | 5.82 | 15.94 |

A first attempt of this run was discarded: another user job started on the
same GPU mid-run and slowed every later measurement by ~10% (it read as a
tsurgeon regression). Benchmark on a GPU verified idle, and interleave.

**What can be claimed.** On an NVIDIA L4, tsurgeon's ONNX→TensorRT export of
Qwen2-0.5B-Instruct matches NVIDIA TensorRT Edge-LLM on single-stream decode:
FP16 with stock TensorRT (1–4% faster) and INT4 when using Edge-LLM's INT4
GEMM plugin (1–8% faster than Edge-LLM AWQ/GPTQ). Prompt prefill is 1.4–1.8×
slower. Not yet covered: the Orin itself, INT4 accuracy vs AWQ/GPTQ
(tsurgeon's per-channel RTN is less accurate), larger KV caches (tsurgeon
scans the full cache), other models, batch > 1.

### 5.1 Decode: parity
FP16 and INT4 decode match or slightly beat Edge-LLM. Caveat on the flat
tsurgeon curve: our attention always scans the whole 1024-slot cache while
Edge-LLM's scans only the filled part — so tsurgeon is ahead late in the
context and would fall behind with a much larger `max_cache_len` (§3).

### 5.2 Prefill: behind at long prompts
FP16 128 tokens 1.39×, 512 tokens 1.78× slower; INT4+plugin 1.03× / 1.40×.
Profiled at 512 tokens: 58% of the time is MLP GEMMs. TensorRT picks
`sm80_xmma_gemm_f16f16_f16f16_f16 … aligna2_alignc2` (fp16 accumulate, 2-element
alignment) for our graph but aligned `f16f16_f16f32_f32` tiles for Edge-LLM's,
on the same TensorRT build. Ruled out: `Gemm` vs `MatMul`+`Add`, profile
`opt` (128/256/512), builder optimization level 5. Root cause open (§7).

### 5.3 INT4: needs a real INT4 GEMV
Plain TensorRT 10.16 has no efficient single-token INT4 kernel: for 896→4864
it fuses the dequant into a generic tensor-core GEMM; for down_proj
(4864→896) it dequantizes the whole weight to fp16 **every step** — INT4 ends
up at fp16 speed. Edge-LLM's `Int4GroupwiseGemmPluginV2` computes
`(nibble − 8) × scale[group, n]`, which is exactly tsurgeon's symmetric INT4
(per-channel scales tile over groups unchanged), so the opt-in pass rewrites
eligible `DequantizeLinear → Gemm` pairs into it (Edge-LLM's own weight
repacker; fallback to Q/DQ for layers with N % 64 or K % 128 ≠ 0, e.g. odd
pruned/LRD shapes). Microbench on down_proj: 2.96 → 1.18 ms / 24 layers.
INT8 weight-only brings nothing on TensorRT (folded to fp16).

## 6. Correctness

- Teacher-forced over 24 steps vs HF fp32: argmax identical at every step,
  logit error flat (0.08–0.22, no drift) — decode path, positions and in-place
  cache are correct.
- The residual error vs HF was tsurgeon's conversion, not TensorRT: tsurgeon in
  eager **fp32** differed from HF fp32 by 0.07–0.10 because RMSNorm `eps` was
  hardcoded to 1e-5 (Qwen2 uses 1e-6). **Fixed** (eps and RoPE theta now come
  from the HF config): fp32 error is now 1e-5 with `rmsnorm_upcast` (0.006 with
  the default prescale norm). The §5 benchmarks predate the fix; it does not
  change the graph's cost.
- TensorRT W4 vs eager tsurgeon W4: max 0.14 (fp16 noise). Plugin W4 vs plain
  W4: 17/17 and 14/15 identical greedy tokens.
- Quantization quality note: tsurgeon's per-channel round-to-nearest INT4 is
  lossy on this 0.5B model (logit error 6.6 vs HF even in eager); AWQ/GPTQ
  exist to fix exactly that. Speed comparisons above are unaffected.
- Two TensorRT 10.16 pitfalls now handled in `quantization.py`: DQ scales that
  underflow to 0 in fp16 are rejected (floored at 2⁻¹⁴), and a single INT4
  block spanning the whole row returns NaNs (per-channel INT4 uses ≥ 2 blocks).

Tests: `test/unit/test_io_inplace_cache.py`, `test/unit/test_onnx_tensorrt_export.py`,
`test/e2e/test_export_pipelines.py::test_export_tensorrt` (FP16, W4,
W4+plugin on the real checkpoint). Full suite: 107 passed; XNNPACK/custom-SDPA
tests re-run in the ExecuTorch env: pass.

## 7. Open items

1. **Run on the Orin** (JetPack 7.2.1): build Edge-LLM there (prebuilt sm_87
   CuTe DSL kernels ship with it), copy `model.onnx` + manifest, build with the
   CLI above, rerun both benchmarks. The Orin's 6-core CPU makes CUDA graphs
   and host overhead matter more; a C++ runner around the same manifest may be
   worth it.
2. **Prefill GEMM tactics** (§5.2) — find why TensorRT picks misaligned fp16-
   accumulate kernels for our graph.
3. **Attention over the used span only**, for large `max_cache_len`
   (bucketed shape input + one CUDA graph per bucket; §3 shows the trade-off),
   or reuse Edge-LLM's `AttentionPlugin` the same way as the INT4 plugin.
4. **RoPE scaling variants** (e.g. Llama 3's `rope_type: llama3`) are not
   implemented; conversion now warns instead of silently using plain RoPE.
5. **AWQ/GPTQ-quality INT4** in tsurgeon (per-group scales, activation-aware).

## 8. Reproduce

```bash
cd trt-models && source env.sh
$PYTHON build_vendor.py && $PYTHON bench_vendor.py --repeats 5
$PYTHON export_tsurgeon.py && $PYTHON bench_tsurgeon.py --repeats 5
$PYTHON verify_tsurgeon.py qwen2-0.5b-fp16
```
