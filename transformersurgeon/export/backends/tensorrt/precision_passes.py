"""TensorRT-target ONNX rewrite (runs as an ONNXExportConfig.graph_passes entry).

fp32_accumulate_lrd: compute the GEMMs of low-rank-factorized (LRD) linears with a
float32 accumulator. Weights stay fp16 in the engine and every activation between
ops stays fp16: the pass casts the GEMM operands to fp32 and the result back, and
TensorRT fuses those casts into the kernel (the engine size does not change).

Why: TensorRT 10.16 runs single-token (M=1) fp16 GEMMs with cuBLAS-style GEMV
kernels that accumulate in fp16 (``gemvx_f16f16_f16_f16``). Dense layers tolerate
it, but plain-SVD LRD makes the factorized layers ill-conditioned and amplifies the
error (measured on a 1.5B-parameter decoder).
Measured on lrd0.5-mlp (L4): KL 0.10 vs the fp16 torch reference (torch's own fp16
floor is 3e-3), and 3.5e-4 with this pass on every factorized GEMM, at 8.94 vs
8.54 ms/token. fp32 accumulation is what torch's fp16 matmul already does, so this
aligns TensorRT with the reference instead of raising the precision of the model.
Enabled by TensorRTExportConfig.fp32_accumulate_lrd (default True).
"""
from __future__ import annotations

import onnx
from onnx import TensorProto, helper

_GEMMS = ("Gemm", "MatMul")


def _lrd_layer_prefixes(model: onnx.ModelProto) -> set[str]:
    """Module paths of LRD layers: those with a ``<path>.linear_V.weight`` tensor."""
    names = {i.name for i in model.graph.initializer}
    names |= {o for n in model.graph.node for o in n.output}
    suffix = ".linear_V.weight"
    return {n[: -len(suffix)] for n in names if n.endswith(suffix)}


def _weight_source(node, producers) -> str:
    """Name of a GEMM's weight tensor, looking through a DequantizeLinear/Cast/Transpose."""
    w = node.input[1]
    seen = 0
    while w in producers and producers[w].op_type in ("DequantizeLinear", "Cast", "Transpose") and seen < 3:
        w = producers[w].input[0]
        seen += 1
    return w


def fp32_accumulate_lrd(model: onnx.ModelProto, manifest: dict | None = None) -> None:
    g = model.graph
    prefixes = _lrd_layer_prefixes(model)
    if not prefixes:
        return
    producers = {o: n for n in g.node for o in n.output}
    targets = []
    for n in g.node:
        if n.op_type not in _GEMMS or len(n.input) < 2:
            continue
        w = _weight_source(n, producers)
        if any(w == p + ".weight" or w == p + ".linear_V.weight" for p in prefixes):
            targets.append(n)
    before: dict[int, list] = {}
    after: dict[int, list] = {}
    touched = set()
    index = {id(n): i for i, n in enumerate(g.node)}
    for k, n in enumerate(targets):
        i = index[id(n)]
        for j, src in enumerate(list(n.input)):
            if not src:
                continue
            dst = f"{src}__acc32_{k}_{j}"
            before.setdefault(i, []).append(
                helper.make_node("Cast", [src], [dst], to=TensorProto.FLOAT, name=f"acc32_in_{k}_{j}_{n.name}"))
            n.input[j] = dst
        out16 = n.output[0]
        n.output[0] = f"{out16}__acc32_{k}"
        after.setdefault(i, []).append(
            helper.make_node("Cast", [n.output[0]], [out16], to=TensorProto.FLOAT16, name=f"acc32_out_{k}_{n.name}"))
        touched.add(out16)
    nodes = list(g.node)
    out = []
    for i, n in enumerate(nodes):
        out.extend(before.get(i, []))
        out.append(n)
        out.extend(after.get(i, []))
    del g.node[:]
    g.node.extend(out)
    keep = [v for v in g.value_info if v.name not in touched]
    del g.value_info[:]
    g.value_info.extend(keep)
    if manifest is not None:
        manifest.setdefault("graph_passes", []).append({"name": "fp32_accumulate_lrd", "gemms": len(targets)})


__all__ = ["fp32_accumulate_lrd"]
