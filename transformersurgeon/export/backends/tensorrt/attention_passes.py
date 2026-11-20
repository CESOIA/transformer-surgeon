"""TensorRT-target ONNX rewrite (runs as an ONNXExportConfig.graph_passes entry).

fp32_attention_scores: run the attention score path  MatMul(q, kᵀ) -> Add(mask) -> Softmax
in float32 and cast the probabilities back to the model dtype (float16 or bfloat16).
Unmasked attention (encoders: MatMul -> Softmax, optionally through a Mul/Div by the
score scale) gets the same treatment. A float32 model is left unchanged.

Why: models with large k-projection biases (up to ~317 in a measured 1.5B decoder)
make some tokens produce
attention logits of ~1e4 whose fp16 partial dot-product sums exceed 65504.
TensorRT 10.16 fuses this path into a kernel that accumulates in fp16 -> inf -> NaN
logits (token 659 at position 0 is a reproducer). Eager PyTorch (fp32 accumulation)
is unaffected. Enabled by TensorRTExportConfig.fp32_attention_scores (default True).
"""
from __future__ import annotations

import onnx
from onnx import TensorProto, helper


def _producers(g):
    return {o: n for n in g.node for o in n.output}


def _model_float_type(g) -> int:
    """ONNX element type of the model's floating-point weights (the compute dtype)."""
    counts: dict[int, int] = {}
    for init in g.initializer:
        if init.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16):
            counts[init.data_type] = counts.get(init.data_type, 0) + 1
    return max(counts, key=counts.get) if counts else TensorProto.FLOAT16


def _score_path(sm, prod):
    """``(matmul, between)`` for a Softmax fed by MatMul [-> Add | Mul | Div], else ``None``."""
    src = prod.get(sm.input[0])
    if src is None:
        return None
    if src.op_type == "MatMul":
        return src, None
    if src.op_type not in ("Add", "Mul", "Div"):
        return None
    mm = next((prod.get(i) for i in src.input if prod.get(i) is not None and prod[i].op_type == "MatMul"), None)
    return (mm, src) if mm is not None else None


def fp32_attention_scores(model: onnx.ModelProto, manifest: dict | None = None) -> None:
    g = model.graph
    float_type = _model_float_type(g)
    if float_type == TensorProto.FLOAT:
        return  # already float32
    prod = _producers(g)
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    new_nodes: dict[int, list] = {}  # insert-before index -> nodes
    touched = set()
    count = 0
    for sm in [n for n in g.node if n.op_type == "Softmax"]:
        path = _score_path(sm, prod)
        if path is None:
            continue
        mm, add = path
        k = count
        casts = []
        operands = [(mm, range(len(mm.input)))]
        if add is not None:
            operands.append((add, [j for j, i in enumerate(add.input) if i != mm.output[0]]))
        for node, idxs in operands:
            for j in idxs:
                src = node.input[j]
                dst = f"{src}__f32_{k}_{node.name}_{j}"
                casts.append(helper.make_node("Cast", [src], [dst], to=TensorProto.FLOAT, name=f"cast_f32_{k}_{node.name}_{j}"))
                node.input[j] = dst
        # softmax output back to the model dtype for the consumers
        out16 = sm.output[0]
        out32 = f"{out16}__f32_{k}"
        sm.output[0] = out32
        back = helper.make_node("Cast", [out32], [out16], to=float_type, name=f"cast_f16_{k}_{sm.name}")
        touched.update([mm.output[0], out16] + ([add.output[0]] if add is not None else []))
        idx_mm = list(g.node).index(mm)
        new_nodes.setdefault(idx_mm, []).extend(casts)
        new_nodes.setdefault(list(g.node).index(sm) + 1, []).append(back)
        count += 1
    if not count:
        if any(n.op_type == "AttentionPlugin" for n in g.node):
            return  # fused plugin attention (io_paged): accumulates in fp32 itself
        raise RuntimeError("fp32_attention_scores: no MatMul[->Add|Mul|Div]->Softmax pattern found")
    nodes = list(g.node)
    out = []
    for i, n in enumerate(nodes):
        out.extend(new_nodes.get(i, []))
        out.append(n)
    out.extend(new_nodes.get(len(nodes), []))
    del g.node[:]
    g.node.extend(out)
    # drop stale fp16 type annotations on retyped intermediates
    keep = [v for v in g.value_info if v.name not in touched]
    del g.value_info[:]
    g.value_info.extend(keep)
    if manifest is not None:
        manifest.setdefault("graph_passes", []).append({"name": "fp32_attention_scores", "patterns": count})
