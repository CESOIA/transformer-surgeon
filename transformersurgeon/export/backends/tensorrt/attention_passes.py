"""TensorRT-target ONNX rewrite (runs as an ONNXExportConfig.graph_passes entry).

fp32_attention_scores: run the attention score path  MatMul(q, kᵀ) -> Add(mask) -> Softmax
in float32 and cast the probabilities back to the model dtype.

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


def fp32_attention_scores(model: onnx.ModelProto, manifest: dict | None = None) -> None:
    g = model.graph
    prod = _producers(g)
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    new_nodes: dict[int, list] = {}  # insert-before index -> nodes
    touched = set()
    count = 0
    for sm in [n for n in g.node if n.op_type == "Softmax"]:
        add = prod.get(sm.input[0])
        if add is None or add.op_type != "Add":
            continue
        mm = next((prod.get(i) for i in add.input if prod.get(i) is not None and prod[i].op_type == "MatMul"), None)
        if mm is None:
            continue
        k = count
        casts = []
        for node, idxs in ((mm, range(len(mm.input))), (add, [j for j, i in enumerate(add.input) if i != mm.output[0]])):
            for j in idxs:
                src = node.input[j]
                dst = f"{src}__f32_{k}_{node.name}_{j}"
                casts.append(helper.make_node("Cast", [src], [dst], to=TensorProto.FLOAT, name=f"cast_f32_{k}_{node.name}_{j}"))
                node.input[j] = dst
        # softmax output back to fp16 for the consumers
        out16 = sm.output[0]
        out32 = f"{out16}__f32_{k}"
        sm.output[0] = out32
        back = helper.make_node("Cast", [out32], [out16], to=TensorProto.FLOAT16, name=f"cast_f16_{k}_{sm.name}")
        touched.update([mm.output[0], add.output[0], out16])
        idx_mm = list(g.node).index(mm)
        new_nodes.setdefault(idx_mm, []).extend(casts)
        new_nodes.setdefault(list(g.node).index(sm) + 1, []).append(back)
        count += 1
    if not count:
        if any(n.op_type == "AttentionPlugin" for n in g.node):
            return  # fused plugin attention (io_paged): accumulates in fp32 itself
        raise RuntimeError("fp32_attention_scores: no MatMul->Add->Softmax pattern found")
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
