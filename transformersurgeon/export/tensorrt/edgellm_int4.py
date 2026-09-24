"""
Opt-in INT4 weight-only GEMM through TensorRT Edge-LLM's plugin.

Plain TensorRT (10.16) has no fast single-token INT4 GEMV: a DequantizeLinear
-> Gemm pair either becomes a generic fused tensor-core GEMM or, for large K,
dequantizes the whole weight to fp16 on every step, so INT4 decode runs at fp16
speed. TensorRT Edge-LLM (Apache-2.0, NVIDIA) ships Int4GroupwiseGemmPluginV2,
which computes ``(nibble - 8) * scale[group, n]`` -- exactly tsurgeon's
symmetric INT4 (q in [-8, 7] stored as q + 8), with per-channel scales tiled
over groups unchanged.

edgellm_int4_pass rewrites every eligible INT4 ``DequantizeLinear -> Gemm``
(from export/onnx/quantization.py) into that plugin node; layers the plugin
cannot take (N or K not a multiple of 64/128, or scales that are not constant
over 128-wide groups) keep the plain Q/DQ path. Finer DQ blocks whose scales
repeat within each 128 group -- tsurgeon's tiled per-channel scales -- are
regrouped exactly. The weight repack uses Edge-LLM's own ``repack_to_cutedsl_fragment`` so
the layout always matches the plugin build it ships with. The resulting engine
needs ``libNvInfer_edgellm_plugin.so`` (built from the same Edge-LLM release)
at build and run time; the manifest records this under "plugins".
"""

import os

import numpy as np

PLUGIN_NAME = "edgellm"
PLUGIN_ENV = "EDGELLM_PLUGIN_PATH"
_DOMAIN = "trt_edgellm"
_OP = "Int4GroupwiseGemmPluginV2"
_GROUP = 128


def _edgellm_repack():
    try:
        from tensorrt_edgellm.checkpoint.repacking import repack_to_cutedsl_fragment
        import tensorrt_edgellm
    except ImportError as e:
        raise ImportError(
            "int4_backend='edgellm_plugin' needs the TensorRT Edge-LLM Python package "
            "(https://github.com/NVIDIA/TensorRT-Edge-LLM, same release as the plugin library)."
        ) from e
    return repack_to_cutedsl_fragment, getattr(tensorrt_edgellm, "__version__", "unknown")


def _plugin_group_scales(scale: np.ndarray, block: int) -> np.ndarray | None:
    """(N, K/block) DQ scales -> (N, K/128) plugin scales, or None if a 128
    group mixes different scales (then the plugin cannot represent it)."""
    per_group = _GROUP // block
    grouped = scale.reshape(scale.shape[0], -1, per_group)
    if not np.all(grouped == grouped[..., :1]):
        return None
    return np.ascontiguousarray(grouped[..., 0])


def edgellm_int4_pass(model, manifest: dict) -> None:
    """Graph pass (ONNXExportConfig.graph_passes): INT4 DQ->Gemm -> Edge-LLM plugin."""
    from onnx import TensorProto, helper, numpy_helper

    repack, version = _edgellm_repack()
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    rewritten, skipped = [], []
    for dq in [n for n in graph.node if n.op_type == "DequantizeLinear"]:
        q_init = inits.get(dq.input[0])
        users = consumers.get(dq.output[0], [])
        if q_init is None or q_init.data_type != TensorProto.INT4 or len(users) != 1:
            continue
        gemm = users[0]
        attrs = {a.name: helper.get_attribute_value(a) for a in gemm.attribute}
        block = next((helper.get_attribute_value(a) for a in dq.attribute if a.name == "block_size"), None)
        n_out, k_in = q_init.dims
        ok = (gemm.op_type == "Gemm" and gemm.input[1] == dq.output[0] and attrs.get("transB", 0) == 1
              and not attrs.get("transA", 0) and attrs.get("alpha", 1.0) == 1.0 and attrs.get("beta", 1.0) == 1.0
              and block is not None and _GROUP % block == 0 and n_out % 64 == 0 and k_in % _GROUP == 0)
        scale = _plugin_group_scales(numpy_helper.to_array(inits[dq.input[1]]), block) if ok else None
        if scale is None:
            skipped.append(dq.output[0])
            continue

        q = numpy_helper.to_array(q_init).astype(np.int64)                  # (N, K) in [-8, 7]
        base = dq.output[0]
        packed = numpy_helper.from_array(repack(q + 8), base + "_edgellm_q")
        scales = numpy_helper.from_array(np.ascontiguousarray(scale.T), base + "_edgellm_s")  # (K/128, N)
        axis0 = numpy_helper.from_array(np.array([0], np.int64), base + "_edgellm_ax")

        x, y = gemm.input[0], gemm.output[0]
        new_nodes = [
            helper.make_node("Unsqueeze", [x, axis0.name], [base + "_x3"]),   # plugin takes [batch, seq, K]
            helper.make_node(_OP, [base + "_x3", packed.name, scales.name], [base + "_y3"], domain=_DOMAIN,
                             gemm_n=int(n_out), gemm_k=int(k_in), group_size=_GROUP, name=base + "_" + _OP),
        ]
        if len(gemm.input) > 2 and gemm.input[2]:
            new_nodes += [helper.make_node("Squeeze", [base + "_y3", axis0.name], [base + "_y2"]),
                          helper.make_node("Add", [base + "_y2", gemm.input[2]], [y])]
        else:
            new_nodes.append(helper.make_node("Squeeze", [base + "_y3", axis0.name], [y]))

        idx = list(graph.node).index(gemm)
        graph.node.remove(gemm)
        for offset, node in enumerate(new_nodes):
            graph.node.insert(idx + offset, node)
        graph.node.remove(dq)
        for name in (dq.input[0], dq.input[1]):
            graph.initializer.remove(inits.pop(name))
        graph.initializer.extend([packed, scales, axis0])
        rewritten.append(base)

    if rewritten:
        if not any(op.domain == _DOMAIN for op in model.opset_import):
            model.opset_import.append(helper.make_opsetid(_DOMAIN, 1))
        manifest.setdefault("plugins", {})[PLUGIN_NAME] = {
            "library_env": PLUGIN_ENV, "op": _OP, "edgellm_version": version,
            "layers": len(rewritten), "fallback_layers": len(skipped),
        }


def resolve_plugin_libraries(manifest: dict, explicit: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Plugin libraries an engine for ``manifest`` needs (explicit paths win)."""
    if explicit or PLUGIN_NAME not in manifest.get("plugins", {}):
        return tuple(explicit)
    path = os.environ.get(PLUGIN_ENV)
    if not path:
        raise RuntimeError(
            f"This export uses TensorRT Edge-LLM's INT4 plugin: pass its library "
            f"(libNvInfer_edgellm_plugin.so) explicitly or set {PLUGIN_ENV}."
        )
    return (path,)


__all__ = ["PLUGIN_ENV", "edgellm_int4_pass", "resolve_plugin_libraries"]
