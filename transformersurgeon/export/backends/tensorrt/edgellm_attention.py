"""
Fused decode attention through TensorRT Edge-LLM's AttentionPlugin (XQA).

Plain TensorRT (10.16) has no usable fused decode attention for a static KV
cache: the exported MatMul/Softmax/MatMul path reads every cache slot each step
(cost grows with max_cache_len, not with the context), and IAttention's fused
``_gemv_mha_v2`` kernel fails to build with any mask (Myelin internal error).
Edge-LLM's ``AttentionPlugin`` applies RoPE, writes the new K/V into a paged
pool in place and runs the XQA decode kernel (NVRTC-compiled for the build GPU
and serialized into the engine) over the ``context_lengths`` filled slots only.

A model converted with ``cache_impl="io_paged"`` carries one
``tsurgeon::paged_attention`` op per layer (``blocks/paged_attention_ops.py``,
whose eager implementation follows the plugin's contract); this module maps it
to the plugin node. Only single-token decode is supported (XQA): prefill runs
token by token (max_input_len = 1). The engine needs
``libNvInfer_edgellm_plugin.so`` at build and run time; the manifest records it
under "plugins".
"""

from .edgellm_int4 import PLUGIN_ENV, _DOMAIN

PLUGIN_KEY = "edgellm_attention"
_OP = "AttentionPlugin"


def _register_schema() -> None:
    """onnxscript only builds nodes with a registered schema: register the subset
    of the plugin's inputs/attributes this export uses (Edge-LLM's own exporter
    registers the full one; whichever comes first wins)."""
    import onnx
    from onnx.defs import OpSchema

    try:
        onnx.defs.get_schema(_OP, 1, _DOMAIN)
        return
    except onnx.defs.SchemaError:
        pass
    param = OpSchema.FormalParameter
    schema = OpSchema(
        _OP, _DOMAIN, 1,
        inputs=[param("qkv", "T"), param("past_key_value", "T"), param("context_lengths", "tensor(int32)"),
                param("rope_rotary_cos_sin", "tensor(float)"), param("kvcache_start_index", "tensor(int32)"),
                param("kv_page_table", "tensor(int32)")],
        outputs=[param("attn_output", "T"), param("present_key_value", "T")],
        type_constraints=[("T", ["tensor(float16)"], "fp16 activations and KV cache")],
        attributes=[OpSchema.Attribute(n, OpSchema.AttrType.INT, n) for n in
                    ("num_q_heads", "num_kv_heads", "head_size", "enable_tree_attention", "enable_fp8_kv_cache")]
        + [OpSchema.Attribute("attention_scale", OpSchema.AttrType.FLOAT, "QK^T multiplier")],
    )
    onnx.defs.register_schema(schema)


def _attention_plugin(qkv, kv_pool, context_lengths, rope_cos_sin, kvcache_start_index, kv_page_table,
                      num_q_heads, num_kv_heads, head_size, attention_scale):
    from onnxscript import values

    _register_schema()
    return values.Opset(_DOMAIN, 1).AttentionPlugin(
        qkv, kv_pool, context_lengths, rope_cos_sin, kvcache_start_index, kv_page_table,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_size=head_size,
        attention_scale=attention_scale, enable_tree_attention=0, enable_fp8_kv_cache=0,
        _outputs=2,
    )


def attention_translations() -> dict:
    from ....blocks.attention.kernels import paged  # noqa: F401  (registers torch.ops.tsurgeon.paged_attention)
    import torch

    return {torch.ops.tsurgeon.paged_attention.default: _attention_plugin}


def edgellm_attention_pass(model, manifest: dict) -> None:
    """Graph pass: record the plugin dependency (and the custom opset) when the
    export contains AttentionPlugin nodes."""
    from onnx import helper

    nodes = [n for n in model.graph.node if n.domain == _DOMAIN and n.op_type == _OP]
    if not nodes:
        return
    if not any(op.domain == _DOMAIN for op in model.opset_import):
        model.opset_import.append(helper.make_opsetid(_DOMAIN, 1))
    manifest.setdefault("plugins", {})[PLUGIN_KEY] = {
        "library_env": PLUGIN_ENV, "op": _OP, "layers": len(nodes), "kernel": "XQA decode",
    }


__all__ = ["PLUGIN_KEY", "attention_translations", "edgellm_attention_pass"]
