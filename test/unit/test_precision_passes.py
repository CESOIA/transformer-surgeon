"""fp32_accumulate_lrd (export/tensorrt/precision_passes.py): only GEMMs of LRD layers
get fp32 operands/accumulation; weights stay fp16 initializers; dense GEMMs untouched."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from transformersurgeon.export.backends.tensorrt.precision_passes import fp32_accumulate_lrd


def _model():
    f16 = lambda name, shape: numpy_helper.from_array(np.zeros(shape, np.float16), name)
    inits = [f16("a.linear_V.weight", (4, 8)), f16("a.weight", (8, 4)), f16("b.weight", (8, 8))]
    nodes = [
        helper.make_node("Gemm", ["x", "a.linear_V.weight"], ["h"], transB=1, name="a_v"),
        helper.make_node("Gemm", ["h", "a.weight"], ["y"], transB=1, name="a_u"),
        helper.make_node("Gemm", ["y", "b.weight"], ["z"], transB=1, name="b"),
    ]
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8])],
                          [helper.make_tensor_value_info("z", TensorProto.FLOAT16, [1, 8])], inits)
    return helper.make_model(g)


def test_only_lrd_gemms_accumulate_in_fp32():
    m, man = _model(), {}
    fp32_accumulate_lrd(m, man)
    assert man["graph_passes"] == [{"name": "fp32_accumulate_lrd", "gemms": 2}]
    gemms = {n.name: n for n in m.graph.node if n.op_type == "Gemm"}
    assert all("__acc32_" in i for i in gemms["a_v"].input) and all("__acc32_" in i for i in gemms["a_u"].input)
    assert list(gemms["b"].input) == ["y", "b.weight"]
    assert all(i.data_type == TensorProto.FLOAT16 for i in m.graph.initializer)  # weights stay fp16
    casts = [n for n in m.graph.node if n.op_type == "Cast"]
    assert sum(helper.get_attribute_value(c.attribute[0]) == TensorProto.FLOAT16 for c in casts) == 2
    onnx.checker.check_model(m)
