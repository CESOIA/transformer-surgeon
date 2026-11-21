"""Encoder export with the sdpa attention atom (4-D ONNX Attention, no GQA flag on equal heads)."""
import pytest
import torch

from _helpers import capabilities as caps
from _helpers.model_factory import FAMILIES
from transformersurgeon.export import export_to_backend


@caps.requires_onnx
def test_vit_encoder_exports_with_the_sdpa_atom(tmp_path):
    import onnx
    from transformersurgeon.export.backends import ONNXExportConfig

    torch.manual_seed(0)
    model = FAMILIES["vit"].build().eval()
    result = export_to_backend(model, ONNXExportConfig(output_path=str(tmp_path / "model.onnx"),
                                                       convert_options={"attn_impl": "sdpa"}))
    graph = onnx.load(result.onnx_path, load_external_data=False).graph
    assert sum(n.op_type == "Attention" for n in graph.node) == 2      # one fused op per block
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        want = model(pixel_values=x).logits
    ort = pytest.importorskip("onnxruntime")
    try:
        sess = ort.InferenceSession(result.onnx_path, providers=["CPUExecutionProvider"])
    except Exception as exc:   # an onnxruntime without opset-23 Attention
        pytest.skip(f"onnxruntime cannot load the graph: {exc}")
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(got), want, rtol=1e-3, atol=1e-3)
