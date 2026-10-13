"""QNN quant_scope="all" + quantize_float_model: an uncompressed model is still
lowered as a fixed-point graph (16a16w everywhere), the activation-only reference
point for isolating weight compression from HTP's 16-bit activations."""
from __future__ import annotations

import pytest
import torch

from _helpers import capabilities as caps
from _helpers.model_factory import FAMILIES

pytestmark = pytest.mark.unit


def _float_qwen2():
    torch.manual_seed(0)
    return FAMILIES["qwen2"].build().eval()


def _components(model, opts):
    from transformersurgeon.utils import convert_for_export

    conv = convert_for_export(model, options=opts)
    return {"embedding": model.get_input_embeddings(), "decoder": conv["text"],
            "final_layer": model.lm_head, "config": model.config}


def test_requires_scope_all(tmp_path):
    pytest.importorskip("executorch")
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.executorch_exporters.qnn import QNNExportConfig, is_qnn_available

    if not is_qnn_available():
        pytest.skip("QNN toolchain not available")
    opts = {"use_sdpa": False, "max_cache_len": 32, "masked_softmax": True}
    cfg = QNNExportConfig(output_path=str(tmp_path / "m.pte"), backend="qnn", soc_model="SM8850", max_seq_len=32,
                          convert_options=opts, quant_scope="linear", quantize_float_model=True,
                          run_weight_mismatch_check=False, verbose=False)
    with pytest.raises(ValueError, match="quantize_float_model"):
        export_to_backend(_components(_float_qwen2(), opts), config=cfg)


@caps.requires_qnn
@pytest.mark.qnn
def test_float_model_lowered_fixed_point(tmp_path):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.common import LLMWrapper
    from transformersurgeon.export.executorch_exporters.qnn import QNNExportConfig

    toks = [5, 17, 99, 3, 42, 200, 7, 64]
    opts = {"use_sdpa": False, "max_cache_len": 32, "masked_softmax": True}

    def run(m):
        outs = []
        with torch.no_grad():
            for p, t in enumerate(toks):
                o = m(torch.tensor([t]), torch.tensor([p]))
                outs.append((o[0] if isinstance(o, (tuple, list)) else o).float().reshape(-1))
        return torch.stack(outs)

    model = _float_qwen2()
    comps = _components(model, opts)
    ref = run(LLMWrapper(comps["embedding"], comps["decoder"], comps["final_layer"]).eval())
    cfg = QNNExportConfig(output_path=str(tmp_path / "m.pte"), backend="qnn", soc_model="SM8850", max_seq_len=32,
                          convert_options=opts, quant_scope="all", quantize_float_model=True,
                          keep_reference_module=True,
                          calibration_token_ids=[[(i * 37 + j * 11) % 256 for j in range(16)] for i in range(4)],
                          run_weight_mismatch_check=False, verbose=False)
    result = export_to_backend(_components(_float_qwen2(), opts), config=cfg)
    gm = result.reference_module
    targets = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
    assert any("quantize_per_tensor" in t for t in targets), "no activation fake-quant in the reference"
    assert any("quantize_per_channel" in t for t in targets), "no 16-bit weight fake-quant in the reference"
    fq = run(gm)
    assert bool((fq.argmax(-1) == ref.argmax(-1)).all()) and (fq - ref).abs().max().item() < 0.1
