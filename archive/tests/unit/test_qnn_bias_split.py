"""QNN quant_scope="all": float (16a16w) linears keep 16-bit weights but their bias
moves to a separate add (QNNExportConfig.split_float_linear_bias).

With 16-bit weights the int32 bias scale (act_scale x weight_scale) is so small
that Qwen2.5's q/k biases overflow it and get clipped: an MLP-only quantized
export then has broken attention although attention was meant to stay ~float.
"""
from __future__ import annotations

import pytest
import torch

from _helpers import capabilities as caps
from _helpers.model_factory import FAMILIES

pytestmark = pytest.mark.unit


def _mlp_w8_qwen2(big_bias: bool):
    from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager

    torch.manual_seed(0)
    model = FAMILIES["qwen2"].build().eval()
    if big_bias:  # Qwen2.5-like q/k biases, far above the 16a16w int32 bias range
        with torch.no_grad():
            for layer in model.model.layers:
                layer.self_attn.k_proj.bias[:4] = 30.0
                layer.self_attn.q_proj.bias[:4] = -20.0
    mgr = Qwen2CompressionSchemesManager(model)
    mgr.set("quantization", "precision", 8, criteria="mlp")
    mgr.set("quantization", "granularity", "per_channel", criteria="mlp")
    mgr.apply(hard=True)
    return model


def _wrapper(model):
    from transformersurgeon.export import LLMWrapper
    from transformersurgeon.export.convert import convert_for_export

    conv = convert_for_export(model, options={"use_sdpa": False, "max_cache_len": 32, "masked_softmax": True})
    return LLMWrapper(model.get_input_embeddings(), conv["text"], model.lm_head).eval()


def test_split_only_float_biased_linears_and_keeps_outputs():
    pytest.importorskip("torchao")
    pytest.importorskip("executorch")
    from torch.export import export as torch_export

    from transformersurgeon.export import extract_layer_quant_info
    from transformersurgeon.export.backends.qnn.passes import (
        _decompose_wrap_with_set_grad_enabled,
        _split_float_linear_bias,
    )

    wrapper = _wrapper(_mlp_w8_qwen2(big_bias=False))
    layer_info = extract_layer_quant_info(wrapper)
    assert layer_info and all(".mlp." in n for n in layer_info)

    inputs = (torch.tensor([3]), torch.tensor([0]))
    with torch.inference_mode():
        gm = torch_export(wrapper, inputs).module()
    _decompose_wrap_with_set_grad_enabled(gm)
    with torch.no_grad():
        before = gm(*inputs).clone()

    split = _split_float_linear_bias(gm, frozenset(layer_info))
    # q/k/v of both layers have a bias and are not quantized; o_proj, MLP and lm_head have none.
    assert sorted(split) == sorted(f"decoder.blocks.{i}.attn.{p}" for i in range(2) for p in ("q_proj", "k_proj", "v_proj"))
    linears = [n for n in gm.graph.nodes if n.op == "call_function" and n.target == torch.ops.aten.linear.default]
    for n in linears:
        names = [q for q, _ in n.meta.get("nn_module_stack", {}).values()]
        if any(q in layer_info for q in names):
            continue  # quantized linears untouched (bias, if any, stays derived from the weight scale)
        assert len(n.args) < 3 or n.args[2] is None, n.name

    for name, buf in gm.named_buffers():  # same (empty) KV state for the second run
        if name.endswith("_cache"):
            buf.zero_()
    with torch.no_grad():
        after = gm(*inputs)
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)


@caps.requires_qnn
@pytest.mark.qnn
@pytest.mark.parametrize("split", [False, True])
def test_qnn_mlp_only_fake_quant_tracks_float(tmp_path, split):
    """End to end through the QNN exporter (PT2E fake-quant reference + HTP lowering)."""
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends.qnn import QNNExportConfig

    toks = [5, 17, 99, 3, 42, 200, 7, 64]

    def run(m):
        outs = []
        with torch.no_grad():
            for p, t in enumerate(toks):
                o = m(torch.tensor([t]), torch.tensor([p]))
                outs.append((o[0] if isinstance(o, (tuple, list)) else o).float().reshape(-1))
        return torch.stack(outs)

    ref = run(_wrapper(_mlp_w8_qwen2(big_bias=True)))
    model = _mlp_w8_qwen2(big_bias=True)
    opts = {"use_sdpa": False, "max_cache_len": 32, "masked_softmax": True}
    from transformersurgeon.export.convert import convert_for_export
    conv = convert_for_export(model, options=opts)
    comps = {"embedding": model.get_input_embeddings(), "decoder": conv["text"],
             "final_layer": model.lm_head, "config": model.config}
    cfg = QNNExportConfig(output_path=str(tmp_path / "m.pte"), backend="qnn", soc_model="SM8850", max_seq_len=32,
                          convert_options=opts, quant_scope="all", keep_reference_module=True,
                          split_float_linear_bias=split,
                          calibration_token_ids=[[(i * 37 + j * 11) % 256 for j in range(16)] for i in range(4)],
                          run_weight_mismatch_check=False, verbose=False)
    result = export_to_backend(comps, config=cfg)
    fq = run(result.reference_module)
    max_err = (fq - ref).abs().max().item()
    if split:
        assert max_err < 0.1 and bool((fq.argmax(-1) == ref.argmax(-1)).all()), max_err
    else:  # pins the failure mode the split fixes: clipped q/k biases
        assert max_err > 0.3, max_err
