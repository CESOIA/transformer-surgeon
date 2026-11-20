"""Exports with the options ported from the compression-benchmark branch (bench-frozen-2026-11): the XNNPACK int8
fused-attention cache, and QNN quantised cache I/O and the attention-matmul 8-bit cache."""
import os

import pytest
import torch

from _helpers import capabilities as caps
from _helpers import model_factory

CACHE = 16


def _model():
    torch.manual_seed(0)
    return model_factory.kv_model(torch.float32)


@caps.requires_custom_sdpa
def test_xnnpack_custom_quantized_sdpa_matches_the_float_kernel_and_exports(tmp_path):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import XNNPACKExportConfig
    from transformersurgeon.export.convert import convert_for_export

    model = _model()
    ref = convert_for_export(model, options={"attn_impl": "custom_sdpa", "max_cache_len": CACHE})["text"]
    q8 = convert_for_export(model, options={"attn_impl": "custom_quantized_sdpa", "max_cache_len": CACHE})["text"]
    assert q8.blocks[0].attn.custom_sdpa_key_cache.dtype == torch.int8
    torch.manual_seed(1)
    x = torch.randn(1, model.config.hidden_size)
    for pos in range(4):
        a, b = ref(x, pos_id=torch.tensor([pos])), q8(x, pos_id=torch.tensor([pos]))
        assert (a - b).abs().max() <= 0.05 * a.abs().max()   # int8 per token and head: close, not equal
    cfg = XNNPACKExportConfig(output_path=str(tmp_path / "model.pte"), max_seq_len=CACHE,
                              run_weight_mismatch_check=False,
                              convert_options={"attn_impl": "custom_quantized_sdpa", "max_cache_len": CACHE})
    export_to_backend(model, cfg)
    assert os.path.getsize(cfg.output_path) > 0


def test_qnn_options_declare_their_requirements():
    pytest.importorskip("torchao")
    from transformersurgeon.export.backends import QNNExportConfig
    from transformersurgeon.export.operations import check_config_choices, check_config_requirements

    good = QNNExportConfig(output_path="x", quant_scope="all", quantized_kv_io=True)
    check_config_choices("qnn", good)
    check_config_requirements("qnn", good, {"cache_impl": "mutable"})
    with pytest.raises(ValueError, match=r"quantized_kv_io=True requires quant_scope='all'"):
        check_config_requirements("qnn", QNNExportConfig(output_path="x", quantized_kv_io=True))
    with pytest.raises(ValueError, match="requires cache_impl in"):
        check_config_requirements("qnn", good, {"cache_impl": "io_scatter"})
    matmuls = QNNExportConfig(output_path="x", quant_scope="all", kv_cache_bits=8,
                              kv_cache_8bit_scheme="attention_matmuls")
    check_config_requirements("qnn", matmuls, {"cache_impl": "mutable"})
    with pytest.raises(ValueError, match="kv_cache_bits=8"):
        check_config_requirements("qnn", QNNExportConfig(output_path="x", quant_scope="all", kv_cache_bits=(16, 8),
                                                         kv_cache_8bit_scheme="attention_matmuls"))


def test_lifted_cache_buffers_are_recognised_by_their_declared_names():
    pytest.importorskip("torchao")
    from transformersurgeon.export.backends.qnn.passes import _kv_cache_buffer_kind

    assert _kv_cache_buffer_kind("b_decoder_blocks_0_attn_key_cache") == "key"
    assert _kv_cache_buffer_kind("b_decoder_blocks_11_attn_value_cache") == "value"
    assert _kv_cache_buffer_kind("b_decoder_blocks_0_attn_key_cache_scale") is None
    assert _kv_cache_buffer_kind("key_caches_0") is None


@caps.requires_qnn
@pytest.mark.parametrize("bits, scheme", [(16, "cache_writes"), (8, "cache_writes"), (8, "attention_matmuls")])
def test_qnn_quantized_kv_io_export(tmp_path, bits, scheme):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends import QNNExportConfig

    cfg = QNNExportConfig(output_path=str(tmp_path / "model.pte"), max_seq_len=CACHE, quant_scope="all",
                          quantize_float_model=True, quantized_kv_io=True, kv_cache_bits=bits,
                          kv_cache_8bit_scheme=scheme, run_weight_mismatch_check=False,
                          convert_options={"max_cache_len": CACHE})
    export_to_backend(_model(), cfg)
    assert os.path.getsize(cfg.output_path) > 0
