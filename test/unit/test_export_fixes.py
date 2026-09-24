"""Regressions for fixes made alongside the ONNX/TensorRT path:
RoPE base and RMSNorm eps read from the HF config, in-range calibration
positions, and the per-layer export manifest used by runners.
"""
from __future__ import annotations

import types
import warnings

import pytest
import torch

from _helpers.model_factory import _qwen2


def test_rope_theta_from_config():
    from transformersurgeon.blocks.rope import rope_theta_from_config

    ns = types.SimpleNamespace
    assert rope_theta_from_config(ns(rope_parameters={"rope_theta": 10000.0, "rope_type": "default"})) == 10000.0
    assert rope_theta_from_config(ns(rope_theta=5e5)) == 5e5          # pre-transformers-5 configs
    assert rope_theta_from_config(ns()) == 1e6                         # historical default
    with pytest.warns(UserWarning, match="llama3"):
        rope_theta_from_config(ns(rope_parameters={"rope_theta": 5e5, "rope_type": "llama3"}))


def test_converted_decoder_uses_config_eps_and_rope_theta():
    from transformersurgeon.blocks.rope import precompute_rope_cos_sin_half, precompute_rope_inv_freqs
    from transformersurgeon.utils import convert_for_export

    model = _qwen2().eval()
    model.config.rms_norm_eps = 1e-6
    model.config.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}
    decoder = convert_for_export(model, options={})["text"]

    assert decoder.norm.eps == 1e-6
    assert all(b.norm_in.eps == 1e-6 and b.norm_out.eps == 1e-6 for b in decoder.blocks)
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cos, _ = precompute_rope_cos_sin_half(precompute_rope_inv_freqs(head_dim=head_dim, base=10000.0),
                                          decoder.max_cache_len, start_pos=0)
    torch.testing.assert_close(decoder.rope_cos.float(), cos.float())


def test_random_calibration_positions_stay_in_range():
    """No tokenizer -> random fallback; positions must be valid cache slots."""
    from transformersurgeon.export.common import calibrate_pt2e_observers

    seen = []
    torch.manual_seed(0)
    calibrate_pt2e_observers(lambda ids, pos, *caches: seen.append(int(pos)),
                             types.SimpleNamespace(vocab_size=10), types.SimpleNamespace(max_seq_len=4),
                             example_inputs=(torch.tensor([1]), torch.tensor([0])))
    assert seen and max(seen) < 4 and min(seen) >= 0


@pytest.mark.parametrize("cache_impl", ["io_scatter", "io_inplace"])
def test_manifest_describes_per_layer_caches(tmp_path, cache_impl):
    from transformersurgeon.export.common import (
        build_llm_manifest,
        build_wrapper,
        zero_caches_from_manifest,
    )
    from transformersurgeon.utils import convert_for_export

    model = _qwen2().eval()
    decoder = convert_for_export(model, options={"cache_impl": cache_impl, "max_cache_len": 16})["text"]
    wrapper = build_wrapper({"embedding": model.get_input_embeddings(), "decoder": decoder,
                             "final_layer": model.lm_head}, model_config=model.config)
    manifest = build_llm_manifest(wrapper, model.config, backend="xnnpack", artifact=str(tmp_path / "m.pte"))

    assert manifest["cache_io"] and manifest["cache_inplace"] == (cache_impl == "io_inplace")
    keys, values = zero_caches_from_manifest(manifest)
    assert [tuple(k.shape) for k in keys] == [b.attn.cache_shape("key") for b in decoder.blocks]
    assert [tuple(v.shape) for v in values] == [b.attn.cache_shape("value") for b in decoder.blocks]
    assert keys[0].dtype == decoder.blocks[0].attn.key_cache.dtype  # config dtype may be None
    # The wrapper accepts exactly these caches.
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrapper(torch.tensor([3]), torch.tensor([0]), keys, values)
