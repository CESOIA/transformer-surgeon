"""RoPE variants of the converted decoder (blocks/rope.py ROPE_FREQUENCIES).

The indexing file names the variant (``'rope_type'``: fixed, or ``'auto'`` =
the config's). Llama 3.x needs the ``llama3`` frequency scaling; plain RoPE
drifts from HF with position.
"""
from __future__ import annotations

import pytest
import torch

import transformersurgeon as T
from transformersurgeon.blocks.positions.rope import (
    precompute_rope_inv_freqs_for,
    resolve_rope_type,
    rope_parameters_from_config,
)
from transformersurgeon.export import LLMWrapper
from transformersurgeon.export.convert import convert_for_export

LLAMA3_SCALING = {"rope_type": "llama3", "factor": 4.0, "low_freq_factor": 1.0,
                  "high_freq_factor": 4.0, "original_max_position_embeddings": 16}


def _tiny_llama(rope_scaling=None):
    torch.manual_seed(0)
    kw = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
              tie_word_embeddings=False, rope_theta=500000.0)
    if rope_scaling is not None:
        kw["rope_scaling"] = dict(rope_scaling)
    return T.LlamaForCausalLMCompress(T.LlamaConfigCompress(**kw)).eval()


def test_llama3_inv_freq_matches_hf():
    model = _tiny_llama(LLAMA3_SCALING)
    params = rope_parameters_from_config(model.config)
    assert params["rope_type"] == "llama3"
    ours = precompute_rope_inv_freqs_for("llama3", 16, params)
    torch.testing.assert_close(ours, model.model.rotary_emb.inv_freq.float(), rtol=1e-6, atol=0)
    assert not torch.allclose(ours, precompute_rope_inv_freqs_for("default", 16, params))


@pytest.mark.parametrize("scaling", [None, LLAMA3_SCALING], ids=["default", "llama3"])
def test_converted_llama_matches_hf(scaling):
    model = _tiny_llama(scaling)
    n = 48
    ids = torch.randint(0, 256, (n,))
    with torch.no_grad():
        ref = model(ids[None]).logits[0]
    dec = convert_for_export(model, options={"use_sdpa": False, "max_cache_len": n})["text"]
    assert dec.config.rope_type == ("llama3" if scaling else "default")
    w = LLMWrapper(model.get_input_embeddings(), dec, model.lm_head).eval()
    with torch.no_grad():
        out = torch.stack([w(ids[i:i + 1], torch.tensor([i])).reshape(-1) for i in range(n)])
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_resolve_rope_type():
    class Cfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    assert resolve_rope_type(Cfg(rope_theta=1e4), "default") == "default"
    assert resolve_rope_type(Cfg(rope_parameters=dict(LLAMA3_SCALING)), "auto") == "llama3"
    # legacy Qwen2-VL marker = default frequencies
    assert resolve_rope_type(Cfg(rope_scaling={"type": "mrope", "mrope_section": [2, 2, 4]}), "default") == "default"
    with pytest.raises(NotImplementedError):  # fixed type contradicted by the config
        resolve_rope_type(Cfg(rope_parameters=dict(LLAMA3_SCALING)), "default")
    with pytest.raises(NotImplementedError):  # no implementation
        resolve_rope_type(Cfg(rope_parameters={"rope_type": "yarn", "factor": 4.0}), "auto")
