"""blocks/positions: rotary embeddings (parameters, frequency atoms, tables, application, pruning)."""
import math
import types

import pytest
import torch

from transformersurgeon.blocks.positions.rope import (
    ROPE_FREQUENCIES,
    apply_rope_multihead,
    build_rope_prune_projection,
    precompute_rope_cos_sin_half,
    precompute_rope_inv_freqs,
    precompute_rope_inv_freqs_for,
    resolve_rope_type,
    rope_parameters_from_config,
    rope_theta_from_config,
)


def test_rope_parameters_from_config_flattens_both_layouts():
    v5 = types.SimpleNamespace(rope_parameters={"rope_type": "default", "rope_theta": 1e4})
    v4 = types.SimpleNamespace(rope_scaling={"type": "llama3", "factor": 8.0}, rope_theta=5e5)
    assert rope_parameters_from_config(v5) == {"rope_type": "default", "rope_theta": 1e4}
    assert rope_parameters_from_config(v4)["rope_type"] == "llama3"
    assert rope_parameters_from_config(v4)["rope_theta"] == 5e5
    sectioned = types.SimpleNamespace(rope_scaling={"type": "mrope"})
    assert rope_parameters_from_config(sectioned)["rope_type"] == "default"
    assert rope_theta_from_config(types.SimpleNamespace()) == 1e6


def test_resolve_rope_type_auto_fixed_and_mismatch():
    scaled = types.SimpleNamespace(rope_parameters={"rope_type": "llama3"})
    assert resolve_rope_type(scaled, "auto") == "llama3"
    assert resolve_rope_type(types.SimpleNamespace(), "default") == "default"
    with pytest.raises(NotImplementedError):
        resolve_rope_type(scaled, "default")
    with pytest.raises(NotImplementedError):
        resolve_rope_type(types.SimpleNamespace(rope_parameters={"rope_type": "yarn"}), "auto")


def test_frequency_atoms():
    assert set(ROPE_FREQUENCIES) == {"default", "llama3"}
    plain = precompute_rope_inv_freqs(8, base=100.0)
    assert torch.allclose(plain, 1.0 / (100.0 ** (torch.arange(0, 8, 2).float() / 8)))
    assert torch.allclose(precompute_rope_inv_freqs_for("default", 8, {"rope_theta": 100.0}), plain)
    params = {"rope_theta": 5e5, "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
              "original_max_position_embeddings": 8192}
    scaled = precompute_rope_inv_freqs_for("llama3", 128, params)
    base = precompute_rope_inv_freqs(128, base=5e5)
    assert torch.allclose(scaled[:4], base[:4]) and torch.allclose(scaled[-1], base[-1] / 8)
    with pytest.raises(ValueError):
        precompute_rope_inv_freqs_for("llama3", 128, {"rope_theta": 1.0})


def test_cos_sin_tables_and_rotation():
    inv = precompute_rope_inv_freqs(4, base=10.0)
    cos, sin = precompute_rope_cos_sin_half(inv, 3, start_pos=2)
    assert cos.shape == (3, 1, 2)
    assert torch.allclose(cos[0, 0, 0], torch.tensor(math.cos(2.0)))
    x = torch.randn(3, 2, 4)
    y = apply_rope_multihead(x, cos, sin)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)  # a rotation keeps the norm
    x1, x2 = x.chunk(2, dim=-1)
    assert torch.allclose(y, torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1))


def test_rope_prune_projection_selects_kept_frequencies():
    head_dim, half = 4, 2
    keep = torch.tensor([True, False, True, False] * 2)  # frequency 0 kept (both halves), per head
    proj = build_rope_prune_projection(keep, keep, head_dim, 2, 2)
    assert proj.shape == (2, 1, half)
    assert torch.equal(proj[0, 0], torch.tensor([1.0, 0.0]))
