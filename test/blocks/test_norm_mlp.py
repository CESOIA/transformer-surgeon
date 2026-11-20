"""blocks/norm and blocks/mlp atoms."""
import pytest
import torch

from transformersurgeon.blocks import LayerNorm, MLP, MLPGated, RMSNorm, RMSNormPrescaled, RMSNormUpcast
from transformersurgeon.blocks.mlp import ACTIVATIONS, build_activation


def _hf_rmsnorm(x, weight, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return weight * (x.float() * torch.rsqrt(variance + eps)).to(x.dtype)


def test_rmsnorm_variants_match_reference_in_float32():
    torch.manual_seed(0)
    x = torch.randn(3, 16) * 0.5  # |x| < 1: the prescale is a no-op
    for cls in (RMSNorm, RMSNormPrescaled, RMSNormUpcast):
        norm = cls(16, eps=1e-6)
        norm.weight.data.uniform_(0.5, 1.5)
        assert torch.allclose(norm(x), _hf_rmsnorm(x, norm.weight, 1e-6), atol=1e-5), cls.__name__


def test_rmsnorm_prescaled_avoids_float16_overflow():
    x = torch.full((1, 64), 300.0, dtype=torch.float16)  # x*x = 9e4 overflows float16
    ones = torch.ones(1, 64, dtype=torch.float16)
    assert not torch.allclose(RMSNorm(64, dtype=torch.float16)(x), ones)  # inf variance -> zeros
    assert torch.allclose(RMSNormPrescaled(64, dtype=torch.float16)(x), ones, atol=1e-2)
    assert torch.allclose(RMSNormUpcast(64, dtype=torch.float16)(x), ones, atol=1e-2)


def test_layernorm_atom_matches_torch():
    norm = LayerNorm(8, eps=1e-12)
    x = torch.randn(2, 8)
    assert torch.allclose(norm(x), torch.nn.functional.layer_norm(x, (8,), eps=1e-12))


def test_activations_registry():
    assert set(ACTIVATIONS) >= {"gelu", "relu", "silu"}
    with pytest.raises(ValueError, match="Unsupported activation"):
        build_activation("swish7")


def test_mlp_and_gated_mlp_forward():
    x = torch.randn(4, 8)
    mlp = MLP(8, 16, activation="relu")
    assert torch.allclose(mlp(x), mlp.down_proj(torch.relu(mlp.up_proj(x))))
    gated = MLPGated(8, 16, activation="silu")
    expected = gated.down_proj(torch.nn.functional.silu(gated.gate_proj(x)) * gated.up_proj(x))
    assert torch.allclose(gated(x), expected)


def test_mlp_takes_pruned_geometry_from_specs():
    specs = {"up_proj": {"out_features": 10, "bias": True}, "gate_proj": {"out_features": 10},
             "down_proj": {"in_features": 10, "rank": 3}}
    gated = MLPGated(8, 16, linear_specs=specs)
    assert gated.up_proj.out_features == 10 and gated.up_proj.bias is not None
    assert gated.down_proj.in_features == 10 and gated.down_proj.rank == 3
    assert gated(torch.randn(2, 8)).shape == (2, 8)


def test_mlp_gated_fused_splits_the_input_projection_gate_first():
    from transformersurgeon.blocks import MLPGatedFused

    torch.manual_seed(0)
    mlp = MLPGatedFused(8, 6, activation="silu")
    assert (mlp.up_proj.out_features, mlp.down_proj.in_features) == (12, 6)
    x = torch.randn(3, 8)
    linear = torch.nn.functional.linear
    gate, up = linear(x, mlp.up_proj.weight, mlp.up_proj.bias).chunk(2, dim=-1)
    want = linear(torch.nn.functional.silu(gate) * up, mlp.down_proj.weight, mlp.down_proj.bias)
    torch.testing.assert_close(mlp(x), want)


def test_pooling_atoms_return_rank_two():
    from transformersurgeon.blocks import build_block

    x = torch.randn(5, 4)
    assert torch.equal(build_block("pooling", "cls")(x), x[:1])
    torch.testing.assert_close(build_block("pooling", "mean")(x), x.mean(0, keepdim=True))
    both = build_block("pooling", "cls_patch_mean")(x)
    assert both.shape == (1, 8)
    torch.testing.assert_close(both[:, 4:], x[1:].mean(0, keepdim=True))
    assert build_block("pooling", "none")(x) is x
