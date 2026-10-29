"""blocks/layers: compressible linear/embedding/convolution layers, VCON block, residual sum."""
import pytest
import torch

from transformersurgeon.blocks import (
    AtomicSum,
    Conv2dCompressed,
    Conv3dCompressed,
    EmbeddingCompressed,
    LinearCompressed,
    VCONBlock,
)
from transformersurgeon.blocks.layers import linear_from_spec


def test_linear_compressed_matches_nn_linear_without_rank():
    torch.manual_seed(0)
    ref = torch.nn.Linear(6, 4)
    lin = LinearCompressed(6, 4)
    lin.load_state_dict(ref.state_dict())
    x = torch.randn(2, 3, 6)  # rank-3 input is flattened and restored
    assert torch.allclose(lin(x), ref(x), atol=1e-6)
    assert lin.rank == "full" and lin.linear_V is None


def test_linear_compressed_low_rank_factors_and_cancel():
    lin = LinearCompressed(6, 4, rank=2)
    assert lin.weight.shape == (4, 2) and lin.linear_V.weight.shape == (2, 6)
    x = torch.randn(5, 6)
    expected = x @ (lin.weight @ lin.linear_V.weight).T + lin.bias
    assert torch.allclose(lin(x), expected, atol=1e-5)
    lin.cancel_lrd()
    assert lin.rank == "full" and lin.linear_V is None and lin.weight.shape == (4, 6)


def test_linear_compressed_fully_pruned_layer_is_identity():
    lin = LinearCompressed(6, 0)
    x = torch.randn(2, 6)
    assert lin.skip and lin(x) is x
    assert "out_features=0" in repr(lin)


def test_linear_from_spec_uses_spec_then_defaults():
    lin = linear_from_spec({"out_features": 3, "rank": 2, "bias": True}, 8, 5)
    assert (lin.in_features, lin.out_features, lin.rank, lin.bias is not None) == (8, 3, 2, True)
    default = linear_from_spec(None, 8, 5)
    assert (default.out_features, default.rank, default.bias) == (5, "full", None)


def test_embedding_compressed_lookup_and_low_rank():
    emb = EmbeddingCompressed(10, 6)
    ids = torch.tensor([1, 4, 9])
    assert torch.equal(emb(ids), emb.weight[ids])
    emb.init_lrd(3)
    assert emb.weight.shape == (10, 3) and emb.linear_V.weight.shape == (3, 6)
    assert torch.allclose(emb(ids), emb.weight[ids] @ emb.linear_V.weight)
    emb.cancel_lrd()
    assert emb.weight.shape == (10, 6)


def test_conv_compressed_are_plain_convolutions():
    conv = Conv2dCompressed(3, 4, kernel_size=2, stride=2)
    ref = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2)
    ref.load_state_dict(conv.state_dict())
    x = torch.randn(1, 3, 4, 4)
    assert torch.equal(conv(x), ref(x))
    assert repr(conv).startswith("Conv2dCompressed(")
    assert repr(Conv3dCompressed(1, 1, 1)).startswith("Conv3dCompressed(")


def test_vcon_block_blends_original_and_compressed():
    a, b = torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)
    block = VCONBlock(a, b)
    x = torch.randn(2, 3)
    assert torch.equal(block(x), a(x))  # beta unset -> original
    block.set_beta(0.25)
    assert torch.allclose(block(x), 0.25 * a(x) + 0.75 * b(x))
    block.set_beta(0.0)
    assert torch.equal(block(x), b(x))
    with pytest.raises(ValueError):
        block.set_beta(1.5)
    lonely = VCONBlock(a)
    lonely.set_beta(0.0)
    with pytest.raises(ValueError, match="Block B"):
        lonely(x)


def test_atomic_sum_adds():
    assert torch.equal(AtomicSum()(torch.ones(2), torch.ones(2)), torch.full((2,), 2.0))
