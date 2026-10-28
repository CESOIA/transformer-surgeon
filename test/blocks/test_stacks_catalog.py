"""blocks/stacks (decoder, encoder, configurations) and blocks/catalog.py."""
import types

import pytest
import torch

from transformersurgeon.blocks import (
    BLOCK_CATALOG,
    CustomDecoderConfigCompress,
    CustomEncoderConfigCompress,
    TransformerDecoder,
    TransformerEncoder,
    build_block,
    implementations_of,
    norm_atom_name,
    resolve_atom,
)
from transformersurgeon.blocks.stacks.config import rmsnorm_impl_from_flags

L = 8


def _decoder_config(**options):
    return CustomDecoderConfigCompress(
        num_hidden_layers=2, hidden_size=32, num_attention_heads=4, intermediate_size=48, hidden_act="silu",
        num_key_value_heads=2, mlp_type="mlp_gated", max_cache_len=L, **options)


def _zero_caches(decoder):
    return ([torch.zeros(b.attn.cache_shape("key")) for b in decoder.blocks],
            [torch.zeros(b.attn.cache_shape("value")) for b in decoder.blocks])


def _run(decoder, xs):
    kc, vc = _zero_caches(decoder) if decoder.kv_cache.graph_io else (None, None)
    outs = []
    for t in range(xs.shape[0]):
        out = decoder(xs[t:t + 1], torch.tensor([t]), kc, vc)
        if isinstance(out, tuple):
            out, kc, vc = out
        outs.append(out)
    return torch.cat(outs)


@pytest.mark.parametrize("cache_impl", ["io_scatter", "io_concat", "io_inplace"])
def test_decoder_is_independent_of_the_cache_atom(cache_impl):
    torch.manual_seed(0)
    ref = TransformerDecoder(_decoder_config()).eval()
    other = TransformerDecoder(_decoder_config(cache_impl=cache_impl)).eval()
    other.load_state_dict(ref.state_dict())
    xs = torch.randn(4, 32)
    with torch.no_grad():
        assert torch.allclose(_run(ref, xs), _run(other, xs), atol=1e-5)


def test_decoder_prefill_equals_token_by_token():
    torch.manual_seed(0)
    dec = TransformerDecoder(_decoder_config(cache_impl="io_inplace")).eval()
    xs = torch.randn(3, 32)
    kc, vc = _zero_caches(dec)
    with torch.no_grad():
        prefill, _, _ = dec(xs, torch.tensor([0]), kc, vc)
        assert torch.allclose(prefill[-1:], _run(dec, xs)[-1:], atol=1e-5)


def test_decoder_add_batch_dim_and_norm_atoms():
    ref = TransformerDecoder(_decoder_config()).eval()
    batched = TransformerDecoder(_decoder_config(add_batch_dim=True)).eval()
    batched.load_state_dict(ref.state_dict())
    x = torch.randn(1, 32)
    with torch.no_grad():
        assert torch.allclose(batched(x[None], torch.tensor([0]))[0], ref(x, torch.tensor([0])), atol=1e-6)
    assert type(ref.norm).__name__ == "RMSNormPrescaled"
    assert type(TransformerDecoder(_decoder_config(rmsnorm_impl="rmsnorm_upcast")).norm).__name__ == "RMSNormUpcast"
    assert type(TransformerDecoder(_decoder_config(rmsnorm_prescale=False)).norm).__name__ == "RMSNorm"


def test_decoder_config_from_source_config_reads_the_model_dictionary():
    source = types.SimpleNamespace(
        to_dict=lambda: {"n_layer": 2, "width": 32, "heads": 4, "ffn": 48, "act": "silu", "kv": 2,
                         "rms_norm_eps": 1e-6, "compression_config": {"x": 1}})
    indexing = {"num_blocks_attr": "n_layer", "embed_dim_attr": "width", "num_heads_attr": "heads",
                "mlp_hidden_dim_attr": "ffn", "mlp_activation_attr": "act", "kv_num_heads_attr": "kv",
                "mlp_type": "mlp_gated"}
    cfg = CustomDecoderConfigCompress.from_source_config(source, indexing, {"blocks.0.mlp.up_proj": {"rank": 4}},
                                                         cache_impl="io_scatter")
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_key_value_heads) == (2, 32, 2)
    assert cfg.mlp_type == "mlp_gated" and cfg.cache_impl == "io_scatter" and cfg.rms_norm_eps == 1e-6
    assert not hasattr(cfg, "compression_config")
    assert cfg.block_layer_specs(0, "mlp") == {"up_proj": {"rank": 4}}
    with pytest.raises(KeyError):
        CustomDecoderConfigCompress.from_source_config(source, {"num_blocks_attr": "n_layer"})


@pytest.mark.parametrize("norm_position", ["pre", "post"])
def test_encoder_residual_ordering(norm_position):
    cfg = CustomEncoderConfigCompress(num_hidden_layers=1, hidden_size=16, num_attention_heads=2,
                                      intermediate_size=32, hidden_act="gelu", norm_position=norm_position)
    enc = TransformerEncoder(cfg).eval()
    block = enc.blocks[0]
    x = torch.randn(3, 16)
    if norm_position == "pre":
        h = x + block.attn(block.norm_in(x))
        expected = enc.norm(h + block.mlp(block.norm_out(h)))
    else:
        h = block.norm_in(x + block.attn(x))
        expected = enc.norm(block.norm_out(h + block.mlp(h)))
    with torch.no_grad():
        assert torch.allclose(enc(x), expected, atol=1e-5)
    with pytest.raises(ValueError):
        enc(torch.randn(1, 3, 16))


def test_catalog_entries_all_resolve_and_are_documented():
    for family, atoms in BLOCK_CATALOG.items():
        for name, entry in atoms.items():
            assert callable(resolve_atom(family, name)), (family, name)
            assert entry["summary"] and entry["selected_by"] in ("model", "export")
    with pytest.raises(ValueError, match="Unknown"):
        resolve_atom("norm", "groupnorm")


def test_catalog_helpers():
    assert set(implementations_of("rmsnorm", "norm")) == {"rmsnorm", "rmsnorm_prescaled", "rmsnorm_upcast"}
    assert norm_atom_name("rmsnorm", "rmsnorm_upcast") == "rmsnorm_upcast"
    assert norm_atom_name("layernorm", "rmsnorm_upcast") == "layernorm"
    with pytest.raises(ValueError):
        norm_atom_name("rmsnorm", "layernorm")
    assert type(build_block("mlp", "mlp", 4, 8)).__name__ == "MLP"
    assert rmsnorm_impl_from_flags(True, False) == "rmsnorm_prescaled"
    assert rmsnorm_impl_from_flags(False, False) == "rmsnorm"
    assert rmsnorm_impl_from_flags(True, True) == "rmsnorm_upcast"
