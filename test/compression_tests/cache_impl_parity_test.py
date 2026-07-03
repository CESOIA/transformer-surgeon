"""
Unit tests: the three KV-cache implementations must be numerically equivalent.

`cache_impl` selects only *how* the fixed-size KV cache is written:
  - "mutable"    : in-place index_copy_ on an internal buffer (legacy default)
  - "io_scatter" : functional index_put on a cache passed as graph I/O
  - "io_concat"  : scatter-free positional-mask write, cache passed as graph I/O

All three attend over the same fixed-size cache, so a prefill+decode run must
produce identical logits. This is the correctness gate for the I/O-cache refactor.

Run:
    cd <repo-root>
    python -m pytest test/compression_tests/cache_impl_parity_test.py -v
"""

import unittest

import torch

from transformersurgeon.blocks import TransformerDecoder
from transformersurgeon.blocks.config import CustomDecoderConfigCompress
from transformersurgeon.blocks.mha import MHACausal, CACHE_IMPLS

# Tiny model — GQA (kv heads < attention heads) to exercise the group path.
_VOCAB = 32
_HIDDEN = 16
_HEADS = 4
_KV_HEADS = 2
_INTER = 32
_LAYERS = 2
_CACHE = 8
_STEPS = 6

_CONVERTED_INDEXING = {
    "num_blocks_attr": "num_hidden_layers",
    "path_template": "blocks.{block_index}.{path}",
    "path_list": {
        "norm_in":  [],
        "attn":     ["q_proj", "k_proj", "v_proj", "out_proj"],
        "norm_out": [],
        "mlp":      ["gate_proj", "up_proj", "down_proj"],
    },
}


def _make_decoder(cache_impl: str) -> TransformerDecoder:
    # Seed before construction so every impl gets identical initial weights.
    torch.manual_seed(0)
    config = CustomDecoderConfigCompress(
        num_hidden_layers=_LAYERS,
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        intermediate_size=_INTER,
        hidden_act="silu",
        attn_type="mha_causal",
        mlp_type="mlp_gated",
        norm_type="rmsnorm",
        max_cache_len=_CACHE,
        cache_impl=cache_impl,
        vocab_size=_VOCAB,
        indexing=_CONVERTED_INDEXING,
    )
    decoder = TransformerDecoder(config)
    decoder.eval()
    return decoder


def _zero_caches(decoder):
    key_caches, value_caches = [], []
    for block in decoder.blocks:
        attn = block.attn
        shape = (attn.max_cache_length, attn.kv_num_heads, attn.head_dim)
        key_caches.append(torch.zeros(shape, dtype=attn.dtype))
        value_caches.append(torch.zeros(shape, dtype=attn.dtype))
    return key_caches, value_caches


def _run_decoder(decoder, seq):
    """Prefill+decode one token at a time; return stacked per-step outputs."""
    outs = []
    with torch.no_grad():
        if decoder.cache_impl == "mutable":
            for t in range(seq.size(0)):
                out = decoder(seq[t:t + 1], pos_id=torch.tensor([t]))
                outs.append(out)
        else:
            key_caches, value_caches = _zero_caches(decoder)
            for t in range(seq.size(0)):
                out, key_caches, value_caches = decoder(
                    seq[t:t + 1],
                    pos_id=torch.tensor([t]),
                    key_caches=key_caches,
                    value_caches=value_caches,
                )
                outs.append(out)
    return torch.cat(outs, dim=0)


class TestCacheImplParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)
        cls.seq = torch.randn(_STEPS, _HIDDEN)

    def test_config_default_is_mutable(self):
        self.assertEqual(_make_decoder("mutable").cache_impl, "mutable")

    def test_all_impls_supported(self):
        self.assertEqual(set(CACHE_IMPLS), {"mutable", "io_scatter", "io_concat"})

    def test_decoder_parity_across_impls(self):
        ref = _run_decoder(_make_decoder("mutable"), self.seq)
        for impl in ("io_scatter", "io_concat"):
            out = _run_decoder(_make_decoder(impl), self.seq)
            self.assertEqual(out.shape, ref.shape)
            torch.testing.assert_close(
                out, ref, rtol=1e-5, atol=1e-5,
                msg=f"{impl} diverged from mutable",
            )

    def test_io_returns_updated_caches(self):
        decoder = _make_decoder("io_scatter")
        key_caches, value_caches = _zero_caches(decoder)
        with torch.no_grad():
            _, new_kc, new_vc = decoder(
                self.seq[0:1], pos_id=torch.tensor([0]),
                key_caches=key_caches, value_caches=value_caches,
            )
        self.assertEqual(len(new_kc), _LAYERS)
        # Row 0 must now hold the written key (non-zero), rest stays zero.
        self.assertGreater(new_kc[0][0].abs().sum().item(), 0.0)
        self.assertEqual(new_kc[0][1].abs().sum().item(), 0.0)

    def test_single_mhacausal_writers_agree(self):
        """The three writers produce identical updated caches for one token."""
        torch.manual_seed(0)
        attn = MHACausal(_HIDDEN, _HEADS, kv_num_heads=_KV_HEADS,
                         max_cache_len=_CACHE, cache_impl="io_scatter", dtype=torch.float32)
        attn.eval()
        k = torch.randn(1, _KV_HEADS, _HIDDEN // _HEADS)
        v = torch.randn(1, _KV_HEADS, _HIDDEN // _HEADS)
        kc, vc = _zero_caches_single(attn)
        idx = torch.tensor([3])
        s_kc, s_vc = attn._write_scatter(k, v, idx, kc.clone(), vc.clone())
        c_kc, c_vc = attn._write_concat(k, v, idx, kc.clone(), vc.clone())
        torch.testing.assert_close(s_kc, c_kc)
        torch.testing.assert_close(s_vc, c_vc)


def _zero_caches_single(attn):
    shape = (attn.max_cache_length, attn.kv_num_heads, attn.head_dim)
    return torch.zeros(shape, dtype=attn.dtype), torch.zeros(shape, dtype=attn.dtype)


if __name__ == "__main__":
    unittest.main()
