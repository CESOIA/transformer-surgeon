"""
End-to-end test: an io_* cache decoder exports through the XNNPACK path with the
explicit KV-cache I/O contract, and (when the runtime is installed) the exported
program matches the float wrapper.

Self-contained (tiny synthetic decoder, no model download). Skips gracefully
when executorch export libs or the runtime pybindings are unavailable.

Run:
    cd <repo-root>
    python -m pytest test/executorch_tests/simple/io_cache_export_test.py -v
"""

import os
import tempfile
import unittest

import torch

from transformersurgeon.blocks import TransformerDecoder
from transformersurgeon.blocks.config import CustomDecoderConfigCompress
from transformersurgeon.export.common import LLMWrapper, build_zero_caches

_IDX = {
    "num_blocks_attr": "num_hidden_layers",
    "path_template": "blocks.{block_index}.{path}",
    "path_list": {
        "norm_in":  [],
        "attn":     ["q_proj", "k_proj", "v_proj", "out_proj"],
        "norm_out": [],
        "mlp":      ["gate_proj", "up_proj", "down_proj"],
    },
}
_LAYERS, _HIDDEN, _HEADS, _KV, _INTER, _CACHE, _VOCAB = 2, 16, 4, 2, 32, 8, 32


def _build(impl):
    torch.manual_seed(0)
    cfg = CustomDecoderConfigCompress(
        num_hidden_layers=_LAYERS, hidden_size=_HIDDEN, num_attention_heads=_HEADS,
        num_key_value_heads=_KV, intermediate_size=_INTER, hidden_act="silu",
        attn_type="mha_causal", mlp_type="mlp_gated", norm_type="rmsnorm",
        max_cache_len=_CACHE, cache_impl=impl, vocab_size=_VOCAB, indexing=_IDX,
    )
    dec = TransformerDecoder(cfg).eval()
    torch.manual_seed(1)
    wrapper = LLMWrapper(torch.nn.Embedding(_VOCAB, _HIDDEN), dec,
                         torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)).eval()
    return wrapper, cfg


class TestIOCacheExport(unittest.TestCase):
    def _export(self, impl):
        try:
            from executorch.exir import to_edge_transform_and_lower
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        except Exception as e:  # pragma: no cover - env dependent
            self.skipTest(f"executorch export libs unavailable: {e}")
        from torch.export import export

        wrapper, _ = _build(impl)
        kc, vc = build_zero_caches(wrapper.decoder)
        ex = (torch.tensor([5]), torch.tensor([0]), kc, vc)

        ep = export(wrapper, ex)
        # Contract: outputs are (logits, *new_key_caches, *new_value_caches).
        n_out = len(ep.graph.find_nodes(op="output")[0].args[0])
        self.assertEqual(n_out, 1 + 2 * _LAYERS)

        et = to_edge_transform_and_lower(
            ep, partitioner=[XnnpackPartitioner()]
        ).to_executorch()
        path = tempfile.mktemp(suffix=".pte")
        with open(path, "wb") as f:
            f.write(et.buffer)
        self.assertGreater(os.path.getsize(path), 0)
        return wrapper, ex, path

    def test_io_scatter_export_contract(self):
        # index_put must survive lowering (→ ScatterNd/ScatterND on QNN/TRT).
        self._export("io_scatter")

    def test_io_concat_export_contract(self):
        self._export("io_concat")

    def test_io_scatter_runtime_parity(self):
        self._maybe_check_runtime(*self._export("io_scatter"))

    def test_io_concat_runtime_parity(self):
        self._maybe_check_runtime(*self._export("io_concat"))

    def _maybe_check_runtime(self, wrapper, ex, path):
        try:
            from executorch.runtime import Runtime
        except Exception as e:  # pragma: no cover - env dependent
            self.skipTest(f"executorch runtime unavailable: {e}")

        with torch.no_grad():
            logits_ref = wrapper(*ex)[0]
        flat = [ex[0], ex[1], *ex[2], *ex[3]]
        program = Runtime.get().load_program(path)
        outs = program.load_method("forward").execute(flat)
        self.assertEqual(len(outs), 1 + 2 * _LAYERS)
        torch.testing.assert_close(outs[0], logits_ref, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
