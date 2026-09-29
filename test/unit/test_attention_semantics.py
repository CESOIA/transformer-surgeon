"""Converted-attention semantics that exports rely on.

- Within-head (q/k RoPE-pair, v channel) structured pruning keeps the original
  1/sqrt(head_dim) score scale: the hard-pruned converted decoder equals the
  soft-pruned HF model (pruning must not change the softmax temperature).
- masked_softmax (masked scores = row min - 20) equals the additive mask.
"""
import torch

import transformersurgeon as T
from transformersurgeon.export.common import LLMWrapper
from transformersurgeon.utils import convert_for_export

N = 24
IDS = torch.arange(1, N + 1) * 11 % 256


def _tiny():
    torch.manual_seed(0)
    cfg = T.Qwen2ConfigCompress(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=False)
    m = T.Qwen2ForCausalLMCompress(cfg).eval()
    return m, T.models.qwen2_c.Qwen2CompressionSchemesManager(m)


def _decode(m, **opts):
    dec = convert_for_export(m, options={"use_sdpa": False, "max_cache_len": N, **opts})["text"]
    w = LLMWrapper(m.get_input_embeddings(), dec, m.lm_head).eval()
    with torch.no_grad():
        return torch.stack([w(IDS[j:j + 1], torch.tensor([j])).reshape(-1) for j in range(N)])


def _prune_within_head(mgr, hard):
    groups = mgr.auto_groups()
    for g in groups:
        mgr.set("structured_pruning", "share_mask", True, group=g)
        mgr.set("structured_pruning", "reduce_op", "add", group=g)
    for g, paths in groups.items():
        if any("q_proj" in p for p in paths):
            for k, v in (("granularity", 16), ("repeated_pattern", "auto"), ("ratio", 0.25)):
                mgr.set("structured_pruning", k, v, group=g)
    for k, v in (("granularity", 16), ("coupled_repeated_pattern", 2), ("ratio", 0.25)):
        mgr.set("structured_pruning", k, v, criteria="v_proj")
    mgr.apply(hard=hard)


def test_within_head_pruning_keeps_score_scale():
    m, mgr = _tiny()
    _prune_within_head(mgr, hard=False)
    with torch.no_grad():
        soft = m(IDS[None]).logits[0]
    m, mgr = _tiny()
    _prune_within_head(mgr, hard=True)
    assert m.model.layers[0].self_attn.q_proj.out_features < 64
    torch.testing.assert_close(_decode(m), soft, rtol=1e-4, atol=1e-4)


def test_masked_softmax_equals_additive_mask():
    m, _ = _tiny()
    torch.testing.assert_close(_decode(m, masked_softmax=True), _decode(m), rtol=1e-5, atol=1e-5)
