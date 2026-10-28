"""Paged KV cache + fused attention (cache_impl="io_paged", attn_impl="paged").

The eager op (blocks/paged_attention_ops.py) follows TensorRT Edge-LLM's
AttentionPlugin contract; the converted decoder built on it must give the same
logits as the io_inplace path, across KV page boundaries and with whole query
heads removed. The TensorRT test runs the plugin (XQA) when its library exists.
"""
from __future__ import annotations

import json

import pytest
import torch

import transformersurgeon as T
from transformersurgeon.blocks.attention.kernels.paged import KV_PAGE_SIZE
from transformersurgeon.export import LLMWrapper, build_zero_caches
from transformersurgeon.export.convert import convert_for_export

from _helpers import capabilities as caps

CACHE = 2 * KV_PAGE_SIZE


def _tiny(dtype=torch.float32, head_dim=16, heads=4, remove_heads=False):
    torch.manual_seed(0)
    cfg = T.Qwen2ConfigCompress(vocab_size=256, hidden_size=heads * head_dim, intermediate_size=128,
                                num_hidden_layers=2, num_attention_heads=heads, num_key_value_heads=2,
                                max_position_embeddings=CACHE, tie_word_embeddings=False)
    m = T.Qwen2ForCausalLMCompress(cfg).to(dtype).eval()
    m.config.dtype = dtype
    if remove_heads:  # one query head per KV group in layer 0 (block pruning)
        mgr = T.models.qwen2_c.Qwen2CompressionSchemesManager(m)
        mgr.set("block_pruning", "blocks", [1, heads // 2 + 1], criteria=[["self_attn.q_proj", 0]])
        mgr.apply(hard=True)
    return m


def _wrapper(m, cache_impl, device="cpu"):
    opts = {"use_sdpa": False, "cache_impl": cache_impl, "max_cache_len": CACHE,
            "rmsnorm_prescale": False, "rmsnorm_upcast": True}
    dec = convert_for_export(m, options=opts)["text"]
    return LLMWrapper(m.get_input_embeddings(), dec, m.lm_head).eval().to(device)


def _decode(w, ids, device="cpu"):
    keys, values = build_zero_caches(w.decoder)
    keys, values = [k.to(device) for k in keys], [v.to(device) for v in values]
    out = []
    with torch.no_grad():
        for pos, t in enumerate(ids):
            args = (torch.tensor([int(t)], device=device), torch.tensor([pos], device=device))
            if w.cache_impl == "io_paged":
                y, keys = w(*args, keys)
            else:
                y, keys, values = w(*args, keys, values)
            out.append(y.float().reshape(-1))
    return torch.stack(out)


@pytest.mark.parametrize("remove_heads", [False, True])
def test_paged_matches_io_inplace_across_pages(remove_heads):
    m = _tiny(remove_heads=remove_heads)
    ids = torch.arange(KV_PAGE_SIZE + 12) * 37 % 256  # crosses the first page boundary
    paged = _wrapper(m, "io_paged")
    assert paged.decoder.blocks[0].attn.cache_shape("kv") == (2, 2, KV_PAGE_SIZE, 2, 16)
    torch.testing.assert_close(_decode(paged, ids), _decode(_wrapper(m, "io_inplace"), ids),
                               rtol=1e-4, atol=1e-4)


def test_paged_rejects_rope_pruned_attention():
    w = _wrapper(_tiny(), "io_paged")
    w.decoder.blocks[1].attn.rope_pruned = True  # per-kv-group RoPE frequencies: no single cos/sin table
    with pytest.raises(NotImplementedError, match="io_paged"):
        _decode(w, torch.tensor([1, 2]))


def test_paged_attention_requires_paged_cache():
    from transformersurgeon.blocks.attention import MHACausal

    with pytest.raises(ValueError, match="io_paged"):
        MHACausal(64, 4, kv_num_heads=2, attn_impl="paged", cache_impl="io_inplace", max_cache_len=16)


@caps.requires_onnx
def test_onnx_backend_rejects_paged_cache(tmp_path):
    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends.onnx import ONNXExportConfig

    cfg = ONNXExportConfig(output_path=str(tmp_path / "m.onnx"), backend="onnx",
                           convert_options={"cache_impl": "io_paged", "max_cache_len": CACHE})
    with pytest.raises(ValueError, match="io_paged"):
        export_to_backend(_tiny(), config=cfg)


@caps.requires_edgellm_plugin
@pytest.mark.parametrize("remove_heads", [False, True])
def test_tensorrt_attention_plugin_matches_eager(tmp_path, remove_heads):
    """XQA decode (AttentionPlugin) == the eager io_inplace graph, past a page boundary."""
    import onnx

    from transformersurgeon.export import export_to_backend
    from transformersurgeon.export.backends.tensorrt import TensorRTExportConfig
    from transformersurgeon.export.backends.tensorrt.session import TensorRTLLMSession

    m = _tiny(torch.float16, head_dim=64, remove_heads=remove_heads)  # XQA: head_dim 32/64/128
    cfg = TensorRTExportConfig(output_path=str(tmp_path / "model.onnx"), backend="tensorrt", max_input_len=1,
                               convert_options={"use_sdpa": False, "cache_impl": "io_paged", "max_cache_len": CACHE,
                                                "rmsnorm_prescale": False, "rmsnorm_upcast": True})
    result = export_to_backend(m, config=cfg)
    graph = onnx.load(result.output_path, load_external_data=False).graph
    assert [n.op_type for n in graph.node].count("AttentionPlugin") == 2
    assert not any(n.op_type == "Softmax" for n in graph.node)
    manifest = json.load(open(result.manifest_path))
    assert manifest["cache_layout"] == "PAGED" and manifest["plugins"]["edgellm_attention"]["layers"] == 2

    session = TensorRTLLMSession(result.manifest_path)
    # The plugin writes the pool in place without declaring the alias (Edge-LLM 0.10.1 WAR):
    # past and present must be one buffer.
    assert session.caches["present_kv_0"] is session.caches["past_kv_0"]
    ids = torch.arange(KV_PAGE_SIZE + 6) * 37 % 256
    got = []
    for pos, t in enumerate(ids):  # teacher-forced, the decode engine call
        session.input_ids.fill_(int(t))
        session.pos_id.fill_(pos)
        session.decoder.run()
        got.append(session.logits.float().reshape(-1).clone())
    ref = _decode(_wrapper(m, "io_inplace", "cuda"), ids, "cuda")
    torch.testing.assert_close(torch.stack(got), ref, atol=3e-2, rtol=3e-2)

    # CUDA-graph decode keeps matching after a reset (stale pool rows are never read).
    session.reset()
    session.prefill(ids[:5])
    tokens = ids[:5].tolist() + [int(session.input_ids.item())]
    for _ in range(3):
        session.step()
        ref = _decode(_wrapper(m, "io_inplace", "cuda"), torch.tensor(tokens), "cuda")[-1]
        torch.testing.assert_close(session.logits.float().reshape(-1), ref, atol=3e-2, rtol=3e-2)
        tokens.append(int(session.input_ids.item()))
