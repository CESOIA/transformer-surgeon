"""export/convert.py: Hugging Face model -> export-ready stacks, numerically equivalent."""
import pytest
import torch

from _helpers.model_factory import FAMILIES
from transformersurgeon.export import LLMWrapper, build_zero_caches, convert_for_export, resolve_convert_options

STEPS = 5


def _hf_last_logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids[None]).logits[0]


def _decode(wrapper, ids):
    keys, values = build_zero_caches(wrapper.decoder)
    outs = []
    with torch.no_grad():
        for t in range(len(ids)):
            args = (ids[t:t + 1], torch.tensor([t]))
            out = wrapper(*args, keys, values) if wrapper.graph_io_cache else wrapper(*args)
            if isinstance(out, tuple):
                out, keys, values = out
            outs.append(out)
    return torch.stack(outs)


def _wrapper(model, options):
    name = next(n for n, i in model.indexing.items() if i.get("structure") == "transformer_decoder")
    decoder = convert_for_export(model, options=options)[name]
    return LLMWrapper(model.get_input_embeddings(), decoder, model.lm_head).eval()


@pytest.mark.parametrize("cache_impl", ["mutable", "io_scatter", "io_inplace"])
@pytest.mark.parametrize("family", ["qwen2", "llama"])
def test_converted_decoder_matches_hf(family, cache_impl):
    torch.manual_seed(0)
    model = FAMILIES[family].build().float().eval()
    ids = torch.randint(0, 256, (STEPS,))
    options = {"cache_impl": cache_impl, "max_cache_len": 16, "rmsnorm_impl": "rmsnorm_upcast"}
    torch.testing.assert_close(_decode(_wrapper(model, options), ids), _hf_last_logits(model, ids),
                               rtol=1e-4, atol=1e-4)


def test_converted_decoder_keeps_compression():
    torch.manual_seed(0)
    spec = FAMILIES["qwen2"]
    model = spec.build().float().eval()
    manager = spec.manager(model)
    for name, paths in manager.auto_groups().items():
        if any("gate_proj" in p for p in paths):
            manager.set("structured_pruning", "share_mask", True, group=name)
            manager.set("structured_pruning", "reduce_op", "add", group=name)
    manager.set("structured_pruning", "ratio", 0.25, criteria=["mlp.gate_proj", "mlp.up_proj"])
    manager.set("lrd", "rank", 8, criteria="self_attn.o_proj")
    manager.apply(hard=True)
    wrapper = _wrapper(model, {"max_cache_len": 16, "rmsnorm_impl": "rmsnorm_upcast"})
    block = wrapper.decoder.blocks[0]
    assert block.mlp.up_proj.out_features == 96 and block.attn.out_proj.rank == 8
    ids = torch.randint(0, 256, (STEPS,))
    torch.testing.assert_close(_decode(wrapper, ids), _hf_last_logits(model, ids), rtol=1e-4, atol=1e-4)


def test_components_without_structure_are_skipped():
    model = FAMILIES["qwen2_vl"].build()
    with pytest.warns(UserWarning, match="Structure not specified"):
        converted = convert_for_export(model)
    assert all(v is None for v in converted.values())


@pytest.mark.parametrize("family", ["bert", "distilbert", "vit"])
def test_encoder_families_convert_to_an_encoder_stack(family):
    model = FAMILIES[family].build().eval()
    stack = next(v for v in convert_for_export(model).values() if v is not None)
    assert type(stack).__name__ == "TransformerEncoder"
    x = torch.randn(4, stack.embed_dim)
    with torch.no_grad():
        assert stack(x).shape == x.shape


def _encoder_reference(family, model):
    """Embeddings and Hugging Face encoder output (no batch dimension) for one sample."""
    if family == "bert":
        emb = model.bert.embeddings(input_ids=torch.randint(0, 256, (1, 6)))
        return emb[0], model.bert.encoder(emb)[0][0]
    if family == "distilbert":
        emb = model.distilbert.embeddings(torch.randint(0, 256, (1, 6)))
        return emb[0], model.distilbert.transformer(emb, attn_mask=torch.ones(1, 6))[0][0]
    emb = model.vit.embeddings(torch.randn(1, 3, 32, 32))
    hidden = emb
    for layer in model.vit.layers:
        hidden = layer(hidden)
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden
    return emb[0], model.vit.layernorm(hidden)[0]


@pytest.mark.parametrize("family", ["bert", "distilbert", "vit"])
def test_converted_encoder_matches_hf(family):
    torch.manual_seed(0)
    model = FAMILIES[family].build().float().eval()
    stack = next(v for v in convert_for_export(model).values() if v is not None).eval()
    with torch.no_grad():
        emb, ref = _encoder_reference(family, model)
        torch.testing.assert_close(stack(emb), ref, rtol=1e-4, atol=1e-4)


def test_resolve_convert_options_defaults():
    assert resolve_convert_options()["cache_impl"] == "mutable"
    assert resolve_convert_options({"rmsnorm_upcast": True})["rmsnorm_impl"] == "rmsnorm_upcast"
