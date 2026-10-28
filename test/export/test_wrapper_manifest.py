"""export/wrapper.py, manifest.py, quant_info.py, checks.py, result.py."""
import json

import pytest
import torch

from _helpers.model_factory import FAMILIES
from transformersurgeon.blocks.attention.kernels.paged import KV_PAGE_SIZE
from transformersurgeon.export import (
    LLMWrapper, build_example_inputs, build_llm_manifest, build_zero_caches, convert_for_export,
    extract_layer_quant_info, load_llm_manifest, manifest_path_for, zero_caches_from_manifest,
)
from transformersurgeon.export.checks import find_weight_mismatches, logit_error_stats
from transformersurgeon.export.result import finalize_export_result

CACHE = 16


def _model(quantize=False):
    torch.manual_seed(0)
    spec = FAMILIES["qwen2"]
    model = spec.build().float().eval()
    if quantize:
        mgr = spec.manager(model)
        mgr.set("quantization", "precision", 8, criteria="mlp")
        mgr.set("quantization", "granularity", "per_channel", criteria="mlp")
        mgr.apply(hard=True, show_progress=False)
    return model


def _wrapper(model, cache_impl="mutable", max_cache_len=CACHE):
    options = {"cache_impl": cache_impl, "max_cache_len": max_cache_len}
    decoder = convert_for_export(model, options=options)["text"]
    return LLMWrapper(model.get_input_embeddings(), decoder, model.lm_head).eval()


@pytest.mark.parametrize("cache_impl, n_keys, n_values", [
    ("mutable", 0, 0), ("io_scatter", 2, 2), ("io_inplace", 2, 2),
])
def test_zero_caches_follow_the_cache_atom(cache_impl, n_keys, n_values):
    wrapper = _wrapper(_model(), cache_impl)
    keys, values = build_zero_caches(wrapper.decoder)
    assert (len(keys), len(values)) == (n_keys, n_values)
    assert wrapper.graph_io_cache == (cache_impl != "mutable")
    for k, block in zip(keys, wrapper.decoder.blocks):
        assert tuple(k.shape) == tuple(block.attn.cache_shape("key")) and not k.any()


def test_paged_wrapper_takes_one_pool_per_layer():
    wrapper = _wrapper(_model(), "io_paged", max_cache_len=KV_PAGE_SIZE)
    pools, values = build_zero_caches(wrapper.decoder)
    assert len(pools) == 2 and values == []
    with torch.no_grad():
        logits, new_pools = wrapper(torch.tensor([3]), torch.tensor([0]), pools)
    assert logits.shape == (256,) and len(new_pools) == 2


@pytest.mark.parametrize("cache_impl", ["mutable", "io_inplace"])
def test_wrapper_signature_returns_last_token_logits(cache_impl):
    wrapper = _wrapper(_model(), cache_impl)
    args = (torch.tensor([3]), torch.tensor([0]))
    with torch.no_grad():
        out = wrapper(*args, *build_zero_caches(wrapper.decoder)) if wrapper.graph_io_cache else wrapper(*args)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (256,)
    if wrapper.graph_io_cache:
        assert len(out) == 3 and len(out[1]) == len(out[2]) == 2


def test_example_inputs_use_vocab_and_batch_dim():
    ids, pos = build_example_inputs(_model().config)
    assert ids.shape == (1,) and pos.tolist() == [1] and 0 <= int(ids) < 256
    assert build_example_inputs(None, add_batch_dim=True)[0].shape == (1, 1)


@pytest.mark.parametrize("cache_impl", ["io_inplace", "io_paged"])
def test_manifest_roundtrip_gives_the_wrapper_caches(tmp_path, cache_impl):
    model = _model()
    wrapper = _wrapper(model, cache_impl, max_cache_len=KV_PAGE_SIZE)
    artifact = str(tmp_path / "model.onnx")
    manifest = build_llm_manifest(wrapper, model.config, backend="onnx", artifact=artifact, note="x")
    assert manifest["cache_impl"] == cache_impl and manifest["num_layers"] == 2 and manifest["note"] == "x"
    assert manifest["cache_layout"] == {"io_inplace": "BHSD", "io_paged": "PAGED"}[cache_impl]
    with open(manifest_path_for(artifact), "w") as f:
        json.dump(manifest, f)
    loaded = load_llm_manifest(artifact)
    assert loaded == load_llm_manifest(manifest_path_for(artifact)) == json.loads(json.dumps(manifest))
    for got, expected in zip(zero_caches_from_manifest(loaded), build_zero_caches(wrapper.decoder)):
        assert [t.shape for t in got] == [t.shape for t in expected]
        assert all(g.dtype == e.dtype for g, e in zip(got, expected))


def test_missing_manifest_is_an_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_llm_manifest(str(tmp_path / "absent.pte"))


def test_quant_info_lists_only_quantized_linears_and_dequantizes_them():
    pytest.importorskip("torchao")
    wrapper = _wrapper(_model(quantize=True))
    info = extract_layer_quant_info(wrapper)
    assert sorted(info) == sorted(f"decoder.blocks.{i}.mlp.{p}" for i in range(2)
                                  for p in ("gate_proj", "up_proj", "down_proj"))
    for name, layer in info.items():
        weight = wrapper.get_submodule(name).weight
        assert layer["precision"] == 8 and layer["hard"] and layer["per_channel"]
        assert layer["scale"].shape[0] == weight.shape[0]
        assert type(weight.data) is torch.Tensor  # plain float, ready for torch.export


class _Exported:
    """Stand-in for a PT2E program: an int8 weight with its per-row scale."""

    def __init__(self, weight, shift=0.0):
        scale = weight.abs().amax(dim=1) / 127
        self._sd = {"_frozen_param0": torch.round(weight / scale[:, None]).to(torch.int8),
                    "_scale_0": scale + shift}

    def state_dict(self):
        return self._sd


def test_weight_mismatch_check_flags_only_wrong_scales():
    wrapper = _wrapper(_model())
    weight = wrapper.final_layer.weight.detach()
    assert find_weight_mismatches(wrapper, _Exported(weight), eps=1e-3) == []
    wrong = find_weight_mismatches(wrapper, _Exported(weight, shift=0.05), eps=1e-3)
    assert [m["matched_weight"] for m in wrong] == ["final_layer.weight"]


def test_logit_error_stats():
    stats = logit_error_stats(torch.zeros(4), (torch.full((4,), 2.0),))
    assert stats == {"max_abs_err": 2.0, "mean_abs_err": 2.0, "mse": 4.0, "rmse": 2.0}


def test_finalize_writes_the_manifest_next_to_the_artifact(tmp_path):
    model = _model()
    wrapper = _wrapper(model, "io_inplace")
    artifact = str(tmp_path / "model.pte")
    result = finalize_export_result(
        output_path=artifact, backend="xnnpack", precision="full", wrapper=wrapper,
        exported_for_mismatch=None, run_weight_mismatch_check=False, weight_mismatch_eps=1e-3,
        verbose=False, model_config=model.config,
    )
    assert result.manifest_path == manifest_path_for(artifact) and result.weight_mismatches == []
    manifest = load_llm_manifest(artifact)
    assert manifest["backend"] == "xnnpack" and manifest["precision"] == "full"
    assert manifest["artifact"] == "model.pte" and manifest["max_cache_len"] == CACHE
