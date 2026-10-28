"""hf/hf_export.py: save a compressed model as a Hugging Face directory (no Hub upload)."""
import json
import os

import pytest
import torch

from _helpers.model_factory import FAMILIES
from transformersurgeon.hf import export_to_hf


def _compressed(family, rank=8):
    torch.manual_seed(0)
    spec = FAMILIES[family]
    model = spec.build().float().eval()
    mgr = spec.manager(model)
    mgr.set("lrd", "rank", rank, criteria=spec.lrd_criteria)
    return spec, model, mgr


@pytest.mark.parametrize("family, auto_class", [
    ("qwen2", "AutoModelForCausalLM"), ("bert", "AutoModelForSequenceClassification"),
])
def test_export_with_manager_reloads_with_identical_outputs(tmp_path, family, auto_class):
    spec, model, mgr = _compressed(family)
    saved = export_to_hf(model, repo_id=f"user/{family}-lrd", out_dir=str(tmp_path), manager=mgr)
    assert saved == os.path.join(str(tmp_path), f"{family}-lrd")

    config = json.load(open(os.path.join(saved, "config.json")))
    assert config["trust_remote_code"] is True
    assert config["auto_map"] == {
        "AutoConfig": f"autoload.{type(model.config).__name__}",
        "AutoModel": f"autoload.{type(model).__name__}",
        auto_class: f"autoload.{type(model).__name__}",
    }
    autoload = open(os.path.join(saved, "autoload.py")).read()
    assert f"from transformersurgeon import {type(model).__name__}" in autoload

    reloaded = type(model).from_pretrained(saved).float().eval()
    inputs = spec.sample_inputs(model)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**inputs).logits, model(**inputs).logits)


def test_export_without_manager_saves_the_model_as_it_is(tmp_path):
    _, model, mgr = _compressed("qwen2")
    mgr.apply(hard=True, show_progress=False)
    mgr.prepare_for_save()
    saved = export_to_hf(model, repo_id="lrd", out_dir=str(tmp_path), readme="# tiny\n")
    assert open(os.path.join(saved, "README.md")).read() == "# tiny\n"
    reloaded = type(model).from_pretrained(saved)
    assert reloaded.model.layers[0].self_attn.q_proj.linear_V.weight.shape == (8, 64)
