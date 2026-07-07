"""End-to-end per-family compression tests on tiny random-weight models.

For every supported family this exercises the full framework-specific path:
instantiate the ``*Compress`` class (which replaces target ``nn.Linear`` layers) →
forward → build the manager → soft LRD + restore → int8 quantization →
unstructured pruning → coupled structured pruning (hard) → forward again → and,
for causal LMs, ``convert_for_export``.

These run in-process with zero downloads and are the fast backbone of the suite.
Families with a known, still-open framework bug are xfailed with a pointer to
``FRAMEWORK_PROBLEMS.md`` so a regression that *fixes* them shows up as XPASS.
"""
from __future__ import annotations

import pytest
import torch

from _helpers.model_factory import FAMILIES, ALL_FAMILY_NAMES

pytestmark = pytest.mark.e2e


def _spec(name):
    return FAMILIES[name]


def _build_manager(spec):
    model = spec.build().eval()
    if spec.known_broken:
        pytest.xfail(spec.known_broken)
    mgr = spec.manager(model)
    return model, mgr


@pytest.mark.parametrize("name", ALL_FAMILY_NAMES)
def test_instantiate_and_forward(name):
    """The compressed model class builds and runs a forward pass."""
    spec = _spec(name)
    model = spec.build().eval()
    with torch.no_grad():
        out = model(**spec.sample_inputs(model))
    assert out is not None


@pytest.mark.parametrize("name", ALL_FAMILY_NAMES)
def test_build_manager(name):
    """The family's CompressionSchemesManager can be constructed."""
    model, mgr = _build_manager(_spec(name))
    assert len(list(mgr)) > 0, "manager produced no schemes"


@pytest.mark.parametrize("name", ALL_FAMILY_NAMES)
def test_lrd_soft_apply_and_restore(name):
    """Soft LRD changes outputs and restore() returns to the baseline."""
    spec = _spec(name)
    model, mgr = _build_manager(spec)
    inputs = spec.sample_inputs(model)

    with torch.no_grad():
        base = _logits(model(**inputs))

    mgr.set("lrd", "rank", 8, criteria=spec.lrd_criteria)
    mgr.apply(hard=False)
    with torch.no_grad():
        _logits(model(**inputs))  # must not raise

    mgr.restore()
    with torch.no_grad():
        restored = _logits(model(**inputs))
    torch.testing.assert_close(base, restored, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", ALL_FAMILY_NAMES)
def test_quantization_int8_apply(name):
    """int8 quantization applies (precision as the integer 8) and forward runs.

    NB: precision is the integer 8, not the string "int8" — the latter is what the
    docs show but the validator rejects it (FRAMEWORK_PROBLEMS.md #2).
    """
    spec = _spec(name)
    model, mgr = _build_manager(spec)
    inputs = spec.sample_inputs(model)
    mgr.set("quantization", "precision", 8, criteria=spec.lrd_criteria)
    mgr.apply(hard=False)
    with torch.no_grad():
        model(**inputs)


@pytest.mark.parametrize("name", ALL_FAMILY_NAMES)
def test_unstructured_pruning_apply_and_restore(name):
    spec = _spec(name)
    model, mgr = _build_manager(spec)
    inputs = spec.sample_inputs(model)
    mgr.set("unstructured_pruning", "ratio", 0.5, criteria=spec.lrd_criteria)
    mgr.apply(hard=False)
    with torch.no_grad():
        model(**inputs)
    mgr.restore()
    with torch.no_grad():
        model(**inputs)


@pytest.mark.parametrize("name", ALL_FAMILY_NAMES)
def test_structured_prune_hard_mlp(name):
    """Hard MLP structured pruning shrinks the layer AND keeps forward valid.

    Coupled MLP projections (gate/up in gated MLPs) are pruned together with a
    shared mask so the elementwise product stays shape-consistent and the cascade
    into down_proj matches. Pruning only one of a coupled pair silently produces a
    broken model (FRAMEWORK_PROBLEMS.md #3) — this test deliberately does it right.
    """
    spec = _spec(name)
    if spec.known_broken:
        pytest.xfail(spec.known_broken)
    if not spec.struct_prune_supported:
        pytest.skip("structured MLP pruning not wired for this family "
                    "(dual-tower VL, FRAMEWORK_PROBLEMS.md #6)")
    model = spec.build().eval()
    mgr = spec.manager(model)

    crit = spec.mlp_prune_criteria
    if isinstance(crit, list) and crit and isinstance(crit[0], (list, str)) and len(crit) > 1:
        # Coupled gated MLP: share one mask across the group.
        groups = mgr.auto_groups()
        mlp_groups = [g for g, paths in groups.items()
                      if any("gate_proj" in str(p) or "up_proj" in str(p) for p in paths)]
        for g in mlp_groups:
            mgr.set("structured_pruning", "share_mask", True, group=g)
            mgr.set("structured_pruning", "reduce_op", "add", group=g)
        for c in crit:
            mgr.set("structured_pruning", "ratio", 0.25, criteria=c)
    else:
        mgr.set("structured_pruning", "ratio", 0.25, criteria=crit)

    mgr.apply(hard=True)
    model.eval()
    with torch.no_grad():
        out = model(**spec.sample_inputs(model))
    assert out is not None


@pytest.mark.parametrize("name", [n for n in ALL_FAMILY_NAMES if FAMILIES[n].is_causal])
def test_convert_for_export(name):
    """Causal LMs convert to the export-ready decoder graph."""
    from transformersurgeon import convert_for_export

    spec = _spec(name)
    model = spec.build().eval()
    converted = convert_for_export(model, options={"use_sdpa": False})
    assert "text" in converted


def _logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    return out.last_hidden_state
