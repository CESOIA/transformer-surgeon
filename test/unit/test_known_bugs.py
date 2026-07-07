"""Regression tests that pin currently-open framework bugs.

Each test asserts the *correct* (documented / expected) behavior and is marked
``xfail`` with a pointer to ``FRAMEWORK_PROBLEMS.md``. When the underlying bug is
fixed the test XPASSes, which surfaces in the summary and is the signal to remove
the ``xfail`` marker. Keep ``strict=False`` so a fix never turns the suite red.

All tests use tiny random-weight models — no downloads, no GPU.
"""
from __future__ import annotations

import pytest
import torch

from _helpers.model_factory import FAMILIES

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# #1  BERT (list-form calibration_groups) — manager cannot be constructed
#     Fixed: _get_calibration_groups_from_indexing now has a list branch.
# --------------------------------------------------------------------------- #
def test_bert_manager_builds():
    spec = FAMILIES["bert"]
    model = spec.build().eval()
    mgr = spec.manager(model)
    assert len(list(mgr)) > 0


# --------------------------------------------------------------------------- #
# #2  Quantization precision string ("int8"/"int4"/"int2") — docs vs validator
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(reason="FRAMEWORK_PROBLEMS.md #2: docs list precision='int8' but "
                          "validate_precision only accepts int / 'full' / 'binary'",
                   strict=False)
def test_quantization_accepts_documented_int8_string():
    spec = FAMILIES["qwen2"]
    model = spec.build().eval()
    mgr = spec.manager(model)
    mgr.set("quantization", "precision", "int8", criteria=spec.lrd_criteria)
    mgr.apply(hard=False)  # raises ValueError today


def test_quantization_precision_int_is_the_real_api():
    """The integer form (precision=8) is what actually works — pin it so it stays."""
    spec = FAMILIES["qwen2"]
    model = spec.build().eval()
    mgr = spec.manager(model)
    mgr.set("quantization", "precision", 8, criteria=spec.lrd_criteria)
    mgr.apply(hard=False)
    with torch.no_grad():
        model(**spec.sample_inputs(model))


# --------------------------------------------------------------------------- #
# #3  Partial coupled MLP prune silently yields a broken model
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(reason="FRAMEWORK_PROBLEMS.md #3: hard-pruning one of a coupled "
                          "gate/up pair should error or stay valid, not silently "
                          "break the forward pass",
                   strict=False)
def test_partial_coupled_gate_prune_is_guarded():
    spec = FAMILIES["qwen2"]
    model = spec.build().eval()
    mgr = spec.manager(model)
    # Prune ONLY gate_proj (its coupled partner up_proj is left full).
    mgr.set("structured_pruning", "ratio", 0.25, criteria="mlp.gate_proj")
    mgr.apply(hard=True)  # succeeds today with no warning
    model.eval()
    with torch.no_grad():
        # ... but the forward then blows up on a shape mismatch. Either the apply
        # should have raised, or the model should still run.
        model(**spec.sample_inputs(model))
