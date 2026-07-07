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
#     Fixed: Compressor.check_coupling() (called from manager._validate_coupling())
#     now rejects inconsistent coupled-mask groups before apply() touches any weights.
# --------------------------------------------------------------------------- #
def test_partial_coupled_gate_prune_is_guarded():
    spec = FAMILIES["qwen2"]
    model = spec.build().eval()
    mgr = spec.manager(model)
    # Prune ONLY gate_proj (its coupled partner up_proj is left full).
    mgr.set("structured_pruning", "ratio", 0.25, criteria="mlp.gate_proj")
    with pytest.raises(ValueError, match="Coupled pruning"):
        mgr.apply(hard=True)


def test_both_coupled_gate_up_pruned_independently_is_guarded():
    """Pruning both members without a shared mask is equally unsafe: independent
    masks over the same ratio can select different neurons."""
    spec = FAMILIES["qwen2"]
    model = spec.build().eval()
    mgr = spec.manager(model)
    mgr.set("structured_pruning", "ratio", 0.25, criteria="mlp.gate_proj")
    mgr.set("structured_pruning", "ratio", 0.25, criteria="mlp.up_proj")
    with pytest.raises(ValueError, match="Coupled pruning"):
        mgr.apply(hard=True)


# --------------------------------------------------------------------------- #
# #5  QNN export: no capability guard, undeclared py-cpuinfo dep crashes on import
#     Fixed: is_qnn_available() probes cheap preconditions without importing the
#     Qualcomm backend package, and export_with_qnn() checks it up front (with a
#     try/except fallback around the real Qualcomm import) instead of letting a
#     bare "install py-cpuinfo" ImportError surface mid-export.
# --------------------------------------------------------------------------- #
def test_qnn_unavailable_raises_clear_error_instead_of_cpuinfo_importerror():
    from transformersurgeon.export.executorch_exporters.qnn import (
        QNNExportConfig,
        export_with_qnn,
        is_qnn_available,
    )

    # This box has no Qualcomm QNN SDK — is_qnn_available() must say so without
    # raising (it must not import executorch.backends.qualcomm to find out).
    assert is_qnn_available() is False

    config = QNNExportConfig(output_path="/tmp/unused.pte", backend="qnn")
    with pytest.raises(RuntimeError, match="QNN backend unavailable"):
        export_with_qnn(None, config=config)
