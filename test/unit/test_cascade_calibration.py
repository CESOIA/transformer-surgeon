"""Cascade-calibration mode: the ``no_cascade_calibration`` indexing guard.

BERT is marked ``no_cascade_calibration=True`` in
``models/bert_c/indexing_bert_c.py`` because its grouped/bidirectional layer
layout is not compatible with the block-wise cascade algorithm in
``utils/cascade.py`` (see FRAMEWORK_PROBLEMS.md). That flag used to be silently
ignored: ``apply_cascade`` built a ``selected_by_block`` dict but never
populated it (a leftover from a prior refactor that dropped the
``scheme.block_name`` population loop because that attribute doesn't exist),
so the guard's ``if len(selected_by_block.get(block_name, [])) == 0: continue``
branch always continued and the ValueError could never fire. This meant
requesting cascade calibration on BERT silently ran cascade's block-wise
calibration instead of failing loudly, even though the family was never
validated against it.

These tests pin the fixed behavior: cascade mode must raise a clear,
actionable error for BERT, and must keep working normally for families that
are not flagged.
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from _helpers.model_factory import FAMILIES

pytestmark = pytest.mark.unit


def _tiny_calibration_loader(seq_len=8, num_batches=4):
    examples = [{"input_ids": torch.randint(0, 256, (seq_len,))} for _ in range(num_batches)]
    return DataLoader(examples, batch_size=1, shuffle=False)


def test_bert_cascade_calibration_raises_clear_error():
    """BERT is indexed as ``no_cascade_calibration``; cascade mode must refuse it."""
    spec = FAMILIES["bert"]
    model = spec.build().eval()
    mgr = spec.manager(model)

    mgr.set("lrd", "method", "aa-svd", criteria=spec.lrd_criteria)
    mgr.set("lrd", "rank", 8, criteria=spec.lrd_criteria)
    mgr.set_calibration_mode(mode="cascade")
    mgr.set_calibration_data(_tiny_calibration_loader())

    with pytest.raises(ValueError, match="no_cascade_calibration"):
        mgr.apply(hard=False, show_progress=False)


def test_modernbert_cascade_calibration_raises_clear_error():
    """ModernBERT is also indexed as ``no_cascade_calibration``.

    Its alternating global/local attention layers each need a per-layer-type
    rotary embedding call that cascade's single-flow position-embedding
    injection doesn't model, so cascade mode is unsupported here too.
    """
    spec = FAMILIES["modernbert"]
    model = spec.build().eval()
    mgr = spec.manager(model)

    mgr.set("lrd", "method", "aa-svd", criteria=spec.lrd_criteria)
    mgr.set("lrd", "rank", 8, criteria=spec.lrd_criteria)
    mgr.set_calibration_mode(mode="cascade")
    mgr.set_calibration_data(_tiny_calibration_loader())

    with pytest.raises(ValueError, match="no_cascade_calibration"):
        mgr.apply(hard=False, show_progress=False)


def test_qwen2_cascade_calibration_still_works():
    """Sanity check that the BERT guard fix didn't disable cascade for other families."""
    spec = FAMILIES["qwen2"]
    model = spec.build().eval()
    mgr = spec.manager(model)

    mgr.set("lrd", "method", "aa-svd", criteria=spec.lrd_criteria)
    mgr.set("lrd", "rank", 8, criteria=spec.lrd_criteria)
    mgr.set_calibration_mode(mode="cascade")
    mgr.set_calibration_data(_tiny_calibration_loader())

    mgr.apply(hard=False, show_progress=False)
    with torch.no_grad():
        model(**spec.sample_inputs(model))
