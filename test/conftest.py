"""Shared pytest configuration for the transformer-surgeon suite.

Responsibilities:
- make ``test._helpers`` importable regardless of pytest import mode by putting the
  ``test/`` directory on ``sys.path`` (so ``import _helpers`` works everywhere);
- print a one-line capability banner so a run's skips are self-explanatory;
- provide small shared fixtures (device, tmp output dir).
"""
from __future__ import annotations

import os
import sys

import pytest

# Ensure `import _helpers` resolves to test/_helpers no matter the import mode.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def pytest_report_header(config):  # noqa: D401 - pytest hook
    from _helpers import capabilities

    return f"transformer-surgeon capabilities: {capabilities.summary()}"


@pytest.fixture(scope="session")
def torch_device():
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def out_dir(tmp_path):
    """A throwaway directory for export artifacts (.pte / .ep / HF dirs)."""
    d = tmp_path / "artifacts"
    d.mkdir(exist_ok=True)
    return str(d)
