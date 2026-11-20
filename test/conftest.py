"""Shared pytest configuration for the transformer-surgeon suite.

Responsibilities:
- make ``test._helpers`` importable regardless of pytest import mode by putting the
  ``test/`` directory on ``sys.path`` (so ``import _helpers`` works everywhere);
- print a one-line capability banner so a run's skips are self-explanatory;
- provide small shared fixtures (device, tmp output dir);
- ``--kv-model NAME``: run the KV-cache export tests on a pretrained model instead of the tiny one
  (``_helpers.model_factory.kv_model``).
"""
from __future__ import annotations

import os
import sys

import pytest

# Ensure `import _helpers` resolves to test/_helpers no matter the import mode.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def pytest_addoption(parser):
    parser.addoption(
        "--kv-model", metavar="NAME", default=None,
        help="pretrained Qwen2-family causal LM (e.g. Qwen/Qwen2.5-0.5B) for the KV-cache export tests "
             "(test/export/test_kv_precision.py) instead of their tiny random Qwen2; downloaded on first use",
    )


def pytest_configure(config):
    from _helpers import model_factory

    model_factory.KV_MODEL = config.getoption("--kv-model")


def pytest_report_header(config):  # noqa: D401 - pytest hook
    from _helpers import capabilities

    header = [f"transformer-surgeon capabilities: {capabilities.summary()}"]
    if config.getoption("--kv-model"):
        header.append(f"KV-cache export tests on the pretrained model {config.getoption('--kv-model')}")
    return header


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
