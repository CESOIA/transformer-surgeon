"""Shared test helpers for the transformer-surgeon test suite.

Two public surfaces:

- ``capabilities`` — runtime feature detection (CUDA / executorch / tensorrt / qnn)
  used to build ``pytest.mark.skipif`` guards so a test run degrades gracefully on
  machines that lack an optional export backend.
- ``model_factory`` — tiny, random-weight compressed models for every supported
  family, so the manager/compression/convert code paths can be exercised without
  downloading multi-GB checkpoints.
"""

from . import capabilities, model_factory  # noqa: F401
