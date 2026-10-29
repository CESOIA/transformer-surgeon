"""Configuration shared by every backend exporter."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BackendExportConfig:
    """Options common to every backend. Backend configurations subclass it.

    Attributes:
        output_path: Path of the artifact to write.
        backend: Backend name (``"xnnpack"``, ``"qnn"``, ``"onnx"``, ``"tensorrt"``).
        max_seq_len: Sequence budget; also the KV-cache length unless
            ``convert_options["max_cache_len"]`` is given.
        weight_mismatch_eps: Tolerance of the post-export weight check.
        run_weight_mismatch_check: Compare exported weights with the source ones.
        convert_options: Atom choices for the export-ready model (``cache_impl``,
            ``attn_impl``, ``rmsnorm_impl``, ...). Missing entries take the
            backend default from ``export/operations.py``.
        check_ir_validity: Let ExecuTorch validate the edge program.
        verbose: Print progress and error statistics.
        float_type: Cast every component to this dtype before export (``None``
            keeps each component's dtype).
    """

    output_path: str
    backend: str
    max_seq_len: int = 2048
    weight_mismatch_eps: float = 1e-4
    run_weight_mismatch_check: bool = True
    convert_options: dict[str, Any] = field(default_factory=dict)
    check_ir_validity: bool = True
    verbose: bool = False
    float_type: torch.dtype | None = None


__all__ = ["BackendExportConfig"]
