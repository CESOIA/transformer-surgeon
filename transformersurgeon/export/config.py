from dataclasses import dataclass, field
from typing import Any

@dataclass
class BackendExportConfig:
    """General-purpose backend export configuration."""

    output_path: str
    backend: str
    adapter: Any | None = None
    convert_options: dict[str, Any] = field(default_factory=lambda: {"use_sdpa": False})
    check_ir_validity: bool = True
    verbose: bool = False
    allow_backend_fallback: bool = False

__all__ = [
    "BackendExportConfig",
]