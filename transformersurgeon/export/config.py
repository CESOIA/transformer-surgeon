from dataclasses import dataclass, field, fields
from typing import Any

@dataclass
class BackendExportConfig:
    """General-purpose backend export configuration."""

    output_path: str
    backend: str
    max_seq_len: int = 2048
    weight_mismatch_eps: float = 1e-4
    run_weight_mismatch_check: bool = True
    convert_options: dict[str, Any] = field(default_factory=lambda: {"use_sdpa": False})
    check_ir_validity: bool = True
    verbose: bool = False
    allow_backend_fallback: bool = False

    @classmethod
    def from_kwargs(
        cls,
        *,
        output_path: str,
        backend: str,
        **kwargs,
    ) -> "BackendExportConfig":
        recognized = {
            f.name
            for f in fields(cls)
            if f.init and f.name not in {"output_path", "backend"}
        }
        unknown = sorted(set(kwargs.keys()) - recognized)
        if unknown:
            unknown_str = ", ".join(unknown)
            raise TypeError(f"Unsupported {backend} export options: {unknown_str}")

        init_kwargs = {
            "output_path": output_path,
            "backend": backend,
        }
        for field_name in recognized:
            if field_name in kwargs:
                init_kwargs[field_name] = kwargs[field_name]

        return cls(**init_kwargs)

__all__ = [
    "BackendExportConfig",
]