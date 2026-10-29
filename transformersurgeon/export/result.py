"""Export result container and the generic final step of every exporter."""

import json
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import torch.nn as nn

from .checks import find_weight_mismatches
from .manifest import build_llm_manifest, manifest_path_for


@dataclass
class BackendExportResult:
    """Backend-agnostic export result.

    ``output_path`` is the on-disk artifact (an ExecuTorch ``.pte`` file, a
    TensorRT engine/program, …).  Backends may subclass this to add
    backend-flavoured aliases (see ExecuTorchExportResult.pte_path).
    """

    output_path: str
    backend: str
    precision: str
    weight_mismatches: list[dict[str, Any]]
    inference_stats: dict[str, float] | None = None
    manifest_path: str = ""  # <artifact stem>.manifest.json (see build_llm_manifest)


def finalize_export_result(
    *,
    output_path: str,
    backend: str,
    precision: str,
    wrapper: nn.Module,
    exported_for_mismatch: Any | None,
    run_weight_mismatch_check: bool,
    weight_mismatch_eps: float,
    verbose: bool,
    inference_stats_fn: Callable[[], dict[str, float] | None] | None = None,
    result_cls: type = BackendExportResult,
    model_config: Any | None = None,
    manifest: dict[str, Any] | None = None,
) -> BackendExportResult:
    """Run post-export sanity checks, write the manifest, and assemble the result.

    This is backend-agnostic.  Backends supply an ``inference_stats_fn`` closure
    (capturing whatever runtime handle they need) to compute float-vs-exported
    error statistics, and may pass a ``result_cls`` subclass of
    BackendExportResult for backend-flavoured field aliases.  The manifest
    (``manifest`` if given, else ``build_llm_manifest``) is written next to
    the artifact as ``<stem>.manifest.json``.
    """

    # --- Weight mismatch check ---
    mismatches: list[dict[str, Any]] = []
    if run_weight_mismatch_check and exported_for_mismatch is not None:
        try:
            mismatches = find_weight_mismatches(wrapper, exported_for_mismatch, eps=weight_mismatch_eps)
        except Exception as exc:
            warnings.warn(f"Weight mismatch check skipped: {exc}", stacklevel=2)
    elif run_weight_mismatch_check and precision == "full":
        warnings.warn(
            "Weight mismatch check skipped for precision='full' (no quantized frozen params).",
            stacklevel=2,
        )

    if mismatches:
        worst = max(mismatches, key=lambda x: x["mean_abs_err"])
        warnings.warn(
            f"Weight mismatch above eps: count={len(mismatches)}, "
            f"worst={worst['matched_weight']} mean_abs_err={worst['mean_abs_err']:.6g} "
            f"(eps={weight_mismatch_eps:.6g})",
            stacklevel=2,
        )

    # --- Inference stats ---
    stats = None
    if verbose and inference_stats_fn is not None:
        stats = inference_stats_fn()
        if stats is not None:
            print(f"[{backend} export] Inference error statistics")
            print(f"  max_abs_err : {stats['max_abs_err']:.8f}")
            print(f"  mean_abs_err: {stats['mean_abs_err']:.8f}")
            print(f"  mse         : {stats['mse']:.10f}")
            print(f"  rmse        : {stats['rmse']:.8f}")

    if manifest is None and hasattr(wrapper, "decoder"):
        manifest = build_llm_manifest(wrapper, model_config, backend=backend, artifact=output_path,
                                      precision=precision)
    written_manifest = ""
    if manifest is not None:
        written_manifest = manifest_path_for(output_path)
        with open(written_manifest, "w") as f:
            json.dump(manifest, f, indent=2)

    return result_cls(
        output_path=output_path,
        backend=backend,
        precision=precision,
        weight_mismatches=mismatches,
        inference_stats=stats,
        manifest_path=written_manifest,
    )


__all__ = ["BackendExportResult", "finalize_export_result"]
