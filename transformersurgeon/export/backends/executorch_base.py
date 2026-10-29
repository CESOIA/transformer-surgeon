"""Steps shared by the ExecuTorch backends (XNNPACK, Qualcomm QNN)."""

import warnings
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from ..checks import logit_error_stats, reference_logits
from ..exporter import Exporter
from ..result import BackendExportResult, finalize_export_result


@dataclass
class ExecuTorchExportResult(BackendExportResult):
    """Result with a ``pte_path`` alias for the written ``.pte`` program."""

    @property
    def pte_path(self) -> str:
        return self.output_path


def run_simple_inference_stats(
    wrapper: nn.Module,
    pte_path: str,
    inference_inputs: tuple[Any, ...],
) -> dict[str, float] | None:
    """Run one forward pass through both the float wrapper and the exported program
    and return absolute/MSE error statistics.  Returns None if the ExecuTorch
    runtime is not installed.
    """
    try:
        from executorch.runtime import Runtime
    except (ImportError, ModuleNotFoundError) as e:
        warnings.warn(
            f"ExecuTorch runtime not available; skipping inference stats. ({e})",
            stacklevel=2,
        )
        return None

    y_ref = reference_logits(wrapper, inference_inputs)

    # Load and run the exported program.
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method_names = _get_method_names(program)
    if "forward" in method_names:
        method_name = "forward"
    elif method_names:
        method_name = method_names[0]
        warnings.warn(
            f"'forward' not found in exported program; using '{method_name}' instead.",
            stacklevel=2,
        )
    else:
        raise RuntimeError(f"Exported program has no runnable methods. pte_path='{pte_path}'")

    # The exported program takes flat tensor inputs; io modes pass the KV caches
    # as nested lists, so flatten them into the runtime argument list.
    flat_inputs: list[Any] = []
    for arg in inference_inputs:
        if isinstance(arg, (list, tuple)):
            flat_inputs.extend(arg)
        else:
            flat_inputs.append(arg)

    return logit_error_stats(y_ref, program.load_method(method_name).execute(flat_inputs))


def _get_method_names(program: Any) -> list[str]:
    attr = getattr(program, "method_names", None)
    if callable(attr):
        return list(attr())
    return list(attr) if attr is not None else []


class ExecuTorchExporter(Exporter):
    """Base of the ExecuTorch exporters: writes the program and runs the shared final checks."""

    def write_program(self, edge, config, *, alloc_graph_input, alloc_graph_output=True):
        """Lower the edge program to an ExecuTorch program and write it to ``config.output_path``.

        ``alloc_graph_input=False`` keeps graph inputs caller-owned (zero copy);
        ``True`` lets the memory planner own them (needed by QNN).
        """
        from executorch.exir import ExecutorchBackendConfig
        from executorch.exir.passes import MemoryPlanningPass

        planning = MemoryPlanningPass(alloc_graph_input=alloc_graph_input, alloc_graph_output=alloc_graph_output)
        program = edge.to_executorch(config=ExecutorchBackendConfig(memory_planning_pass=planning))
        with open(config.output_path, "wb") as f:
            f.write(program.buffer)

    def finalize(self, config, *, precision, wrapper, example_inputs, model_config, exported_for_mismatch,
                 run_weight_mismatch_check=None):
        """Weight check, error statistics through the ExecuTorch runtime, manifest, result."""
        if run_weight_mismatch_check is None:
            run_weight_mismatch_check = config.run_weight_mismatch_check
        return finalize_export_result(
            output_path=config.output_path,
            backend=self.backend,
            precision=precision,
            wrapper=wrapper,
            exported_for_mismatch=exported_for_mismatch,
            run_weight_mismatch_check=run_weight_mismatch_check,
            weight_mismatch_eps=config.weight_mismatch_eps,
            verbose=config.verbose,
            inference_stats_fn=lambda: run_simple_inference_stats(wrapper, config.output_path, example_inputs),
            result_cls=ExecuTorchExportResult,
            model_config=model_config,
        )


__all__ = ["ExecuTorchExportResult", "ExecuTorchExporter", "run_simple_inference_stats"]
