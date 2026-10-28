"""
TensorRT via ONNX: export a portable ONNX + manifest, and build the engine.

The ONNX export is target-agnostic (``export/backends/onnx``); this module only adds the
TensorRT build. Because an engine only runs on the GPU architecture and
TensorRT version it was built with, the ONNX + manifest pair is the artifact
to ship; build the engine where it will run, e.g. on the Jetson:

    python -m transformersurgeon.export.backends.tensorrt.exporter model.manifest.json

Optimization profiles follow the manifest: profile 0 is decode (seq = 1, so
TensorRT tunes for single-token GEMV), profile 1 -- present when the graph was
exported with max_input_len > 1 -- is prompt prefill (seq = 1..max_input_len).

Attention: models converted with cache_impl="io_paged" export their fused
paged attention as TensorRT Edge-LLM's AttentionPlugin (XQA decode kernels, cost
grows with the context, not max_cache_len; see edgellm_attention.py). Other
cache implementations keep TensorRT's own MatMul/Softmax attention.
"""

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field

from ...exporter import Exporter, register_exporter
from ...operations import config_default
from ..onnx import ONNXExportConfig, ONNXExportResult, export_onnx
from .edgellm_attention import attention_translations, edgellm_attention_pass
from .edgellm_int4 import edgellm_int4_pass, resolve_plugin_libraries
from .engine import EngineBuildConfig, OptimizationProfile, build_engine

DECODE_PROFILE = 0
PREFILL_PROFILE = 1


@dataclass
class TensorRTExportResult(ONNXExportResult):
    engine_path: str | None = None


@dataclass
class TensorRTExportConfig(ONNXExportConfig):
    """ONNXExportConfig plus the (optional) local TensorRT build.

    build_engine       - also build an engine for the local GPU. Leave False
                         when the target is another device (build it there).
    engine_path        - default: <onnx stem>.engine next to the ONNX file.
    optimization_level - TensorRT builder optimization level (None = default).
    workspace_bytes    - builder workspace limit (0 = TensorRT default).
    timing_cache_path  - persisted timing cache for faster rebuilds.
    plugin_libraries   - TensorRT plugin .so files needed by custom ONNX ops.
    int4_backend       - "dequantize" (plain TensorRT DequantizeLinear, no
                         extra dependency) or "edgellm_plugin" (TensorRT
                         Edge-LLM's INT4 GEMM plugin for eligible layers: fast
                         single-token INT4, but the engine then needs
                         libNvInfer_edgellm_plugin.so; see edgellm_int4.py).
    """

    backend: str = "tensorrt"
    build_engine: bool = True
    int4_backend: str = field(default_factory=lambda: config_default("tensorrt", "int4_backend"))
    engine_path: str | None = None
    optimization_level: int | None = None
    workspace_bytes: int = 0
    timing_cache_path: str | None = None
    plugin_libraries: tuple[str, ...] = ()
    # Run MatMul(q, k^T) -> Add(mask) -> Softmax in fp32: TensorRT 10.16 fuses it
    # into an fp16-accumulating kernel that overflows on models with large
    # attention logits (NaN logits otherwise). See attention_passes.py.
    fp32_attention_scores: bool = True
    # Accumulate the GEMMs of LRD-factorized linears in fp32 (weights/activations stay
    # fp16): TensorRT's M=1 fp16 GEMV kernels accumulate in fp16, which LRD models do not
    # tolerate. See precision_passes.py.
    fp32_accumulate_lrd: bool = True


def engine_profiles(manifest: dict) -> list[OptimizationProfile]:
    """Decode profile (+ prefill profile when the graph has a dynamic seq)."""
    profiles = [OptimizationProfile(shapes={"input_ids": ((1,), (1,), (1,))})]
    max_input_len = int(manifest.get("max_input_len", 1))
    if max_input_len > 1:
        opt = min(128, max_input_len)
        profiles.append(OptimizationProfile(shapes={"input_ids": ((1,), (opt,), (max_input_len,))}))
    return profiles


def build_engine_from_manifest(
    manifest_path: str,
    engine_path: str | None = None,
    *,
    optimization_level: int | None = None,
    workspace_bytes: int = 0,
    timing_cache_path: str | None = None,
    plugin_libraries: tuple[str, ...] = (),
    verbose: bool = False,
) -> str:
    """Build the engine for an exported ONNX (on the machine that will run it)."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    plugin_libraries = resolve_plugin_libraries(manifest, tuple(plugin_libraries))
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), manifest["onnx_file"])
    engine_path = engine_path or os.path.splitext(onnx_path)[0] + ".engine"
    serialized = build_engine(onnx_path, EngineBuildConfig(
        profiles=engine_profiles(manifest),
        optimization_level=optimization_level,
        workspace_bytes=workspace_bytes,
        timing_cache_path=timing_cache_path,
        plugin_libraries=tuple(plugin_libraries),
        verbose=verbose,
    ))
    with open(engine_path, "wb") as f:
        f.write(serialized)
    return engine_path


@register_exporter
class TensorRTExporter(Exporter):
    """ONNX export with TensorRT-specific translations and graph passes, then the engine build."""

    backend = "tensorrt"
    config_class = TensorRTExportConfig

    def lower(self, components, config):
        return export_tensorrt(components, config)


def export_tensorrt(components: dict, config: TensorRTExportConfig) -> TensorRTExportResult:
    if config.int4_backend == "edgellm_plugin":
        config = dataclasses.replace(config, graph_passes=[*config.graph_passes, edgellm_int4_pass])
    config = dataclasses.replace(config, custom_translations={**attention_translations(), **config.custom_translations},
                                 graph_passes=[*config.graph_passes, edgellm_attention_pass])
    if config.fp32_accumulate_lrd:
        from .precision_passes import fp32_accumulate_lrd
        config = dataclasses.replace(config, graph_passes=[*config.graph_passes, fp32_accumulate_lrd])
    if config.fp32_attention_scores:
        from .attention_passes import fp32_attention_scores
        config = dataclasses.replace(config, graph_passes=[fp32_attention_scores, *config.graph_passes])
    onnx_result = export_onnx(components, config)
    result = TensorRTExportResult(**vars(onnx_result))
    if config.build_engine:
        result.engine_path = build_engine_from_manifest(
            onnx_result.manifest_path,
            config.engine_path,
            optimization_level=config.optimization_level,
            workspace_bytes=config.workspace_bytes,
            timing_cache_path=config.timing_cache_path,
            plugin_libraries=config.plugin_libraries,
            verbose=config.verbose,
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from a tsurgeon ONNX manifest.")
    parser.add_argument("manifest", help="<model>.manifest.json written by the ONNX exporter")
    parser.add_argument("--engine", default=None, help="output engine path (default: next to the ONNX)")
    parser.add_argument("--optimization-level", type=int, default=None)
    parser.add_argument("--timing-cache", default=None)
    parser.add_argument("--plugin", action="append", default=[], help="TensorRT plugin library (repeatable)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    path = build_engine_from_manifest(
        args.manifest, args.engine,
        optimization_level=args.optimization_level,
        timing_cache_path=args.timing_cache,
        plugin_libraries=tuple(args.plugin),
        verbose=args.verbose,
    )
    print(path)


if __name__ == "__main__":
    main()


__all__ = [
    "DECODE_PROFILE",
    "PREFILL_PROFILE",
    "TensorRTExportConfig",
    "TensorRTExportResult",
    "build_engine_from_manifest",
    "engine_profiles",
    "TensorRTExporter",
    "export_tensorrt",
]
