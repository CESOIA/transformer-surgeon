"""
Plain-TensorRT engine building and execution (ONNX path).

This is the TensorRT half of the ONNX export path (``export/onnx``): the ONNX
file is the portable artifact, and an engine is built from it *on the machine
that will run it* (a TensorRT engine is tied to the GPU architecture and the
TensorRT version it was built with -- an x86 build does not run on a Jetson).
Nothing here depends on torch-tensorrt; only the ``tensorrt`` Python package
and torch (for device buffers) are needed.

  * build_engine()        ONNX -> serialized engine (strongly typed).
  * TensorRTEngineRunner   Binds torch CUDA tensors to an engine's I/O, runs it,
                           and optionally replays it through a CUDA graph.
"""

import ctypes
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import torch


def _trt():
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "The ONNX->TensorRT path needs the `tensorrt` Python package "
            "(e.g. `pip install tensorrt-cu13==<version matching the target>`)."
        ) from e
    return trt


_TRT_TO_TORCH_DTYPE = {
    "FLOAT": torch.float32,
    "HALF": torch.float16,
    "BF16": torch.bfloat16,
    "INT8": torch.int8,
    "INT32": torch.int32,
    "INT64": torch.int64,
    "BOOL": torch.bool,
    "UINT8": torch.uint8,
}


def _logger(verbose: bool):
    trt = _trt()
    return trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)


def load_plugin_libraries(paths: Iterable[str]) -> None:
    """dlopen TensorRT plugin libraries so their creators self-register."""
    for path in paths:
        ctypes.CDLL(os.path.abspath(path), mode=ctypes.RTLD_GLOBAL)


@dataclass
class OptimizationProfile:
    """One TensorRT optimization profile.

    shapes - {input_name: (min_shape, opt_shape, max_shape)} for inputs with
             dynamic dimensions.
    values - {input_name: (min_values, opt_values, max_values)} for *shape
             tensor* inputs (host-side int tensors whose values drive shapes,
             e.g. an attention span used as a Slice end).
    """

    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = field(default_factory=dict)
    values: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = field(default_factory=dict)


@dataclass
class EngineBuildConfig:
    """Options for build_engine().

    profiles             - optimization profiles for dynamic inputs; empty for
                           fully static graphs.
    workspace_bytes      - builder workspace limit (0 = TensorRT default).
    optimization_level   - builder optimization level 0-5 (None = default, 3).
    plugin_libraries     - plugin .so files to load before parsing.
    timing_cache_path    - persisted timing cache (speeds up rebuilds).
    profiling_verbosity  - "layer_names_only" (default) or "detailed" (keeps
                           tactic/kernel names for IEngineInspector / profiles).
    """

    profiles: list[OptimizationProfile] = field(default_factory=list)
    workspace_bytes: int = 0
    optimization_level: int | None = None
    plugin_libraries: tuple[str, ...] = ()
    timing_cache_path: str | None = None
    profiling_verbosity: str = "layer_names_only"
    verbose: bool = False


def build_engine(onnx_path: str, config: EngineBuildConfig | None = None) -> bytes:
    """Parse an ONNX model and build a strongly typed TensorRT engine.

    Strong typing is the only mode TensorRT 11 supports and the recommended one
    since 10.12: tensor precisions come from the ONNX graph itself (fp16
    weights, explicit Q/DQ), never from builder precision flags.
    """
    trt = _trt()
    config = config or EngineBuildConfig()
    load_plugin_libraries(config.plugin_libraries)

    logger = _logger(config.verbose)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(os.path.abspath(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")

    build_config = builder.create_builder_config()
    if config.workspace_bytes:
        build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, config.workspace_bytes)
    if config.optimization_level is not None:
        build_config.builder_optimization_level = config.optimization_level
    build_config.profiling_verbosity = getattr(trt.ProfilingVerbosity, config.profiling_verbosity.upper())
    for spec in config.profiles:
        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in spec.shapes.items():
            profile.set_shape(name, lo, opt, hi)
        for name, (lo, opt, hi) in spec.values.items():
            profile.set_shape_input(name, lo, opt, hi)
        build_config.add_optimization_profile(profile)

    cache = None
    if config.timing_cache_path:
        blob = b""
        if os.path.isfile(config.timing_cache_path):
            with open(config.timing_cache_path, "rb") as f:
                blob = f.read()
        cache = build_config.create_timing_cache(blob)
        build_config.set_timing_cache(cache, ignore_mismatch=False)

    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path} (see log above).")
    if cache is not None:
        with open(config.timing_cache_path, "wb") as f:
            f.write(memoryview(cache.serialize()))
    return bytes(serialized)


class TensorRTEngineRunner:
    """Execute a TensorRT engine on torch CUDA tensors.

    Buffers are owned by the caller: bind() records their addresses on the
    execution context, run() enqueues on the given (or current) torch stream.
    Because addresses are fixed after bind(), the same step can be captured
    into a CUDA graph (capture_graph()) and replayed with replay(); the caller
    then updates inputs *in place* (``tensor.copy_(...)``) between replays
    instead of re-binding.

    Shape-tensor inputs (engine.is_shape_inference_io) live in host memory and
    must be bound as CPU tensors. Their values are baked into a captured
    graph, so capture one graph per distinct value under its own ``key``
    (e.g. one per attention-span bucket) and replay(key) the matching one.
    """

    def __init__(
        self,
        engine: "bytes | str | TensorRTEngineRunner",
        *,
        profile: int = 0,
        device: str | torch.device = "cuda",
        plugin_libraries: Iterable[str] = (),
    ):
        """``engine`` is serialized bytes, an engine file path, or another
        runner whose deserialized engine (and weights) this one shares --
        e.g. one runner per optimization profile of the same engine."""
        trt = _trt()
        self.device = torch.device(device)
        if isinstance(engine, TensorRTEngineRunner):
            self._runtime = engine._runtime
            self.engine = engine.engine
        else:
            load_plugin_libraries(plugin_libraries)
            if isinstance(engine, str):
                with open(engine, "rb") as f:
                    engine = f.read()
            logger = _logger(False)
            trt.init_libnvinfer_plugins(logger, "")
            self._runtime = trt.Runtime(logger)
            self.engine = self._runtime.deserialize_cuda_engine(engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine.")
        self.context = self.engine.create_execution_context()
        self.profile = profile
        if profile:
            stream = torch.cuda.current_stream(self.device)
            self.context.set_optimization_profile_async(profile, stream.cuda_stream)
            stream.synchronize()
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)
        self._bound: dict[str, torch.Tensor] = {}
        self._graphs: dict[Any, torch.cuda.CUDAGraph] = {}

    def is_shape_input(self, name: str) -> bool:
        return name in self.inputs and self.engine.is_shape_inference_io(name)

    def dtype(self, name: str) -> torch.dtype:
        return _TRT_TO_TORCH_DTYPE[self.engine.get_tensor_dtype(name).name]

    def aliased_input(self, output_name: str) -> str | None:
        """Input that TensorRT requires to share memory with ``output_name``
        (in-place layers such as KV-cache update), or None."""
        name = self.engine.get_aliased_input_tensor(output_name)
        return name or None

    def output_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.context.get_tensor_shape(name))

    def bind(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Record tensor addresses (and dynamic input shapes) on the context.

        Invalidates captured graphs: they hold the previous addresses.
        """
        for name, tensor in tensors.items():
            if not tensor.is_contiguous():
                raise ValueError(f"TensorRT I/O '{name}' must be contiguous.")
            expected = self.dtype(name)
            if tensor.dtype != expected:
                raise TypeError(f"TensorRT I/O '{name}' expects {expected}, got {tensor.dtype}.")
            if self.is_shape_input(name) != (tensor.device.type == "cpu"):
                where = "host (CPU)" if self.is_shape_input(name) else "device (CUDA)"
                raise ValueError(f"TensorRT I/O '{name}' must be a {where} tensor.")
            if name in self.inputs and -1 in tuple(self.engine.get_tensor_shape(name)):
                self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
            self._bound[name] = tensor
        self._graphs.clear()

    def allocate_outputs(self, skip: Iterable[str] = ()) -> dict[str, torch.Tensor]:
        """Allocate (and bind) device buffers for every output not in ``skip``."""
        skip = set(skip)
        out = {
            name: torch.empty(self.output_shape(name), dtype=self.dtype(name), device=self.device)
            for name in self.outputs if name not in skip
        }
        self.bind(out)
        return out

    def run(self, stream: torch.cuda.Stream | None = None) -> None:
        stream = stream or torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT enqueue failed.")

    def capture_graph(self, key: Any = None, *, after: Callable[[], None] | None = None, warmup: bool = True) -> None:
        """Capture one run() (followed by ``after()``, e.g. token selection on
        the logits) into a CUDA graph under ``key``.

        Addresses and shape-input values must already be final for this key.
        Neither the warm-up (TensorRT allocates lazily on its first enqueue)
        nor the capture must change state the caller cares about, so pass
        ``warmup=False`` once the context has already run, and snapshot/restore
        anything ``after`` mutates (the capture itself does not execute).
        """
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        if warmup:
            with torch.cuda.stream(stream):
                self.run(stream)
            stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.run(stream)
            if after is not None:
                after()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._graphs[key] = graph

    def has_graph(self, key: Any = None) -> bool:
        return key in self._graphs

    def replay(self, key: Any = None) -> None:
        graph = self._graphs.get(key)
        if graph is None:
            raise RuntimeError(f"No CUDA graph captured for key {key!r} (capture_graph() after every bind()).")
        graph.replay()


def time_cuda(fn, *, warmup: int = 20, iters: int = 200) -> float:
    """Median per-call wall time in ms of a CUDA callable, via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


__all__ = [
    "OptimizationProfile",
    "EngineBuildConfig",
    "build_engine",
    "load_plugin_libraries",
    "TensorRTEngineRunner",
    "time_cuda",
]
