"""Export of compressed models to device backends.

Layers of this package:

- generic steps shared by every backend: ``convert.py`` (Hugging Face model ->
  ``blocks/`` stacks), ``wrapper.py`` (traceable wrapper), ``quant_info.py``
  (quantisation read from the layers), ``pt2e.py`` (PT2E quantisation flow),
  ``checks.py``, ``manifest.py``, ``result.py``;
- ``operations.py``: the dictionary that maps each block type to the device
  operation used by each backend (defaults and supported alternatives);
- ``exporter.py``: the generic :class:`Exporter` and ``export_to_backend``;
- ``backends/``: one exporter per backend (XNNPACK, QNN, ONNX, TensorRT).
"""
from .config import BackendExportConfig
from .convert import convert_for_export, resolve_convert_options, DEFAULT_CONVERT_OPTIONS
from .operations import BACKEND_OPERATIONS, BACKENDS, default_for, options_for
from .exporter import Exporter, EXPORTERS, register_exporter, get_exporter, export_to_backend
from .wrapper import LLMWrapper, build_wrapper, build_zero_caches, build_example_inputs
from .quant_info import extract_layer_quant_info
from .pt2e import calibrate_pt2e_observers, inject_scales_into_pt2e_observers
from .manifest import build_llm_manifest, load_llm_manifest, zero_caches_from_manifest, manifest_path_for
from .result import BackendExportResult
