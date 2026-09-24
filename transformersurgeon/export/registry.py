from .executorch_exporters.xnnpack import XNNPACKExportConfig, export_with_xnnpack
from .executorch_exporters.qnn import QNNExportConfig, export_with_qnn
from .onnx import ONNXExportConfig, export_with_onnx
from .tensorrt import (
    TensorRTExportConfig,
    TensorRTONNXExportConfig,
    export_with_tensorrt,
    export_with_tensorrt_onnx,
)

EXPORT_ROUTINES = {
    "xnnpack": {
        "export": export_with_xnnpack,
        "config_class": XNNPACKExportConfig,
    },
    "qnn": {
        "export": export_with_qnn,
        "config_class": QNNExportConfig,
    },
    # torch-tensorrt: engine compiled in-process for the local GPU.
    "tensorrt": {
        "export": export_with_tensorrt,
        "config_class": TensorRTExportConfig,
    },
    # Portable ONNX + manifest (any ONNX consumer).
    "onnx": {
        "export": export_with_onnx,
        "config_class": ONNXExportConfig,
    },
    # ONNX + plain TensorRT engine built from it (e.g. Jetson: build on device).
    "tensorrt_onnx": {
        "export": export_with_tensorrt_onnx,
        "config_class": TensorRTONNXExportConfig,
    },
}
