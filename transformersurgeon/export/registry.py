from .executorch_exporters.xnnpack import XNNPACKExportConfig, export_with_xnnpack
from .executorch_exporters.qnn import QNNExportConfig, export_with_qnn
from .onnx import ONNXExportConfig, export_with_onnx
from .tensorrt import TensorRTExportConfig, export_with_tensorrt

EXPORT_ROUTINES = {
    "xnnpack": {
        "export": export_with_xnnpack,
        "config_class": XNNPACKExportConfig,
    },
    "qnn": {
        "export": export_with_qnn,
        "config_class": QNNExportConfig,
    },
    # Portable ONNX + manifest (any ONNX consumer).
    "onnx": {
        "export": export_with_onnx,
        "config_class": ONNXExportConfig,
    },
    # ONNX + plain TensorRT engine built from it (e.g. Jetson: build on device).
    "tensorrt": {
        "export": export_with_tensorrt,
        "config_class": TensorRTExportConfig,
    },
}
