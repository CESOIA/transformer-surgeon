
from .executorch_exporters.qnn import QNNExportConfig, export_with_qnn
from .executorch_exporters.xnnpack import XNNPACKExportConfig, export_with_xnnpack

EXPORT_ROUTINES = {
    "xnnpack": {
        "export": export_with_xnnpack,
        "config_class": XNNPACKExportConfig,
    },
    "qnn": {
        "export": export_with_qnn,
        "config_class": QNNExportConfig,
    },
}
