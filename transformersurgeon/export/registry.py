
from .executorch_exporters.xnnpack import XNNPACKExportConfig, export_with_xnnpack

EXPORT_ROUTINES = {
    "xnnpack": {
        "export": export_with_xnnpack,
        "config_class": XNNPACKExportConfig,
    },
}
