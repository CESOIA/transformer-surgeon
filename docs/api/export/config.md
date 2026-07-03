# BackendExportConfig

Base configuration shared by every backend exporter (`output_path`, `backend`,
`max_seq_len`, weight-mismatch checking, `convert_options`, etc.). Backend
configs (`XNNPACKExportConfig`, `QNNExportConfig`, `TensorRTExportConfig`)
extend `export.common.ExporterConfig`, which extends this class.

::: transformersurgeon.export.config.BackendExportConfig
