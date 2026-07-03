# XNNPACK Export

Exports to an ExecuTorch `.pte` file targeting the XNNPACK delegate. Mixed
INT8/INT4 + float export is driven by per-layer compression metadata already
on the model.

::: transformersurgeon.export.executorch_exporters.xnnpack.xnnpack_export
