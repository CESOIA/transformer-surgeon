# UnstructuredPruner

Weight-element sparsity compressor. Supports three scoring methods:

| Method | Calibration needed | Notes |
|---|---|---|
| `magnitude` | None | `abs(weight)` |
| `gradient` | `weight_grad` | `abs(weight * weight_grad)` |
| `random` | None | Random scores, for ablations |

See [Compression Methods Reference](../../compression_methods.md#3-unstructured_pruning-weight-level-sparsity)
for the full parameter table and the STE fine-tuning mask lifecycle.

::: transformersurgeon.compression.unstructured_pruning.UnstructuredPruner
