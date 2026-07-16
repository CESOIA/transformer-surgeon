# StructuredPruner

Output-neuron (row) removal compressor. Supports three scoring methods:

| Method | Calibration needed | Notes |
|---|---|---|
| `magnitude` | None | L2 norm of each output row |
| `gradient` | `weight_grad` | L2 norm of `weight * weight_grad` |
| `random` | None | Random scores, for ablations |

See [Compression Methods Reference](../../compression_methods.md#2-structured_pruning-output-neuron-removal)
for the full parameter table (`granularity`, `repeated_pattern`,
`coupled_repeated_pattern`, `reduce_op`, `share_mask`) and the grouping/coupling
model.

::: transformersurgeon.compression.structured_pruning.StructuredPruner
