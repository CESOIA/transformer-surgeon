# Quantizer

Fixed-point (and binary) weight/activation quantization compressor. Supports
two weight methods:

| Method | Calibration needed | Notes |
|---|---|---|
| `vanilla` | None | Max-abs (or sign+mean-abs for binary) round-to-nearest |
| `gptq` | `covariance` | Hessian-based error-propagating quantization (Frantar et al., 2022) |

Activation quantization (`precision_activation != "full"`) is always
fake-quant, applied via forward hooks, and is independent of the weight
`method`/`precision`.

See [Compression Methods Reference](../../compression_methods.md#4-quantization-reduced-precision-weights-and-activations)
for the full parameter table, soft-vs-hard apply semantics, and the LRD
interaction.

::: transformersurgeon.compression.quantization.Quantizer
