# LRDer

Low-rank decomposition compressor. Supports three methods:

| Method | Calibration needed | Notes |
|---|---|---|
| `svd` | None | Plain truncated SVD |
| `svd-llm-v2` | `covariance` | Whitened SVD using activation covariance |
| `aa-svd` | `cross_covariance`, `shifted_covariance` | Requires cascade calibration mode |

See [Compression Methods Reference](../../compression_methods.md#1-lrd--low-rank-decomposition)
for the full parameter table and a plain-language explanation of each method.

::: transformersurgeon.compression.lrd.LRDer
