# LRDer

Low-rank decomposition compressor. Supports three methods:

| Method | Calibration needed | Notes |
|---|---|---|
| `svd` | None | Plain truncated SVD |
| `svd-llm-v2` | `covariance` | Whitened SVD using activation covariance |
| `aa-svd` | `cross_covariance`, `shifted_covariance` | Requires cascade calibration mode |

::: transformersurgeon.compression.lrd.LRDer
