# Covariance Summaries

Three concrete `CalibrationSummary` implementations for computing activation
covariance matrices used by `svd-llm-v2` and `aa-svd`.

| Summary | Raw streams | Used by |
|---|---|---|
| `CovarianceSummary` | `activation` | `svd-llm-v2` |
| `ShiftedCovarianceSummary` | `activation_shifted` | `aa-svd` |
| `CrossCovarianceSummary` | `activation` + `activation_shifted` | `aa-svd` |

::: transformersurgeon.calibration.summaries.covariance.CovarianceSummary

::: transformersurgeon.calibration.summaries.covariance.ShiftedCovarianceSummary

::: transformersurgeon.calibration.summaries.covariance.CrossCovarianceSummary
