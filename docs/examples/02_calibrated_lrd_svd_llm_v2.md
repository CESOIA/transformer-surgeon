# Calibrated LRD (SVD-LLM-v2)

Loads Qwen2-0.5B, builds a small WikiText-2 calibration DataLoader, and applies
SVD-LLM-v2 — a calibration-aware variant of LRD that uses activation covariance
statistics to find a better low-rank factorization than plain SVD.

**Run it:**

```bash
python examples/02_calibrated_lrd_svd_llm_v2.py
python examples/02_calibrated_lrd_svd_llm_v2.py --lrd-rank 128 --num-calibration-samples 4
```

**Source** (`examples/02_calibrated_lrd_svd_llm_v2.py`):

```python
--8<-- "examples/02_calibrated_lrd_svd_llm_v2.py"
```
