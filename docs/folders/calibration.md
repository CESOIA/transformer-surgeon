# `calibration/` — collecting statistics

Calibration only gathers information; the compressors decide what to do with it.

| Module | Contents |
|---|---|
| `backbone.py` | `run_compression_calibration`: registers hooks, runs the data, fills each scheme's `calibration_data` |
| `raw_data/` | collectors (hooks) of activations, parameters and weight gradients; `registry.py` |
| `summaries/` | running statistics: covariance, shifted/cross covariance, activation ranges, weight gradient, raw tensors; `registry.py` |
| `input_absmax.py` | `collect_input_absmax`: per-channel max \|x\| (used by SmoothQuant) |
| `datasets.py` | `token_windows_loader`, `wikitext2_loader`: calibration data as token windows |

Flow: a collector's hook emits a raw tensor → the summaries that need that stream update their running state
→ the result lands in the scheme's `calibration_data` → the compressor reads it in `apply`.

```python
from transformersurgeon.calibration import wikitext2_loader

manager.set_calibration_data(wikitext2_loader(tokenizer, num_samples=32, seq_len=512))
manager.set("lrd", "method", "svd-llm-v2")   # needs the "covariance" summary
manager.set("lrd", "rank", 256, criteria="mlp")
manager.apply()
```

Two modes (`manager.set_calibration_mode`): `"standard"` (one pass over all layers) and `"cascade"` (block by
block, needed by AA-SVD; scheduled by the model dictionary's `calibration_groups`).

API: [calibration](../api/calibration.md).
