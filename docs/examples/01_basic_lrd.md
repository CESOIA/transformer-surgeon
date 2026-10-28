# Basic LRD

Loads Qwen2-0.5B-Instruct, applies reversible low-rank decomposition to two
specific projection layers using the HuggingFace manager, runs a generation,
then restores the model to its original weights.

**Run it:**

```bash
python examples/01_basic_lrd.py
python examples/01_basic_lrd.py --model-name Qwen/Qwen2-1.5B-Instruct --lrd-rank 320
```

**Source** (`examples/01_basic_lrd.py`):

```python
--8<-- "examples/01_basic_lrd.py"
```
