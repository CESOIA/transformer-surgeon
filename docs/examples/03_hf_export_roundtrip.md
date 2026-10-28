# HF Export Roundtrip

Demonstrates the full end-to-end workflow: apply LRD with VCON blending,
export the compressed model to a local HuggingFace-format directory, reload
it, and verify the roundtrip with a forward pass.

**Run it:**

```bash
python examples/03_hf_export_roundtrip.py
python examples/03_hf_export_roundtrip.py --model-type qwen2_vl_c --rank 64
python examples/03_hf_export_roundtrip.py --out-dir /tmp/my_export
```

**Source** (`examples/03_hf_export_roundtrip.py`):

```python
--8<-- "examples/03_hf_export_roundtrip.py"
```
