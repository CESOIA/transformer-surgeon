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
import argparse, os, shutil, tempfile
import torch
from transformers import AutoTokenizer
from transformersurgeon.hf import export_to_hf


def _get_classes(model_type):
    if model_type == "qwen2_vl_c":
        from transformersurgeon import (
            Qwen2VLForConditionalGenerationCompress,
            Qwen2VLConfigCompress,
            Qwen2VLCompressionSchemesManager,
        )
        return Qwen2VLForConditionalGenerationCompress, Qwen2VLConfigCompress, \
               Qwen2VLCompressionSchemesManager, "Qwen/Qwen2-VL-2B-Instruct"
    else:
        from transformersurgeon import (
            Qwen2_5_VLForConditionalGenerationCompress,
            Qwen2_5_VLConfigCompress,
            Qwen2_5_VLCompressionSchemesManager,
        )
        return Qwen2_5_VLForConditionalGenerationCompress, Qwen2_5_VLConfigCompress, \
               Qwen2_5_VLCompressionSchemesManager, "Qwen/Qwen2.5-VL-3B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="qwen2_5_vl_c",
                        choices=["qwen2_5_vl_c", "qwen2_vl_c"])
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class, _, manager_class, model_name = _get_classes(args.model_type)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, torch_dtype="auto").to(device).eval()
    params_before = sum(p.numel() for p in model.parameters())
    print(f"Parameters before: {params_before / 1e6:.2f} M")

    # Compress two MLP layers with VCON wrapping
    manager = manager_class(model)
    manager.set("lrd", "rank", args.rank, criteria=[
        ["language_model", "mlp.down_proj", 26],
        ["language_model", "mlp.down_proj", 27],
    ])
    manager.init_vcon()
    manager.set_vcon_beta(beta=1.0)
    manager.apply(hard=True)
    manager.update_config()

    params_after = sum(p.numel() for p in model.parameters())
    print(f"Parameters after:  {params_after / 1e6:.2f} M")

    # Export
    auto_tmp = args.out_dir is None
    out_dir = args.out_dir or tempfile.mkdtemp(prefix="ts_roundtrip_")
    export_to_hf(model, repo_id="local/roundtrip-demo", base_model=model_name,
                 out_dir=out_dir, embed_code=True, token=None, private=True, exist_ok=True)

    subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    export_path = os.path.join(out_dir, subdirs[0]) if subdirs else out_dir

    # Reload and verify
    model_reloaded = model_class.from_pretrained(export_path, torch_dtype="auto").to(device).eval()
    dummy = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_reloaded(**dummy)
    print(f"Roundtrip param count match: {sum(p.numel() for p in model_reloaded.parameters()) == params_after}")
    print(f"Logits finite: {torch.isfinite(out.logits).all().item()}")
    print("Roundtrip successful.")

    if auto_tmp:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
```
