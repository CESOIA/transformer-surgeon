"""
Compress a model, export it to a local HuggingFace-format directory, and reload it.

Demonstrates the full end-to-end workflow:
  1. Load a VLM (Qwen2.5-VL or Qwen2-VL)
  2. Apply LRD with VCON blending to two MLP layers
  3. Export the compressed model to disk in HF format
  4. Reload the model from disk
  5. Run a forward pass to verify the roundtrip

Usage:
    python examples/03_hf_export_roundtrip.py
    python examples/03_hf_export_roundtrip.py --model-type qwen2_vl_c --rank 64
    python examples/03_hf_export_roundtrip.py --out-dir /tmp/my_export

Requirements:
    pip install transformer-surgeon transformers torch
"""

import argparse
import os
import shutil
import tempfile

import torch
from transformers import AutoTokenizer

from transformersurgeon.hf import export_to_hf


def parse_args():
    parser = argparse.ArgumentParser(description="HF export roundtrip example")
    parser.add_argument(
        "--model-type",
        type=str,
        default="qwen2_5_vl_c",
        choices=["qwen2_5_vl_c", "qwen2_vl_c"],
        help="Which VLM family to use",
    )
    parser.add_argument("--rank", type=int, default=128,
                        help="LRD rank for the two compressed MLP layers")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to export into. Auto-created temp dir if not set.")
    return parser.parse_args()


def _get_classes(model_type):
    if model_type == "qwen2_vl_c":
        from transformersurgeon import (
            Qwen2VLForConditionalGenerationCompress,
            Qwen2VLConfigCompress,
            Qwen2VLCompressionSchemesManager,
        )
        return (
            Qwen2VLForConditionalGenerationCompress,
            Qwen2VLConfigCompress,
            Qwen2VLCompressionSchemesManager,
            "Qwen/Qwen2-VL-2B-Instruct",
        )
    elif model_type == "qwen2_5_vl_c":
        from transformersurgeon import (
            Qwen2_5_VLForConditionalGenerationCompress,
            Qwen2_5_VLConfigCompress,
            Qwen2_5_VLCompressionSchemesManager,
        )
        return (
            Qwen2_5_VLForConditionalGenerationCompress,
            Qwen2_5_VLConfigCompress,
            Qwen2_5_VLCompressionSchemesManager,
            "Qwen/Qwen2.5-VL-3B-Instruct",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class, config_class, manager_class, model_name = _get_classes(args.model_type)
    print(f"Device:     {device}")
    print(f"Model:      {model_name}")
    print(f"LRD rank:   {args.rank}")

    # --- Load model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, torch_dtype="auto").to(device).eval()

    params_before = param_count(model)
    print(f"\nParameters before compression: {params_before / 1e6:.2f} M")

    # --- Apply LRD with VCON ---
    # VCON wraps each target layer in an (original, compressed) pair before
    # applying compression. This lets you fine-tune with a smooth beta blend
    # before collapsing to the compressed module.
    manager = manager_class(model)
    manager.set(
        "lrd", "rank", args.rank,
        criteria=[
            ["language_model", "mlp.down_proj", 26],
            ["language_model", "mlp.down_proj", 27],
        ],
    )
    manager.init_vcon()            # wrap target layers in VCONBlock
    manager.set_vcon_beta(beta=1.0)  # start at original output (beta=1 → no compression visible)
    manager.apply(hard=True)       # hard=True: irreversible, suitable for export
    manager.update_config()        # write compression_config back into model.config

    params_after = param_count(model)
    print(f"Parameters after  compression: {params_after / 1e6:.2f} M "
          f"({100 * (1 - params_after / params_before):.1f}% reduction)")

    # --- Export to disk ---
    auto_tmpdir = args.out_dir is None
    out_dir = args.out_dir if args.out_dir else tempfile.mkdtemp(prefix="ts_roundtrip_")
    print(f"\nExporting to {out_dir} ...")

    export_to_hf(
        model,
        repo_id="local/roundtrip-demo",
        base_model=model_name,
        out_dir=out_dir,
        readme="TransformerSurgeon roundtrip demo export.",
        embed_code=True,
        token=None,
        private=True,
        exist_ok=True,
    )

    # export_to_hf may write into a subdirectory named after the repo_id slug.
    subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    export_path = os.path.join(out_dir, subdirs[0]) if subdirs else out_dir
    print(f"Saved to: {export_path}")

    # --- Reload and verify ---
    print("Reloading from disk...")
    model_reloaded = model_class.from_pretrained(export_path, torch_dtype="auto").to(device).eval()
    params_reloaded = param_count(model_reloaded)
    print(f"Parameters after reload: {params_reloaded / 1e6:.2f} M "
          f"({'OK' if params_reloaded == params_after else 'MISMATCH'})")

    dummy_input = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_reloaded(**dummy_input)

    logits_ok = out.logits is not None and torch.isfinite(out.logits).all().item()
    print(f"Forward pass logits finite: {logits_ok}")

    if not logits_ok or params_reloaded != params_after:
        raise RuntimeError("Roundtrip verification failed.")

    print("\nRoundtrip successful.")

    # --- Cleanup temp dir if we created it ---
    if auto_tmpdir:
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"Cleaned up temp dir: {out_dir}")


if __name__ == "__main__":
    main()
