#!/usr/bin/env python3
"""
Compress a CIFAR-100-finetuned ViT (LRD + structured pruning + quantization,
including the Conv2d patch-embed layer) and run HF classification directly on
the compressed model — no export/backend involved.

Everything happens in plain HF/torch:
  1. Load Ahmed9275/Vit-Cifar100 (google/vit-base-patch16-224-in21k finetuned
     on CIFAR-100, 100 classes).
  2. Print the model structure (`print(model)`) so block/layer names are
     handy when writing `criteria=` strings below -- including
     `vit.embeddings.patch_embeddings.projection`, the Conv2d patch-embed
     layer, now compressible (Conv2dCompressed) via the same
     'preprocessing_conv' sentinel mechanism used for embedding layers.
  3. Build calibration and test DataLoaders from a slice of the CIFAR-100
     test split (used by calibration-dependent methods, e.g.
     `structured_pruning.method in {"magnitude", "gradient"}`; ignored by
     calibration-free methods like `structured_pruning.method="random"`).
  4. `configure_compression()` -- a single block where you dial in LRD /
     pruning / quantization settings per layer or block, including the
     Conv2d patch-embed. Edit this to experiment.
  5. `manager.apply(hard=...)` compresses in place; `model(pixel_values=...)`
     runs as normal HF inference. Top-1 accuracy is reported on the held-out
     test slice.

Usage:
    cd transformer-surgeon
    python scripts/compression/compress_infer_vit.py
    python scripts/compression/compress_infer_vit.py --hard
    python scripts/compression/compress_infer_vit.py --skip-compress   # float baseline
"""

import argparse
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

from transformersurgeon import ViTForImageClassificationCompress
from transformersurgeon.models.vit_c import ViTCompressionSchemesManager


MODEL_NAME = "Ahmed9275/Vit-Cifar100"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compress a CIFAR-100 ViT and run HF inference (no export)"
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--data-root", default=None,
                         help="HF `datasets` cache dir for CIFAR-100 (default: "
                              "the usual ~/.cache/huggingface/datasets)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-cal-samples", type=int, default=32,
        help="Number of CIFAR-100 test images used for calibration",
    )
    parser.add_argument(
        "--num-test-samples", type=int, default=200,
        help="Number of CIFAR-100 test images used to measure top-1 accuracy "
             "(disjoint from the calibration slice)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--hard", action="store_true",
        help="Hard-apply (irreversible, resizes weights). Default is soft "
             "(reversible, same-shape fake-compression) — safer for mixing "
             "LRD/pruning/quantization on the same layers.",
    )
    parser.add_argument(
        "--skip-compress", action="store_true",
        help="Run the unmodified float baseline for comparison",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Calibration / test data
# --------------------------------------------------------------------------

def build_cifar100_loaders(processor, data_root, num_cal, num_test, batch_size, seed):
    """A calibration DataLoader and a disjoint test DataLoader, both drawn
    from the CIFAR-100 test split (via the HF `datasets` hub copy, which
    downloads far faster here than the original torchvision mirror) and
    preprocessed with the model's own `ViTImageProcessor`
    (resize/rescale/normalize matched to the checkpoint). Fine-label indices
    in this dataset copy line up 1:1 with the checkpoint's `id2label`.
    """
    dataset = load_dataset("uoft-cs/cifar100", split="test", cache_dir=data_root)

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    cal_indices = indices[:num_cal]
    test_indices = indices[num_cal:num_cal + num_test]

    def collate(indices):
        examples = dataset[indices]
        pixel_values = processor(examples["img"], return_tensors="pt")["pixel_values"]
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(examples["fine_label"], dtype=torch.long),
        }

    def make_loader(subset_indices):
        batches = [subset_indices[i:i + batch_size] for i in range(0, len(subset_indices), batch_size)]
        return DataLoader(batches, batch_size=None, shuffle=False, collate_fn=collate)

    return make_loader(cal_indices), make_loader(test_indices)


# --------------------------------------------------------------------------
# Model inspection
# --------------------------------------------------------------------------

def print_model_structure(model) -> None:
    """Dump the module tree so block/layer names are easy to copy into
    `criteria=` below (e.g. "mlp.fc1", "attention.q_proj", block index 3, or
    the patch-embed conv via `preprocessing_conv`)."""
    print("=" * 70)
    print("MODEL STRUCTURE")
    print("=" * 70)
    print(model)
    print(f"\nnum_hidden_layers = {model.config.num_hidden_layers}")
    print(f"hidden_size       = {model.config.hidden_size}")
    print(f"intermediate_size = {model.config.intermediate_size}")
    print(f"num_attention_heads = {model.config.num_attention_heads}")
    print(f"patch_embed conv  = {model.vit.embeddings.patch_embeddings.projection}")
    print("=" * 70)


# --------------------------------------------------------------------------
# ============================================================
# COMPRESSION SETTINGS — edit this block to try different setups
# ============================================================
#
# manager.set(<compression_type>, <param>, <value>, criteria=<criteria>)
#
# compression_type: "lrd" | "structured_pruning" | "quantization"
# criteria:
#   None / "all"        -> every compressible layer
#   int                  -> all layers in that block index, e.g. criteria=3
#   str                  -> layers whose path contains the substring,
#                            e.g. "mlp", "attention", "fc2", "patch_embeddings"
#   [[str, int, ...]]    -> AND within the inner list, e.g. [["mlp", 0]]
#   [str, int, ...]      -> OR across items, e.g. ["fc1", 3]
#
# Layer names inside each block (see printed model structure above):
#   attention.q_proj / k_proj / v_proj / o_proj
#   mlp.fc1 / fc2
# Plus the patch-embed Conv2d, addressed via the 'preprocessing_conv'
# sentinel (criteria="patch_embeddings" matches its path,
# vit.embeddings.patch_embeddings.projection).
#
# Full parameter reference: transformer-surgeon/AGENTS.md
# --------------------------------------------------------------------------

def configure_compression(manager: ViTCompressionSchemesManager) -> None:
    # --- 1) Low-Rank Decomposition (LRD) ------------------------------------
    # rank="full" disables LRD. Not supported on the Conv2d patch-embed layer
    # (low-rank factorization only applies to a 2-D matmul weight) --
    # attempting it on "patch_embeddings" raises a clear ValueError.
    manager.set("lrd", "rank", "full", criteria=None)
    manager.set("lrd", "method", "svd", criteria=None)
    # Example: low-rank the MLP's fc1 of every block.
    # manager.set("lrd", "rank", 32, criteria="mlp.fc1")

    # --- 2) Structured pruning (removes output neurons) ---------------------
    # Coupled masks so q_proj/k_proj within a block prune the SAME neurons.
    # The patch-embed conv joins the residual-wide coupled group
    # automatically (see 'coupled_masks_all' in indexing_vit_c.py) -- pruning
    # its output channels cascades through every block's norm layers.
    groups = manager.auto_groups()
    for g in groups:
        print(f"Auto-group {g}: {groups[g]}")
    for g in groups:
        manager.set("structured_pruning", "share_mask", True, group=g)
        manager.set("structured_pruning", "reduce_op", "add", group=g)

    # "magnitude"/"gradient" need calibration; "random" doesn't.
    manager.set("structured_pruning", "method", "magnitude", group="group13")
    # manager.set("structured_pruning", "granularity", 128, group="group49")
    # manager.set("structured_pruning", "repeated_pattern", True, group="group49")
    manager.set("structured_pruning", "ratio", 0.04, group="group13")
    # Example: prune the patch-embed conv + residual hidden dim by 10%,
    # cascaded through every block via coupled_masks_all (group13 = every
    # attention.o_proj/mlp.fc2 + the patch-embed conv -- ALL of them must be
    # pruned together, use the group name, not a criteria= substring, or
    # manager.apply(hard=True) raises a coupling-consistency error).
    # NOTE: this specific checkpoint's `vit.embeddings` also concatenates a
    # cls_token and adds position_embeddings that are NOT part of the pruned
    # group (ViTEmbeddings is a composite module, not a plain nn.Embedding --
    # see 'preprocessing' vs. 'preprocessing_conv' in indexing_vit_c.py), so
    # hard-pruning this group still breaks the *full* model(pixel_values=...)
    # forward pass with a shape mismatch at that concat -- a known,
    # pre-existing limitation of this family, not of the conv support itself.
    # Soft pruning (hard=False, the default) and quantization are unaffected
    # since they don't resize tensors.
    # manager.set("structured_pruning", "method", "magnitude", group="group13")
    # manager.set("structured_pruning", "ratio", 0.1, group="group13")

    # --- 3) Quantization ------------------------------------------------------
    # precision="full" disables quantization. precision is an int bit-width
    # (8, 4, 2, ...), NOT a string like "int8".
    manager.set("quantization", "precision", "full", criteria=None)
    manager.set("quantization", "granularity", "per_channel", criteria=None)
    # Example: int8 weight-only (soft) quantization on the patch-embed conv.
    # manager.set("quantization", "precision", 8, criteria="patch_embeddings")
    # Example: int8 weight-only quantization on attention projections.
    # manager.set("quantization", "precision", 8, criteria="attention")


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def evaluate(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            logits = model(pixel_values=batch["pixel_values"]).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].numel()
    return correct, total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading {args.model_name} ...")
    model = ViTForImageClassificationCompress.from_pretrained(
        args.model_name, torch_dtype=torch.float32
    )
    model.eval()
    processor = ViTImageProcessor.from_pretrained(args.model_name)

    print_model_structure(model)

    print(f"\nBuilding CIFAR-100 calibration ({args.num_cal_samples} images) "
          f"and test ({args.num_test_samples} images) sets ...")
    cal_loader, test_loader = build_cifar100_loaders(
        processor, args.data_root, args.num_cal_samples, args.num_test_samples,
        args.batch_size, args.seed,
    )

    if not args.skip_compress:
        manager = ViTCompressionSchemesManager(model)
        manager.set_calibration_data(cal_loader)

        print("\nApplying compression settings from configure_compression() ...")
        configure_compression(manager)
        manager.apply(hard=args.hard, criteria=None, verbose=True)
        print(f"\nCompression applied ({'hard' if args.hard else 'soft'}).")
    else:
        print("\nSkipping compression — running float baseline.")

    print(f"\nEvaluating on {args.num_test_samples} held-out CIFAR-100 test images ...")
    correct, total = evaluate(model, test_loader)
    print(f"\nTop-1 accuracy: {correct}/{total} = {100.0 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
