import argparse
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from transformersurgeon import (
    ViTCompressionSchemesManager,
    ViTForImageClassificationCompress,
)

# ---------------------------
# CLI configuration
# ---------------------------
parser = argparse.ArgumentParser(description="ViT + CIFAR-100 calibration/compression test")
# parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
parser.add_argument("--model-name", type=str, default="Ahmed9275/Vit-Cifar100")
parser.add_argument("--calibration-portion", type=float, default=0.1)
parser.add_argument("--calibration-size", type=int, default=0)
parser.add_argument("--calibration-batch-size", type=int, default=8)
parser.add_argument("--eval-size", type=int, default=256)
parser.add_argument("--eval-batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.calibration_portion <= 0 or args.calibration_portion > 1:
    raise ValueError("--calibration-portion must be in (0, 1].")
if args.calibration_size < 0:
    raise ValueError("--calibration-size must be >= 0.")
if args.calibration_batch_size <= 0 or args.eval_batch_size <= 0:
    raise ValueError("Batch sizes must be positive integers.")
if args.eval_size <= 0:
    raise ValueError("--eval-size must be a positive integer.")

# ---------------------------
# Compression configuration
# ---------------------------
# Method is the same SVD-LLM-v2 approach used in the Qwen script.
DEFAULT_RANK = 196
MLP_RANK_OVERRIDES = {
    "intermediate.dense": 196,
    "output.dense": 196,
}

# ---------------------------
# Device and reproducibility
# ---------------------------
random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Load model and processor
# ---------------------------
processor = AutoImageProcessor.from_pretrained(args.model_name)
model = ViTForImageClassificationCompress.from_pretrained(args.model_name, torch_dtype="auto").to(device)
model.eval()
print(f"Loaded model: {args.model_name}")
print(model)

# ---------------------------
# Load CIFAR-100 dataset
# ---------------------------
print("Loading CIFAR-100 from Hugging Face...")
train_dataset = load_dataset("cifar100", split="train")
test_dataset = load_dataset("cifar100", split="test")

available_columns = set(train_dataset.column_names)
print(f"CIFAR-100 columns: {sorted(available_columns)}")

if "img" in available_columns:
    image_column = "img"
elif "image" in available_columns:
    image_column = "image"
else:
    raise ValueError(f"No supported image column found. Available columns: {sorted(available_columns)}")

if "fine_label" in available_columns:
    label_column = "fine_label"
elif "label" in available_columns:
    label_column = "label"
else:
    raise ValueError(f"No supported label column found. Available columns: {sorted(available_columns)}")

# ---------------------------
# Build calibration subset
# ---------------------------
# Calibration uses a variable portion of CIFAR-100 training data.
portion_size = max(1, int(len(train_dataset) * args.calibration_portion))
if args.calibration_size > 0:
    calibration_size = min(portion_size, args.calibration_size)
else:
    calibration_size = portion_size

train_shuffled = train_dataset.shuffle(seed=args.seed)
calibration_subset = train_shuffled.select(range(calibration_size))
print(
    f"Calibration subset size: {calibration_size} "
    f"(portion={args.calibration_portion}, explicit_size={args.calibration_size})"
)

calibration_examples = []
for sample in calibration_subset:
    image = sample[image_column].convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
    calibration_examples.append({"pixel_values": pixel_values})

calibration_loader = DataLoader(
    calibration_examples,
    batch_size=args.calibration_batch_size,
    shuffle=False,
)
print(f"Calibration DataLoader batches: {len(calibration_loader)}")

# ---------------------------
# Configure and apply SVD-LLM-v2
# ---------------------------
manager = ViTCompressionSchemesManager(model)
manager.set("lrd", "method", "svd-llm-v2", criteria="all", verbose=True)

# Apply a default rank to all compressible linear layers.
configured_default = 0
for scheme in manager.iter_filtered(criteria="all"):
    module = scheme.get_module()
    if hasattr(module, "weight"):
        manager.set("lrd", "rank", DEFAULT_RANK, criteria=scheme.path, verbose=False)
        configured_default += 1

# Override MLP ranks with a less aggressive value.
for layer_key, rank_value in MLP_RANK_OVERRIDES.items():
    manager.set("lrd", "rank", rank_value, criteria=layer_key, verbose=False)

print(f"Default rank applied to {configured_default} schemes.")
print(f"MLP rank overrides: {MLP_RANK_OVERRIDES}")

manager.set_calibration_data(calibration_loader)
manager.apply(
    hard=False,
    criteria="all",
    verbose=True,
    device=device,
    offload_to_cpu=False,
)
print("Compression completed successfully.")
print("Model summary:")
print(model)

# ---------------------------
# Post-compression evaluation
# ---------------------------
# Evaluate on a subset of CIFAR-100 test set.
eval_size = min(args.eval_size, len(test_dataset))
test_subset = test_dataset.select(range(eval_size))

eval_examples = []
for sample in test_subset:
    image = sample[image_column].convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
    eval_examples.append({
        "pixel_values": pixel_values,
        "labels": int(sample[label_column]),
    })

eval_loader = DataLoader(
    eval_examples,
    batch_size=args.eval_batch_size,
    shuffle=False,
)

correct = 0
total = 0

with torch.no_grad():
    for batch in eval_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        logits = model(pixel_values=pixel_values).logits
        predictions = torch.argmax(logits, dim=-1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = 100.0 * correct / max(total, 1)
print(f"Evaluation subset size: {total}")
print(f"Top-1 accuracy on CIFAR-100 subset: {accuracy:.2f}%")
