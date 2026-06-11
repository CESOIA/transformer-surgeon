import argparse
import random
import sys
import unittest

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
parser = argparse.ArgumentParser(description="ViT + CIFAR-100 quantization unit test")
parser.add_argument("--model-name", type=str, default="Ahmed9275/Vit-Cifar100")
parser.add_argument("--eval-size", type=int, default=256)
parser.add_argument("--eval-batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--precision", type=int, default=8,
                    help="Integer bit-width for quantization (default: 8)")
parser.add_argument("--granularity", type=str, default="per_tensor",
                    choices=["per_tensor", "per_channel"],
                    help="Quantization granularity (default: per_tensor)")
parser.add_argument("--hard", action="store_true",
                    help="Use hard (torchao) quantization instead of soft (fake-quant)")
parser.add_argument("--min-accuracy", type=float, default=50.0,
                    help="Minimum acceptable top-1 accuracy %% (default: 50.0)")

# unittest discovers tests via sys.argv, so parse only known args to avoid conflicts.
args, _remaining = parser.parse_known_args()


class ViTCifar100QuantizationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if args.eval_batch_size <= 0:
            raise ValueError("--eval-batch-size must be a positive integer.")
        if args.eval_size <= 0:
            raise ValueError("--eval-size must be a positive integer.")

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cls.mode = "hard (torchao)" if args.hard else "soft (fake-quant)"

        print(f"\nUsing device: {cls.device}")
        print(f"Quantization mode: {cls.mode}, precision={args.precision}, granularity={args.granularity}")

        processor = AutoImageProcessor.from_pretrained(args.model_name)
        cls.model = ViTForImageClassificationCompress.from_pretrained(
            args.model_name, torch_dtype="auto"
        ).to(cls.device)
        cls.model.eval()
        print(f"Loaded model: {args.model_name}")

        manager = ViTCompressionSchemesManager(cls.model)
        manager.set("quantization", "precision", args.precision, criteria="all", verbose=True)
        manager.set("quantization", "granularity", args.granularity, criteria="all", verbose=True)
        manager.apply(
            hard=args.hard,
            criteria="all",
            verbose=True,
            device=cls.device,
            offload_to_cpu=False,
        )
        print(f"Quantization ({cls.mode}) completed.")

        test_dataset = load_dataset("cifar100", split="test")
        available_columns = set(test_dataset.column_names)
        image_column = "img" if "img" in available_columns else "image"
        label_column = "fine_label" if "fine_label" in available_columns else "label"

        eval_size = min(args.eval_size, len(test_dataset))
        test_subset = test_dataset.select(range(eval_size))

        eval_examples = [
            {
                "pixel_values": processor(
                    images=sample[image_column].convert("RGB"), return_tensors="pt"
                )["pixel_values"].squeeze(0),
                "labels": int(sample[label_column]),
            }
            for sample in test_subset
        ]

        cls.eval_loader = DataLoader(
            eval_examples,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )

    def test_accuracy_above_threshold(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.eval_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = self.model(pixel_values=pixel_values).logits
                predictions = torch.argmax(logits, dim=-1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = 100.0 * correct / max(total, 1)
        print(
            f"\nTop-1 accuracy on CIFAR-100 subset "
            f"({self.mode}, precision={args.precision}, granularity={args.granularity}): "
            f"{accuracy:.2f}%  (threshold: {args.min_accuracy:.1f}%)"
        )
        self.assertGreaterEqual(
            accuracy,
            args.min_accuracy,
            msg=(
                f"Accuracy {accuracy:.2f}% is below the minimum threshold "
                f"of {args.min_accuracy:.1f}%."
            ),
        )


if __name__ == "__main__":
    # Strip custom args before handing off to unittest so it only sees its own flags.
    sys.argv = [sys.argv[0]] + _remaining
    unittest.main(verbosity=2)
