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
parser.add_argument("--calibration-size", type=int, default=64,
                    help="Number of calibration samples for activation quantization (default: 64)")
parser.add_argument("--calibration-batch-size", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--precision", type=int, default=8,
                    help="Integer bit-width for weight quantization (default: 8)")
parser.add_argument("--precision-activation", type=int, default=0,
                    help="Integer bit-width for activation quantization (0 = disabled, default: 0)")
parser.add_argument("--scheme-activation", type=str, default="asymmetric",
                    choices=["symmetric", "asymmetric"],
                    help="Activation quantization scheme (default: asymmetric)")
parser.add_argument("--granularity", type=str, default="per_tensor",
                    choices=["per_tensor", "per_channel"],
                    help="Weight quantization granularity (default: per_tensor)")
parser.add_argument("--hard", action="store_true",
                    help="Use hard (torchao) quantization instead of soft (fake-quant)")
parser.add_argument("--min-accuracy", type=float, default=50.0,
                    help="Minimum acceptable top-1 accuracy %% (default: 50.0)")

# unittest discovers tests via sys.argv, so parse only known args to avoid conflicts.
args, _remaining = parser.parse_known_args()

_activation_quant_enabled = args.precision_activation > 0


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
        print(
            f"Quantization mode: {cls.mode}, "
            f"precision={args.precision}, granularity={args.granularity}"
        )
        if _activation_quant_enabled:
            print(
                f"Activation quantization: precision={args.precision_activation}, "
                f"scheme={args.scheme_activation}"
            )

        processor = AutoImageProcessor.from_pretrained(args.model_name)
        cls.model = ViTForImageClassificationCompress.from_pretrained(
            args.model_name, torch_dtype="auto"
        ).to(cls.device)
        cls.model.eval()
        print(f"Loaded model: {args.model_name}")

        cls.manager = ViTCompressionSchemesManager(cls.model)
        cls.manager.set("quantization", "precision", args.precision, criteria="all", verbose=True)
        cls.manager.set("quantization", "granularity", args.granularity, criteria="all", verbose=True)

        if _activation_quant_enabled:
            cls.manager.set(
                "quantization", "precision_activation", args.precision_activation,
                criteria="all", verbose=True,
            )
            cls.manager.set(
                "quantization", "scheme_activation", args.scheme_activation,
                criteria="all", verbose=True,
            )

            # Build calibration subset from CIFAR-100 training split.
            train_dataset = load_dataset("cifar100", split="train")
            available_columns = set(train_dataset.column_names)
            image_column = "img" if "img" in available_columns else "image"

            calib_size = min(args.calibration_size, len(train_dataset))
            calib_subset = train_dataset.shuffle(seed=args.seed).select(range(calib_size))
            calib_examples = [
                {
                    "pixel_values": processor(
                        images=sample[image_column].convert("RGB"), return_tensors="pt"
                    )["pixel_values"].squeeze(0)
                }
                for sample in calib_subset
            ]
            calib_loader = DataLoader(
                calib_examples,
                batch_size=args.calibration_batch_size,
                shuffle=False,
            )
            print(f"Calibration loader: {len(calib_loader)} batches ({calib_size} samples)")
            cls.manager.set_calibration_data(calib_loader)

        cls.manager.apply(
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

    # ------------------------------------------------------------------
    # Weight quantization: sanity checks
    # ------------------------------------------------------------------

    def test_weight_quantization_applied(self):
        """All quantized layers must have finite, non-NaN weights."""
        for scheme in self.manager.iter_filtered(criteria="all"):
            module = scheme.get_module()
            if not hasattr(module, "weight"):
                continue
            w = module.weight
            self.assertTrue(
                torch.isfinite(w).all(),
                msg=f"Non-finite weights in {scheme.path} after quantization.",
            )

    # ------------------------------------------------------------------
    # Activation quantization: hook and attribute checks
    # ------------------------------------------------------------------

    @unittest.skipUnless(_activation_quant_enabled, "activation quantization not enabled")
    def test_activation_hooks_registered(self):
        """Every quantized layer must have exactly one activation fake-quant pre-hook."""
        missing = []
        for scheme in self.manager.iter_filtered(criteria="all"):
            module = scheme.get_module()
            if not hasattr(module, "_act_quant_hook_handle"):
                missing.append(scheme.path)
        self.assertEqual(
            missing, [],
            msg=f"Missing _act_quant_hook_handle on {len(missing)} module(s): {missing[:5]}",
        )

    @unittest.skipUnless(_activation_quant_enabled, "activation quantization not enabled")
    def test_activation_quant_attributes_stored(self):
        """scale, zero_point, precision, scheme must be stored on every quantized layer."""
        for scheme in self.manager.iter_filtered(criteria="all"):
            module = scheme.get_module()
            path = scheme.path
            for attr in ("_act_quant_scale", "_act_quant_zero_point",
                         "_act_quant_precision", "_act_quant_scheme"):
                self.assertTrue(
                    hasattr(module, attr),
                    msg=f"Missing attribute '{attr}' on module {path}.",
                )

    @unittest.skipUnless(_activation_quant_enabled, "activation quantization not enabled")
    def test_activation_quant_scale_positive(self):
        """Scale must be a positive finite scalar on every quantized layer."""
        for scheme in self.manager.iter_filtered(criteria="all"):
            module = scheme.get_module()
            path = scheme.path
            scale = module._act_quant_scale
            self.assertIsInstance(scale, torch.Tensor, msg=f"{path}: scale is not a Tensor.")
            self.assertTrue(
                (scale > 0).all(),
                msg=f"{path}: scale has non-positive values: {scale}.",
            )
            self.assertTrue(
                torch.isfinite(scale).all(),
                msg=f"{path}: scale has non-finite values: {scale}.",
            )

    @unittest.skipUnless(_activation_quant_enabled, "activation quantization not enabled")
    def test_activation_quant_config_matches_args(self):
        """Stored precision and scheme must match the CLI arguments on every layer."""
        for scheme in self.manager.iter_filtered(criteria="all"):
            module = scheme.get_module()
            path = scheme.path
            self.assertEqual(
                module._act_quant_precision, args.precision_activation,
                msg=f"{path}: _act_quant_precision mismatch.",
            )
            self.assertEqual(
                module._act_quant_scheme, args.scheme_activation,
                msg=f"{path}: _act_quant_scheme mismatch.",
            )

    @unittest.skipUnless(_activation_quant_enabled, "activation quantization not enabled")
    def test_activation_quant_zero_point_range(self):
        """zero_point must lie in the valid quantization range."""
        for scheme in self.manager.iter_filtered(criteria="all"):
            module = scheme.get_module()
            path = scheme.path
            zp = module._act_quant_zero_point
            scheme_name = module._act_quant_scheme
            precision = module._act_quant_precision
            if scheme_name == "symmetric":
                expected_zp = 0
                self.assertTrue(
                    (zp == expected_zp).all(),
                    msg=f"{path}: symmetric zero_point must be 0, got {zp}.",
                )
            else:
                qmax = 2 ** precision - 1
                self.assertTrue(
                    ((zp >= 0) & (zp <= qmax)).all(),
                    msg=f"{path}: asymmetric zero_point {zp} out of [0, {qmax}].",
                )

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------

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
        act_info = (
            f", act_precision={args.precision_activation}, scheme={args.scheme_activation}"
            if _activation_quant_enabled else ""
        )
        print(
            f"\nTop-1 accuracy on CIFAR-100 subset "
            f"({self.mode}, precision={args.precision}, granularity={args.granularity}"
            f"{act_info}): {accuracy:.2f}%  (threshold: {args.min_accuracy:.1f}%)"
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
