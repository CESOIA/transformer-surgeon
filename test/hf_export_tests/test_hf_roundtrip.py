"""
Unit test: compress a model, export to a local HF directory, and reload it.
Run with: python -m pytest test_hf_roundtrip.py -v
or:        python test_hf_roundtrip.py
"""
import sys
import shutil
import tempfile
import unittest

import torch
from transformers import AutoProcessor, AutoTokenizer

sys.path.append("../test_data")

### TEST CONFIGURATION ###
MODEL_TYPE = "qwen2_5_vl_c"
HARD_MODE = True
VERBOSE = False
GPU_NUM = 0
##########################


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


def _sizeof_dtype(dtype):
    sizes = {
        torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
        torch.int8: 1,
    }
    return sizes.get(dtype, 4)


def _param_count(model):
    return sum(p.numel() for p in model.parameters())


def _disk_size_gb(model):
    return sum(p.numel() * _sizeof_dtype(p.dtype) for p in model.parameters()) / 2**30


class TestHFRoundtrip(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_class, config_class, manager_class, model_name = _get_classes(MODEL_TYPE)
        cls.model_class = model_class
        cls.manager_class = manager_class
        cls.model_name = model_name

        device_str = f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu"
        cls.device = torch.device(device_str)

        print(f"\nLoading base model '{model_name}' on {device_str}...")
        cls.processor = AutoProcessor.from_pretrained(model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
        cls.model = model_class.from_pretrained(model_name, torch_dtype="auto").to(cls.device)

        cls.params_before = _param_count(cls.model)
        print(f"  Parameters before compression: {cls.params_before / 1e6:.2f} M "
              f"(~{_disk_size_gb(cls.model):.2f} GB)")

        # Apply compression
        print("Applying compression...")
        manager = manager_class(cls.model)
        manager.set(
            "lrd", "rank", 128,
            [
                ["language_model", "mlp.down_proj", 26],
                ["language_model", "mlp.down_proj", 27],
            ],
            verbose=VERBOSE,
        )
        manager.init_vcon(verbose=VERBOSE)
        manager.set_vcon_beta(beta=1.0)
        manager.apply(hard=HARD_MODE, verbose=VERBOSE)
        manager.update_config(verbose=VERBOSE)

        cls.params_after_compress = _param_count(cls.model)
        print(f"  Parameters after compression:  {cls.params_after_compress / 1e6:.2f} M "
              f"(~{_disk_size_gb(cls.model):.2f} GB)")

        # Export to a temp directory
        cls.tmp_dir = tempfile.mkdtemp(prefix="hf_roundtrip_")
        print(f"Exporting model to {cls.tmp_dir} ...")
        from transformersurgeon.hf import export_to_hf
        export_to_hf(
            cls.model,
            repo_id="test/roundtrip",
            base_model=model_name,
            out_dir=cls.tmp_dir,
            readme="Roundtrip unit-test export.",
            embed_code=True,
            token=None,
            private=True,
            exist_ok=True,
        )

        # Determine the actual saved path (export_to_hf may create a subdir)
        import os
        subdirs = [d for d in os.listdir(cls.tmp_dir)
                   if os.path.isdir(os.path.join(cls.tmp_dir, d))]
        cls.export_path = os.path.join(cls.tmp_dir, subdirs[0]) if subdirs else cls.tmp_dir
        print(f"  Saved to: {cls.export_path}")

        # Load back
        print("Loading model from export path...")
        cls.model_loaded = model_class.from_pretrained(
            cls.export_path, torch_dtype="auto"
        ).to(cls.device)
        cls.params_loaded = _param_count(cls.model_loaded)
        print(f"  Parameters after reload:       {cls.params_loaded / 1e6:.2f} M")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_compression_reduces_params(self):
        """Compression must reduce the total parameter count."""
        self.assertLess(
            self.params_after_compress, self.params_before,
            "Compressed model should have fewer parameters than the original.",
        )

    def test_roundtrip_param_count(self):
        """Reloaded model must have the same parameter count as the compressed model."""
        self.assertEqual(
            self.params_loaded, self.params_after_compress,
            f"Parameter count mismatch after roundtrip: "
            f"compressed={self.params_after_compress}, loaded={self.params_loaded}",
        )

    def test_roundtrip_weight_values(self):
        """Spot-check that a few weight tensors are numerically identical after roundtrip."""
        orig_sd = self.model.state_dict()
        loaded_sd = self.model_loaded.state_dict()

        self.assertEqual(set(orig_sd.keys()), set(loaded_sd.keys()),
                         "State-dict keys differ after roundtrip.")

        # Check first 5 keys
        for key in list(orig_sd.keys())[:5]:
            with self.subTest(key=key):
                self.assertTrue(
                    torch.equal(orig_sd[key].cpu(), loaded_sd[key].cpu()),
                    f"Weight mismatch for '{key}' after roundtrip.",
                )

    def test_loaded_model_config(self):
        """Reloaded model config must match the compressed model config."""
        self.assertEqual(
            type(self.model.config), type(self.model_loaded.config),
            "Config type mismatch after roundtrip.",
        )

    def test_loaded_model_forward(self):
        """Reloaded model must run a forward pass without errors."""
        dummy_input = self.tokenizer("Hello", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model_loaded(**dummy_input)
        self.assertIsNotNone(out.logits, "Forward pass produced no logits.")


if __name__ == "__main__":
    # Allow passing GPU number as first arg, e.g.: python test_hf_roundtrip.py 1
    if len(sys.argv) > 1:
        try:
            GPU_NUM = int(sys.argv.pop(1))
        except ValueError:
            pass
    unittest.main(verbosity=2)
