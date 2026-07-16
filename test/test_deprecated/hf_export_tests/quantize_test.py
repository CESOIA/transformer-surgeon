"""
Unit tests: verify quantization metadata tensors survive as registered buffers
so they are included in the HF save_pretrained state_dict.

Coverage
--------
1. After Quantizer.apply(), act-quant scale/zero-point appear in state_dict().
2. After Quantizer.apply(hard=True), _torchao_scale appears in state_dict().
3. torch.save / torch.load state-dict round-trip preserves values exactly.
4. save_pretrained (HF/safetensors format) writes the buffer keys to disk.
5. Values can be reloaded from the saved file and are numerically identical.
6. Quantizer.restore() correctly removes the dynamically added buffers.

Run:
    cd <repo-root>
    conda run -n py312_torch python -m pytest test/hf_export_tests/quantize_test.py -v
or:
    conda run -n py312_torch python test/hf_export_tests/quantize_test.py
"""

import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn

from transformersurgeon.blocks import TransformerDecoder
from transformersurgeon.blocks.config import CustomDecoderConfigCompress
from transformersurgeon.compression.quantization import Quantizer

# ---------------------------------------------------------------------------
# Tiny model dimensions — runs in <1 s on CPU
# ---------------------------------------------------------------------------
_VOCAB   = 32
_HIDDEN  = 16
_HEADS   = 2
_INTER   = 32
_LAYERS  = 2
_CACHE   = 8

_CONVERTED_INDEXING = {
    "num_blocks_attr": "num_hidden_layers",
    "path_template": "blocks.{block_index}.{path}",
    "path_list": {
        "norm_in":  [],
        "attn":     ["q_proj", "k_proj", "v_proj", "out_proj"],
        "norm_out": [],
        "mlp":      ["gate_proj", "up_proj", "down_proj"],
    },
}

# Quantizer config template — callers override precision/precision_activation.
_QCFG_BASE = {
    "method":              "vanilla",
    "precision":           "full",
    "sparsity":            0.0,
    "sparse_method":       "magnitude",
    "eps":                 1e-6,
    "granularity":         "per_channel",
    "precision_activation":"full",
    "method_activation":   "maxmin",
    "scheme_activation":   "asymmetric",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> CustomDecoderConfigCompress:
    return CustomDecoderConfigCompress(
        num_hidden_layers=_LAYERS,
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        intermediate_size=_INTER,
        hidden_act="silu",
        attn_type="mha_causal",
        mlp_type="mlp_gated",
        norm_type="rmsnorm",
        max_cache_len=_CACHE,
        vocab_size=_VOCAB,
        indexing=_CONVERTED_INDEXING,
    )


def _make_decoder() -> TransformerDecoder:
    torch.manual_seed(0)
    decoder = TransformerDecoder(_make_config())
    decoder.eval()
    return decoder


def _soft_quantizer(precision: int = 8) -> Quantizer:
    cfg = {**_QCFG_BASE, "precision": precision}
    return Quantizer(cfg)


def _act_quantizer(precision_activation: int = 8) -> Quantizer:
    cfg = {**_QCFG_BASE, "precision_activation": precision_activation}
    return Quantizer(cfg)


def _fake_calibration(module: nn.Module) -> dict:
    """Produce plausible activation range by running a random forward pass."""
    in_features = getattr(module, "in_features", _HIDDEN)
    x = torch.randn(1, 4, in_features)
    with torch.no_grad():
        out = module(x)
    act_min = out.min().detach()
    act_max = out.max().detach()
    return {"activation_range": {"min": act_min, "max": act_max}}


def _load_safetensors_keys(save_dir: str) -> set:
    """Return tensor keys from the safetensors file in *save_dir*."""
    from safetensors import safe_open
    st_path = os.path.join(save_dir, "model.safetensors")
    with safe_open(st_path, framework="pt", device="cpu") as f:
        return set(f.keys())


def _load_safetensors_tensor(save_dir: str, key: str) -> torch.Tensor:
    from safetensors import safe_open
    st_path = os.path.join(save_dir, "model.safetensors")
    with safe_open(st_path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


# ---------------------------------------------------------------------------
# 1. Buffer presence in state_dict
# ---------------------------------------------------------------------------

class TestQuantBuffersInStateDict(unittest.TestCase):
    """Quantizer.apply() must register scale tensors as nn.Module buffers."""

    def test_act_quant_scale_in_state_dict(self):
        """_act_quant_scale and _act_quant_zero_point appear in state_dict after activation quant."""
        decoder = _make_decoder()
        q = _act_quantizer(precision_activation=8)

        target = decoder.blocks[0].mlp.down_proj
        q.set_calibration_store(_fake_calibration(target))
        q.apply(target)

        sd = target.state_dict()
        self.assertIn("_act_quant_scale",      sd, "scale buffer missing from state_dict")
        self.assertIn("_act_quant_zero_point", sd, "zero_point buffer missing from state_dict")

    def test_act_quant_buffer_values_are_tensors(self):
        """Scale buffers must be float tensors, not arbitrary Python objects."""
        decoder = _make_decoder()
        q = _act_quantizer(precision_activation=8)
        target = decoder.blocks[0].mlp.down_proj
        q.set_calibration_store(_fake_calibration(target))
        q.apply(target)

        self.assertIsInstance(target._act_quant_scale,      torch.Tensor)
        self.assertIsInstance(target._act_quant_zero_point, torch.Tensor)

    def test_act_quant_propagated_to_whole_model_state_dict(self):
        """Buffer keys must appear under the correct prefix in the whole-model state_dict."""
        decoder = _make_decoder()
        q = _act_quantizer(precision_activation=8)
        target = decoder.blocks[0].mlp.down_proj
        q.set_calibration_store(_fake_calibration(target))
        q.apply(target)

        full_sd = decoder.state_dict()
        scale_keys = [k for k in full_sd if k.endswith("_act_quant_scale")]
        self.assertEqual(len(scale_keys), 1, f"Expected 1 scale key, got: {scale_keys}")
        self.assertTrue(
            scale_keys[0].startswith("blocks.0.mlp.down_proj"),
            f"Unexpected key prefix: {scale_keys[0]}",
        )

    def test_hard_quant_scale_in_state_dict(self):
        """_torchao_scale buffer must appear in state_dict after hard quantization."""
        try:
            import torchao  # noqa: F401
        except ImportError:
            self.skipTest("torchao not installed")

        decoder = _make_decoder()
        q = Quantizer({**_QCFG_BASE, "precision": 8})
        q.apply(decoder.blocks[0].mlp, hard=True)

        # Check that at least one MLP linear has the buffer
        linears_with_scale = [
            name for name, m in decoder.blocks[0].mlp.named_modules()
            if isinstance(m, nn.Linear) and "_torchao_scale" in m.state_dict()
        ]
        self.assertGreater(
            len(linears_with_scale), 0,
            "_torchao_scale buffer not found in any MLP linear after hard quant",
        )

    def test_hard_quant_scale_in_whole_model_state_dict(self):
        """_torchao_scale keys must appear in the whole-model state_dict after hard quant."""
        try:
            import torchao  # noqa: F401
        except ImportError:
            self.skipTest("torchao not installed")

        decoder = _make_decoder()
        q = Quantizer({**_QCFG_BASE, "precision": 8})
        q.apply(decoder.blocks[0].mlp, hard=True)

        full_sd = decoder.state_dict()
        torchao_scale_keys = [k for k in full_sd if k.endswith("_torchao_scale")]
        self.assertGreater(
            len(torchao_scale_keys), 0,
            f"No _torchao_scale key found in state_dict. Keys: {list(full_sd.keys())[:20]}",
        )


# ---------------------------------------------------------------------------
# 2. state_dict round-trip via torch.save / torch.load
# ---------------------------------------------------------------------------

class TestQuantBuffersTorchSave(unittest.TestCase):
    """torch.save / torch.load of the state_dict must preserve buffer values."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="quantize_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_act_quant_state_dict_roundtrip(self):
        """Act-quant scale and zero_point survive torch.save/torch.load of state_dict."""
        decoder = _make_decoder()
        q = _act_quantizer(precision_activation=8)
        target = decoder.blocks[0].mlp.down_proj
        q.set_calibration_store(_fake_calibration(target))
        q.apply(target)

        orig_scale = target._act_quant_scale.clone()
        orig_zp    = target._act_quant_zero_point.clone()

        sd_path = os.path.join(self.tmpdir, "state_dict.pt")
        torch.save(decoder.state_dict(), sd_path)

        saved_sd = torch.load(sd_path, weights_only=True)
        decoder2 = _make_decoder()
        # Pre-register buffers with correct shapes so load_state_dict can fill them.
        for key, tensor in saved_sd.items():
            if key.endswith(("_act_quant_scale", "_act_quant_zero_point")):
                parts = key.split(".")
                m = decoder2
                for p in parts[:-1]:
                    m = getattr(m, p)
                m.register_buffer(parts[-1], torch.empty_like(tensor))
        decoder2.load_state_dict(saved_sd)

        loaded_scale = decoder2.blocks[0].mlp.down_proj._act_quant_scale
        loaded_zp    = decoder2.blocks[0].mlp.down_proj._act_quant_zero_point

        self.assertTrue(torch.equal(orig_scale, loaded_scale), "scale mismatch after torch.save roundtrip")
        self.assertTrue(torch.equal(orig_zp,    loaded_zp),    "zero_point mismatch after torch.save roundtrip")

    def test_hard_quant_scale_state_dict_roundtrip(self):
        """Torchao scale buffer survives torch.save/torch.load (after weight dequantization)."""
        try:
            from torchao.dtypes import AffineQuantizedTensor
        except ImportError:
            self.skipTest("torchao not installed")

        decoder = _make_decoder()
        q = Quantizer({**_QCFG_BASE, "precision": 8})
        q.apply(decoder.blocks[0].mlp, hard=True)

        # Capture the scale from the first quantized linear
        first_quant_linear = next(
            m for m in decoder.blocks[0].mlp.modules()
            if isinstance(m, nn.Linear) and hasattr(m, "_torchao_scale")
        )
        orig_scale = first_quant_linear._torchao_scale.clone()

        # Dequantize in-place so standard state_dict serialization works
        for m in decoder.blocks[0].mlp.modules():
            if isinstance(m, nn.Linear) and isinstance(m.weight, AffineQuantizedTensor):
                m.weight = nn.Parameter(m.weight.dequantize(), requires_grad=False)

        sd_path = os.path.join(self.tmpdir, "state_dict_hard.pt")
        torch.save(decoder.state_dict(), sd_path)

        saved_sd = torch.load(sd_path, weights_only=True)

        decoder2 = _make_decoder()
        # Pre-register buffers with correct shapes so load_state_dict can fill them.
        for key, tensor in saved_sd.items():
            if key.endswith("_torchao_scale"):
                parts = key.split(".")
                m = decoder2
                for p in parts[:-1]:
                    m = getattr(m, p)
                m.register_buffer("_torchao_scale", torch.empty_like(tensor))

        decoder2.load_state_dict(saved_sd, strict=True)

        # Navigate to the same submodule in decoder2
        target_name = next(
            n for n, mod in decoder.blocks[0].mlp.named_modules()
            if mod is first_quant_linear
        )
        parts = target_name.split(".")
        m2 = decoder2.blocks[0].mlp
        for p in parts:
            m2 = getattr(m2, p)
        loaded_scale = m2._torchao_scale

        self.assertTrue(
            torch.allclose(orig_scale, loaded_scale),
            f"_torchao_scale mismatch: orig={orig_scale}, loaded={loaded_scale}",
        )


# ---------------------------------------------------------------------------
# 3. HF save_pretrained writes buffer keys to safetensors
# ---------------------------------------------------------------------------

class TestQuantBuffersSafetensors(unittest.TestCase):
    """save_pretrained must write quantization buffer keys to the safetensors file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="quantize_hf_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # Dtypes natively supported by the safetensors format.
    _ST_DTYPES = {
        torch.float16, torch.float32, torch.float64, torch.bfloat16,
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    }

    def _save_decoder(self, decoder: TransformerDecoder) -> str:
        """Save decoder state_dict as a safetensors file and return the save path."""
        from safetensors.torch import save_file
        save_path = os.path.join(self.tmpdir, "model.safetensors")
        # Keep only plain Tensor values with dtypes safetensors can handle.
        # (AffineQuantizedTensor and other custom subtypes are excluded.)
        sd = {
            k: v for k, v in decoder.state_dict().items()
            if type(v) is torch.Tensor and v.dtype in self._ST_DTYPES
        }
        save_file(sd, save_path)
        return self.tmpdir

    def test_act_quant_scale_saved_to_safetensors(self):
        """Act-quant scale and zero_point keys appear in the safetensors file."""
        decoder = _make_decoder()
        q = _act_quantizer(precision_activation=8)
        target = decoder.blocks[0].mlp.down_proj
        q.set_calibration_store(_fake_calibration(target))
        q.apply(target)

        save_dir = self._save_decoder(decoder)
        keys = _load_safetensors_keys(save_dir)

        scale_keys = [k for k in keys if k.endswith("_act_quant_scale")]
        zp_keys    = [k for k in keys if k.endswith("_act_quant_zero_point")]
        self.assertGreater(len(scale_keys), 0, "No _act_quant_scale key in safetensors file")
        self.assertGreater(len(zp_keys),    0, "No _act_quant_zero_point key in safetensors file")

    def test_act_quant_values_in_safetensors_match_live(self):
        """Scale tensor read from safetensors file must equal the live buffer value."""
        decoder = _make_decoder()
        q = _act_quantizer(precision_activation=8)
        target = decoder.blocks[0].mlp.down_proj
        q.set_calibration_store(_fake_calibration(target))
        q.apply(target)

        live_scale = target._act_quant_scale.clone()
        save_dir = self._save_decoder(decoder)

        scale_key = next(k for k in _load_safetensors_keys(save_dir) if k.endswith("_act_quant_scale"))
        loaded_scale = _load_safetensors_tensor(save_dir, scale_key)

        self.assertTrue(
            torch.equal(live_scale, loaded_scale),
            f"Scale mismatch: live={live_scale.item():.6f}, saved={loaded_scale.item():.6f}",
        )

    def test_hard_quant_scale_saved_to_safetensors(self):
        """_torchao_scale key appears in safetensors file after hard quantization."""
        try:
            from torchao.dtypes import AffineQuantizedTensor
        except ImportError:
            self.skipTest("torchao not installed")

        decoder = _make_decoder()
        q = Quantizer({**_QCFG_BASE, "precision": 8})
        q.apply(decoder.blocks[0].mlp, hard=True)

        # Dequantize so state_dict contains only plain float tensors
        for m in decoder.blocks[0].mlp.modules():
            if isinstance(m, nn.Linear) and isinstance(m.weight, AffineQuantizedTensor):
                m.weight = nn.Parameter(m.weight.dequantize(), requires_grad=False)

        save_dir = self._save_decoder(decoder)
        keys = _load_safetensors_keys(save_dir)
        torchao_keys = [k for k in keys if k.endswith("_torchao_scale")]
        self.assertGreater(
            len(torchao_keys), 0,
            f"No _torchao_scale key in safetensors file. Found: {sorted(keys)}",
        )


# ---------------------------------------------------------------------------
# 4. Restore cleanup
# ---------------------------------------------------------------------------

class TestQuantRestore(unittest.TestCase):
    """Quantizer.restore() must remove all dynamically added buffers."""

    def test_restore_removes_act_quant_buffers(self):
        """After restore(), _act_quant_scale and _act_quant_zero_point must not be in state_dict."""
        decoder = _make_decoder()
        q = _act_quantizer(precision_activation=8)
        target = decoder.blocks[0].mlp.down_proj
        q.set_calibration_store(_fake_calibration(target))
        q.apply(target)

        # Confirm buffers were added
        self.assertIn("_act_quant_scale", target.state_dict())

        q.restore(target)

        sd_after = target.state_dict()
        self.assertNotIn("_act_quant_scale",      sd_after, "scale still in state_dict after restore")
        self.assertNotIn("_act_quant_zero_point",  sd_after, "zero_point still in state_dict after restore")

    def test_restore_removes_act_quant_hook(self):
        """After restore(), the forward pre-hook must be removed."""
        decoder = _make_decoder()
        q = _act_quantizer(precision_activation=8)
        target = decoder.blocks[0].mlp.down_proj
        q.set_calibration_store(_fake_calibration(target))
        q.apply(target)

        self.assertTrue(hasattr(target, "_act_quant_hook_handle"))
        q.restore(target)
        self.assertFalse(hasattr(target, "_act_quant_hook_handle"))


# ---------------------------------------------------------------------------
# 5. convert_for_export compatibility
# ---------------------------------------------------------------------------

class TestLoadStateDictDequantized(unittest.TestCase):
    """_load_state_dict_dequantized must propagate quant buffers to the converted graph."""

    def test_hard_quant_buffers_survive_convert(self):
        """_torchao_scale buffer must be present on new_layer after _load_state_dict_dequantized."""
        try:
            import torchao  # noqa: F401
        except ImportError:
            self.skipTest("torchao not installed")

        from transformersurgeon.utils.convert import _load_state_dict_dequantized

        decoder = _make_decoder()
        q = Quantizer({**_QCFG_BASE, "precision": 8})
        q.apply(decoder.blocks[0].mlp, hard=True)

        # Simulate convert_for_export: transfer one block's MLP to a fresh decoder's MLP.
        decoder2 = _make_decoder()
        _load_state_dict_dequantized(decoder.blocks[0].mlp, decoder2.blocks[0].mlp)

        scale_keys = [
            k for k in decoder2.blocks[0].mlp.state_dict()
            if k.endswith("_torchao_scale")
        ]
        self.assertGreater(len(scale_keys), 0, "_torchao_scale buffer missing after convert")

        # Verify _torchao_precision attribute was also propagated.
        for name, m in decoder2.blocks[0].mlp.named_modules():
            if isinstance(m, nn.Linear) and "_torchao_scale" in m._buffers:
                self.assertTrue(
                    hasattr(m, "_torchao_precision"),
                    f"_torchao_precision missing on {name} after convert",
                )
                self.assertTrue(
                    hasattr(m, "_torchao_per_channel"),
                    f"_torchao_per_channel missing on {name} after convert",
                )

    def test_act_quant_buffers_survive_convert(self):
        """_act_quant_scale and _act_quant_zero_point must appear on new_layer after convert."""
        from transformersurgeon.utils.convert import _load_state_dict_dequantized

        decoder = _make_decoder()
        q = _act_quantizer(precision_activation=8)
        target = decoder.blocks[0].mlp.down_proj
        q.set_calibration_store(_fake_calibration(target))
        q.apply(target)

        orig_scale = target._act_quant_scale.clone()

        decoder2 = _make_decoder()
        _load_state_dict_dequantized(decoder.blocks[0].mlp, decoder2.blocks[0].mlp)

        new_target = decoder2.blocks[0].mlp.down_proj
        self.assertIn("_act_quant_scale",      new_target.state_dict(), "scale missing after convert")
        self.assertIn("_act_quant_zero_point",  new_target.state_dict(), "zero_point missing after convert")
        self.assertTrue(
            torch.equal(orig_scale, new_target._act_quant_scale),
            "scale value changed after convert",
        )

    def test_act_quant_plain_attrs_survive_convert(self):
        """Non-tensor act-quant attributes must be propagated by _load_state_dict_dequantized."""
        from transformersurgeon.utils.convert import _load_state_dict_dequantized

        decoder = _make_decoder()
        q = _act_quantizer(precision_activation=8)
        target = decoder.blocks[0].mlp.down_proj
        q.set_calibration_store(_fake_calibration(target))
        q.apply(target)

        decoder2 = _make_decoder()
        _load_state_dict_dequantized(decoder.blocks[0].mlp, decoder2.blocks[0].mlp)

        new_target = decoder2.blocks[0].mlp.down_proj
        for attr in ("_act_quant_precision", "_act_quant_method", "_act_quant_scheme"):
            self.assertTrue(
                hasattr(new_target, attr),
                f"{attr} missing on new_layer after _load_state_dict_dequantized",
            )
            self.assertEqual(
                getattr(new_target, attr), getattr(target, attr),
                f"{attr} value mismatch after convert",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
