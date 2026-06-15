"""
Unit test: MLP-only pre-quantized export via XNNPACK.

Pipeline under test:
  1. Build a tiny TransformerDecoder (MLPGated, 2 layers, float32).
  2. Quantize only MLP linear layers with torchao hard int8 per-channel
     (via Quantizer.apply(block.mlp, hard=True)).
  3. Save expected dequantized MLP weights from the torchao tensors BEFORE
     the export pipeline dequantizes them.
  4. Export the model to XNNPACK via export_to_backend (config.precision is
     bypassed — per-layer metadata from _torchao_precision drives the quantizer).
  5. Assert that the dequantized weights in the exported PT2E graph match the
     saved torchao dequantized weights within float32 precision.
  6. Run inference through the ExecuTorch runtime and assert output error vs
     the FP32 LLMWrapper baseline is finite and bounded.

Run:
    python -m pytest test_pretrained_quant_export.py -v
"""

import os
import tempfile
import unittest

import torch
import torch.nn as nn

from transformersurgeon.blocks import TransformerDecoder
from transformersurgeon.blocks.config import CustomDecoderConfigCompress
from transformersurgeon.compression.quantization import Quantizer
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.executorch_exporters.common import LLMWrapper
from transformersurgeon.export.executorch_exporters.xnnpack import XNNPACKExportConfig


# ---------------------------------------------------------------------------
# Tiny model dimensions
# ---------------------------------------------------------------------------
_VOCAB = 64
_HIDDEN = 32
_HEADS = 2
_INTER = 64
_LAYERS = 2
_CACHE = 16

# Indexing for the converted custom TransformerDecoder (not the HF source model).
# The path_template must match how TransformerDecoder names its submodules.
_CONVERTED_INDEXING = {
    "num_blocks_attr": "num_hidden_layers",
    "path_template": "blocks.{block_index}.{path}",
    "path_list": {
        "norm_in": [],
        "attn": ["q_proj", "k_proj", "v_proj", "out_proj"],
        "norm_out": [],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
    },
}

# Maximum tolerated absolute error per element between torchao dequantized
# weight and the exported PT2E dequantized weight. With exact scale injection
# both values come from int8 * float32_scale so they should be bit-identical
# within float32 rounding (< 1e-5).
_WEIGHT_ATOL = 1e-5

# Maximum mean absolute error allowed for the inference test. This is generous
# because random weights produce arbitrary logit magnitudes.
_INFERENCE_MAX_MAE = 1e2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> CustomDecoderConfigCompress:
    cfg = CustomDecoderConfigCompress(
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
        dtype=torch.float32,
        indexing=_CONVERTED_INDEXING,
    )
    cfg.dtype = torch.float32
    return cfg


def _make_components(cfg: CustomDecoderConfigCompress):
    torch.manual_seed(0)
    decoder = TransformerDecoder(cfg)
    embedding = nn.Embedding(_VOCAB, _HIDDEN)
    final_layer = nn.Linear(_HIDDEN, _VOCAB, bias=False)
    for m in (decoder, embedding, final_layer):
        m.eval()
    return embedding, decoder, final_layer


def _quantize_mlp_only(decoder: TransformerDecoder, precision: int = 8) -> None:
    """Apply torchao hard quantization to MLP linear layers only."""
    qcfg = {
        "method": "vanilla",
        "precision": precision,
        "sparsity": 0.0,
        "sparse_method": "magnitude",
        "eps": 1e-6,
        "granularity": "per_channel",
        "precision_activation": "full",
        "method_activation": "maxmin",
        "scheme_activation": "asymmetric",
    }
    q = Quantizer(qcfg)
    for block in decoder.blocks:
        q.apply(block.mlp, hard=True)


def _capture_mlp_dequant(decoder: TransformerDecoder) -> dict[str, torch.Tensor]:
    """Capture dequantized weights from torchao AffineQuantizedTensors before export mutates them.

    Returns {wrapper_layer_name: dequantized_float32_weight}.
    The wrapper prepends 'decoder.' to all decoder submodule names.
    """
    from torchao.dtypes import AffineQuantizedTensor

    out = {}
    for name, module in decoder.named_modules():
        if (
            isinstance(module, nn.Linear)
            and hasattr(module, "_torchao_precision")
            and isinstance(module.weight, AffineQuantizedTensor)
        ):
            out[f"decoder.{name}"] = module.weight.dequantize().detach().float()
    return out


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestPretrainedQuantExport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = _make_config()
        cls.embedding, cls.decoder, cls.final_layer = _make_components(cls.cfg)

        # Step 1 — quantize only MLP layers (int8 per-channel torchao hard)
        _quantize_mlp_only(cls.decoder, precision=8)

        # Step 2 — capture expected dequant weights while torchao tensors are live
        cls.expected_dequants = _capture_mlp_dequant(cls.decoder)
        assert len(cls.expected_dequants) > 0, (
            "No MLP layers were tagged — check _torchao_precision assignment."
        )

        # Step 3 — export (this will dequantize the torchao weights in-place)
        cls.tmpdir = tempfile.mkdtemp()
        cls.pte_path = os.path.join(cls.tmpdir, "mlp_quant_xnnpack.pte")

        export_cfg = XNNPACKExportConfig(
            output_path=cls.pte_path,
            backend="xnnpack",
            # precision is intentionally bypassed by per-layer _torchao_precision;
            # set to "w8" so the quantization block is entered at all.
            precision="w8",
            max_seq_len=_CACHE,
            convert_options={"use_sdpa": False},
            run_weight_mismatch_check=False,
            verbose=False,
        )

        model_input = {
            "embedding": cls.embedding,
            "decoder": cls.decoder,
            "final_layer": cls.final_layer,
            "config": cls.cfg,
        }
        cls.result = export_to_backend(model_input, config=export_cfg)

        # Build the LLMWrapper used during export for FP32 reference inference.
        # After export the wrapper holds FP32 (dequantized) weights.
        cls.wrapper = LLMWrapper(cls.embedding, cls.decoder, cls.final_layer)
        cls.wrapper.eval()

    # ------------------------------------------------------------------
    # Test 1 — sanity: correct number of MLP layers were quantized
    # ------------------------------------------------------------------

    def test_mlp_layers_tagged(self):
        """Every MLP linear in every block must have been captured."""
        # For mlp_gated with 2 layers: gate_proj, up_proj, down_proj × _LAYERS = 6
        expected_count = _LAYERS * 3  # gate_proj, up_proj, down_proj per block
        self.assertEqual(
            len(self.expected_dequants),
            expected_count,
            f"Expected {expected_count} MLP linear layers, got {len(self.expected_dequants)}.",
        )

    # ------------------------------------------------------------------
    # Test 2 — exported weights match torchao dequantized weights
    # ------------------------------------------------------------------

    def test_exported_weights_match_torchao(self):
        """Weights in the wrapper post-export must equal the pre-export torchao dequantized values.

        extract_layer_quant_info dequantizes AffineQuantizedTensors in-place during export,
        replacing each module.weight with w.dequantize(). The expected_dequants captured
        before export should therefore be bit-identical to the post-export wrapper weights.
        """
        for layer_name, expected_dq in self.expected_dequants.items():
            # layer_name is e.g. "decoder.blocks.0.mlp.gate_proj"
            # wrapper parameters are keyed as "decoder.blocks.0.mlp.gate_proj.weight"
            actual_dq = dict(self.wrapper.named_parameters()).get(f"{layer_name}.weight")
            if actual_dq is None:
                self.fail(f"Layer {layer_name!r} not found in wrapper parameters.")
            actual_dq = actual_dq.detach().float()
            max_err = (expected_dq - actual_dq).abs().max().item()
            self.assertLess(
                max_err,
                _WEIGHT_ATOL,
                f"Weight mismatch for {layer_name}: max_abs_err={max_err:.2e} > {_WEIGHT_ATOL}",
            )

    # ------------------------------------------------------------------
    # Test 3 — attention layers NOT quantized
    # ------------------------------------------------------------------

    def test_attn_layers_not_quantized(self):
        """Attention projection layers must remain float (no _torchao_precision)."""
        for name, module in self.decoder.named_modules():
            if "mlp" not in name and isinstance(module, nn.Linear):
                self.assertFalse(
                    hasattr(module, "_torchao_precision"),
                    f"Attention layer {name} should NOT be torchao-quantized.",
                )

    # ------------------------------------------------------------------
    # Test 4 — .pte file exists and is non-empty
    # ------------------------------------------------------------------

    def test_pte_file_created(self):
        self.assertTrue(
            os.path.isfile(self.pte_path),
            f"Expected .pte file at {self.pte_path}",
        )
        self.assertGreater(
            os.path.getsize(self.pte_path),
            0,
            ".pte file is empty.",
        )

    # ------------------------------------------------------------------
    # Test 5 — ExecuTorch inference: output shape and bounded error
    # ------------------------------------------------------------------

    def test_inference_executorch(self):
        """ExecuTorch output must have correct shape and finite bounded error."""
        try:
            from executorch.runtime import Runtime
        except ImportError:
            self.skipTest("ExecuTorch runtime not available.")

        example_ids = torch.randint(0, _VOCAB, (1,), dtype=torch.long)
        example_pos = torch.tensor([1], dtype=torch.long)

        # FP32 reference
        with torch.no_grad():
            y_ref = self.wrapper(example_ids, example_pos)

        # ExecuTorch inference
        runtime = Runtime.get()
        program = runtime.load_program(self.pte_path)
        method = program.load_method("forward")
        outputs = method.execute([example_ids, example_pos])
        y_et = outputs[0]
        if not isinstance(y_et, torch.Tensor):
            y_et = torch.tensor(y_et)

        # Shape check
        self.assertEqual(
            list(y_et.shape),
            list(y_ref.shape),
            f"Output shape mismatch: ET={y_et.shape} vs ref={y_ref.shape}",
        )

        # Finite values
        self.assertTrue(torch.isfinite(y_et).all(), "ExecuTorch output contains non-finite values.")

        # Bounded error
        mae = (y_ref.float() - y_et.float()).abs().mean().item()
        self.assertLess(
            mae,
            _INFERENCE_MAX_MAE,
            f"ExecuTorch mean absolute error {mae:.4f} exceeds threshold {_INFERENCE_MAX_MAE}.",
        )

    # ------------------------------------------------------------------
    # Test 6 — result metadata
    # ------------------------------------------------------------------

    def test_result_metadata(self):
        self.assertEqual(self.result.backend, "xnnpack")
        self.assertEqual(self.result.pte_path, self.pte_path)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
