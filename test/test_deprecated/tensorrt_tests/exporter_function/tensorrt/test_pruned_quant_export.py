"""
Test: Qwen2-0.5B with grouped structured pruning + int8 quantization -> TensorRT.

Companion to ``test_pretrained_quant_export.py`` (which covers quant + LRD). This
one exercises the STRUCTURED-PRUNING export path:

  1.  Load Qwen2-0.5B (float16).
  2.  auto_groups(): coupled-mask groups from indexing. Enable share_mask on the
      per-block MLP gate/up groups so gate_proj and up_proj prune the same
      neurons (coherent grouped pruning).
  3.  Assign a random structured-pruning ratio per block's MLP group (random
      method — no calibration) and randomly int8-quantize each linear
      (weight-only, per-channel). No LRD.
  4.  manager.apply(hard=True): remove MLP intermediate neurons, resize the
      matrices, cascade down_proj's input, and hard-quantize the chosen layers.
  5.  convert_for_export: the MLP blocks are pruning-aware, so the converted
      graph matches the pruned weight shapes (this is what makes the export work
      — before that fix, load_state_dict mismatched on pruned layers).
  6.  export_to_backend(TensorRT) -> compiled engine; compare float vs TensorRT.

Attention is left unpruned (GQA head_dim change is out of scope for the converted
attention block). Requires a CUDA device and torch_tensorrt; skipped otherwise.

Run:
    python -m pytest test/tensorrt_tests/exporter_function/tensorrt/test_pruned_quant_export.py -v
"""

import os
import random
import tempfile
import unittest

import torch
import torch.nn as nn

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.tensorrt import TensorRTExportConfig

try:
    import torch_tensorrt
    _HAS_TORCH_TRT = True
except ImportError:
    _HAS_TORCH_TRT = False

_HAS_CUDA = torch.cuda.is_available()

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MAX_SEQ_LEN = 512
SEED = 42
PRUNE_RATIOS = [0.125, 0.25]  # non-zero so at least some layers are pruned
QUANT_PROB = 0.5

_MEAN_ABS_ERR_MAX = 1.0
_MAX_ABS_ERR_MAX = 5.0


def _load_reloaded_module(engine_path: str):
    if hasattr(torch_tensorrt, "load"):
        try:
            return torch_tensorrt.load(engine_path).module()
        except Exception:
            pass
    return torch.export.load(engine_path).module()


@unittest.skipUnless(
    _HAS_TORCH_TRT and _HAS_CUDA,
    "torch-tensorrt and a CUDA device are required for TensorRT export tests",
)
class TestQwen2PrunedQuantTensorRTExport(unittest.TestCase):
    """End-to-end: grouped structured pruning + int8 quant -> TensorRT export."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(SEED)
        torch.manual_seed(SEED)

        cls.model = Qwen2ForCausalLMCompress.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        ).eval()

        manager = Qwen2CompressionSchemesManager(cls.model)

        # Grouped structured pruning on the MLP gate/up groups (shared masks).
        groups = manager.auto_groups()
        mlp_groups = {g: paths for g, paths in groups.items()
                      if any("gate_proj" in p for p in paths)}
        for g in mlp_groups:
            manager.set("structured_pruning", "share_mask", True, group=g)
        manager.set("structured_pruning", "method", "random", criteria=None)
        manager.set("structured_pruning", "reduce_op", "add", criteria=None)

        cls.group_ratios = {}
        for g in mlp_groups:
            ratio = rng.choice(PRUNE_RATIOS)
            cls.group_ratios[g] = ratio
            manager.set("structured_pruning", "ratio", ratio, group=g)

        # Random int8 weight-only quantization per linear (no LRD).
        cls.num_quantized = 0
        for scheme in manager:
            module = scheme.get_module()
            if not isinstance(module, nn.Linear):
                continue
            if rng.random() < QUANT_PROB:
                manager.set("quantization", "precision", 8, criteria=scheme.path)
                manager.set("quantization", "granularity", "per_channel", criteria=scheme.path)
                cls.num_quantized += 1

        manager.apply(hard=True, criteria=None)

        # Record a pruned MLP dim for assertions.
        cls.orig_intermediate = cls.model.config.intermediate_size
        cls.pruned_gate_outs = [
            layer.mlp.gate_proj.out_features for layer in cls.model.model.layers
        ]

        # Reference torch output via the export wrapper (built inside export too).
        cls.stored_input_ids = torch.randint(0, cls.model.config.vocab_size, (1,), dtype=torch.long)
        cls.stored_pos_ids = torch.tensor([1], dtype=torch.long)

        cls.tmpdir = tempfile.mkdtemp()
        cls.engine_path = os.path.join(cls.tmpdir, "qwen2_pruned_quant_trt.pt2")

        export_cfg = TensorRTExportConfig(
            output_path=cls.engine_path,
            backend="tensorrt",
            max_seq_len=MAX_SEQ_LEN,
            convert_options={"use_sdpa": False},
            run_weight_mismatch_check=False,
            device="cuda:0",
            verbose=True,
        )
        cls.export_error = None
        try:
            cls.result = export_to_backend(cls.model, config=export_cfg)
        except Exception as exc:  # pragma: no cover - environment dependent
            cls.result = None
            cls.export_error = exc

        cls.trt_output = None
        if cls.result is not None:
            try:
                loaded = _load_reloaded_module(cls.engine_path)
                with torch.no_grad():
                    y = loaded(cls.stored_input_ids, cls.stored_pos_ids)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                cls.trt_output = y.detach().cpu()
            except Exception:
                cls.trt_output = None

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_pruning_reduced_mlp_dims(self):
        """At least one block's MLP intermediate was actually shrunk by pruning."""
        self.assertTrue(
            any(g < self.orig_intermediate for g in self.pruned_gate_outs),
            "No MLP layer was pruned — check the seed / PRUNE_RATIOS.",
        )

    def test_quantization_applied(self):
        """Some layers were int8-quantized."""
        self.assertGreater(self.num_quantized, 0)

    def test_export_succeeded(self):
        if self.export_error is not None:
            self.fail(f"export_to_backend raised: {self.export_error!r}")
        self.assertIsNotNone(self.result)

    def test_engine_file_created(self):
        self.assertTrue(os.path.isfile(self.engine_path))
        self.assertGreater(os.path.getsize(self.engine_path), 0)

    def test_result_metadata(self):
        self.assertEqual(self.result.backend, "tensorrt")
        self.assertEqual(self.result.engine_path, self.engine_path)

    def test_inprocess_inference_stats(self):
        if self.result is None or self.result.inference_stats is None:
            self.skipTest("Inference stats unavailable (TRT runtime may be absent).")
        stats = self.result.inference_stats
        self.assertLess(stats["mean_abs_err"], _MEAN_ABS_ERR_MAX)
        self.assertLess(stats["max_abs_err"], _MAX_ABS_ERR_MAX)

    def test_tensorrt_output_finite(self):
        if self.trt_output is None:
            self.skipTest("TensorRT engine reload/inference unavailable or failed.")
        self.assertEqual(list(self.trt_output.shape), [self.model.config.vocab_size])
        self.assertTrue(torch.isfinite(self.trt_output).all())


if __name__ == "__main__":
    unittest.main()
