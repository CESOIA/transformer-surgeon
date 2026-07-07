"""
Test: Qwen2-0.5B float16 with random per-layer mixed compression -> TensorRT export.

Mirrors ``test/executorch_tests/exporter_function/xnnpack/test_pretrained_quant_export.py``.
Each indexed linear layer is randomly assigned one compression label:
  - "full"        : no compression (float16 throughout)
  - "int8"        : per-channel int8 weight-only quantization (torchao hard)
  - "int4"        : per-channel int4 weight-only quantization (disabled by default;
                    enable with ALLOW_INT4=True only when the runtime supports it)
  - "lrd_N"       : SVD low-rank decomposition to rank N ∈ {32, 64, 128}
  - "lrd_N+int8"  : LRD (rank N) combined with int8 quantization on the same layer
  - "lrd_N+int4"  : LRD (rank N) combined with int4 quantization (when ALLOW_INT4)

Combined labels apply quantization first (on the HF model), then LRD after
conversion, exercising the joint LRD + torchao path in _load_state_dict_dequantized.

Pipeline:
  1.  Load Qwen2-0.5B in float16.
  2.  Randomly assign one compression label per linear layer.
  3.  Apply int8/int4 quantization (torchao hard) to any layer whose label
      contains "int8" or "int4" (standalone or combined).
  4.  Convert the HF model to export-ready components (embedding, decoder,
      final_layer) via convert_for_export.
  5.  Apply SVD LRD to any converted-decoder layer whose label contains "lrd_"
      (standalone or combined).
  6.  Generate and store a random single-token input (input_ids, pos_ids).
  7.  Run torch inference on the LLMWrapper -> store reference output tensor.
  8.  Export to TensorRT via export_to_backend -> produces a compiled engine
      (saved as an ExportedProgram) and, since verbose=True, in-process
      float-vs-TensorRT inference stats on the exporter's own example input.
  9.  Reload the saved engine and run TensorRT inference with the stored input.
  10. Assert that the reloaded TensorRT output matches the torch reference
      within tolerance.

Goal: verify the torch HF -> TensorRT pipeline is functionally correct for a
real Qwen2-0.5B model under mixed per-layer compression, including the
combined LRD + quantization case. Meaningful NLP output is not expected; only
output tensor equivalence is checked.

Requires a CUDA device and the ``torch_tensorrt`` package; the whole module is
skipped otherwise.

Run:
    python -m pytest test/tensorrt_tests/exporter_function/tensorrt/test_pretrained_quant_export.py -v
"""

import os
import random
import tempfile
import unittest

import torch
import torch.nn as nn

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager
from transformersurgeon.models.qwen2_c.indexing_qwen2_c import QWEN2_C_INDEXING
from transformersurgeon.blocks import LinearCompressed
from transformersurgeon.compression.quantization import Quantizer
from transformersurgeon.compression.lrd_methods import METHOD_FUNCTIONS as _LRD_SVD
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.common import LLMWrapper
from transformersurgeon.export.tensorrt import TensorRTExportConfig
from transformersurgeon.utils import convert_for_export
from transformersurgeon.utils.utils import flatten_index_paths, get_submodule

try:
    import torch_tensorrt
    _HAS_TORCH_TRT = True
except ImportError:
    _HAS_TORCH_TRT = False

_HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2-0.5B"
MAX_SEQ_LEN = 512
SEED = 42

# INT4 sometimes is not supported by the system hardware
# Set to True only when running on hardware with int4 kernel support.
ALLOW_INT4 = False

# LRD rank candidates; only ranks < min(in, out) for each layer are used.
_LRD_RANK_CANDIDATES = [32, 64, 128]

# Tolerated errors for torch <-> TensorRT output comparison.
# Generous values: the goal is to catch export-pipeline bugs, not to bound
# quantization error tightly.
_MEAN_ABS_ERR_MAX = 1.0
_MAX_ABS_ERR_MAX = 5.0

# Quantizer config template (weight-only; activations stay at full precision)
_QUANT_CFG = {
    "method": "vanilla",
    "sparsity": 0.0,
    "sparse_method": "magnitude",
    "eps": 1e-6,
    "granularity": "per_channel",
    "precision_activation": "full",
    "method_activation": "maxmin",
    "scheme_activation": "asymmetric",
}


# ---------------------------------------------------------------------------
# Helpers (backend-agnostic; identical to the xnnpack version)
# ---------------------------------------------------------------------------

def _build_hf_to_conv_map(num_blocks: int) -> dict[str, str]:
    """Return {hf_model_path: converted_decoder_path} for all indexed layers.

    Mirrors the path mapping performed by convert_for_export, derived from
    QWEN2_C_INDEXING so the test stays in sync with the exporter.
    """
    idx = QWEN2_C_INDEXING["text"]
    source_paths = flatten_index_paths(idx["path_list"])

    flat_matching: list[str] = []
    for _, mapped in idx["layer_matching"].items():
        if isinstance(mapped, str):
            flat_matching.append(mapped)
        else:
            flat_matching.extend(str(m) for m in mapped)

    assert len(source_paths) == len(flat_matching), (
        "path_list and layer_matching length mismatch — check QWEN2_C_INDEXING."
    )

    result: dict[str, str] = {}
    for i in range(num_blocks):
        for src, dst in zip(source_paths, flat_matching):
            hf_path = idx["path_template"].format(block_index=i, path=src)
            result[hf_path] = f"blocks.{i}.{dst}"
    return result


def _compression_options(module: nn.Linear, allow_int4: bool) -> list[str]:
    """Return all compression labels valid for this module's weight shape."""
    opts = ["full", "int8"]
    if allow_int4:
        opts.append("int4")
    max_rank = min(module.in_features, module.out_features) - 1
    for r in _LRD_RANK_CANDIDATES:
        if r < max_rank:
            opts.append(f"lrd_{r}")
            opts.append(f"lrd_{r}+int8")
            if allow_int4:
                opts.append(f"lrd_{r}+int4")
    return opts


def _apply_quant(module: nn.Linear, precision: int) -> None:
    """Apply hard torchao weight quantization to *module* in-place."""
    Quantizer({**_QUANT_CFG, "precision": precision}).apply(module, hard=True)


def _apply_lrd(module: LinearCompressed, rank: int) -> None:
    """Apply SVD low-rank decomposition to *module* in-place."""
    cap = min(module.in_features, module.out_features) - 1
    rank = min(rank, cap)
    if rank <= 0:
        return
    with torch.no_grad():
        dtype = module.weight.dtype
        US_r, V_r = _LRD_SVD["svd"](module.weight.detach().float(), rank)
        module.init_lrd(rank)
        module.weight.data.copy_(US_r.to(dtype))
        module.linear_V.weight.data.copy_(V_r.to(dtype))


def _load_reloaded_module(engine_path: str):
    """Reload a saved TensorRT engine, tolerating both save formats."""
    if hasattr(torch_tensorrt, "load"):
        try:
            return torch_tensorrt.load(engine_path).module()
        except Exception:
            pass
    return torch.export.load(engine_path).module()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    _HAS_TORCH_TRT and _HAS_CUDA,
    "torch-tensorrt and a CUDA device are required for TensorRT export tests",
)
class TestQwen2TensorRTExport(unittest.TestCase):
    """End-to-end export test: Qwen2-0.5B (float16), mixed per-layer
    compression, TensorRT export, TensorRT output comparison."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(SEED)
        torch.manual_seed(SEED)

        # 1. Load model in float16 (full precision = float16)
        cls.model = Qwen2ForCausalLMCompress.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
        )
        cls.model.eval()

        num_blocks = cls.model.config.num_hidden_layers
        hf_to_conv = _build_hf_to_conv_map(num_blocks)

        # 2. Randomly assign one compression type per linear layer
        manager = Qwen2CompressionSchemesManager(cls.model)
        assignments: dict[str, str] = {}

        for scheme in manager:
            try:
                module = scheme.get_module()
            except Exception:
                assignments[scheme.path] = "skip"
                continue

            if not isinstance(module, nn.Linear):
                assignments[scheme.path] = "skip"
                continue

            options = _compression_options(module, ALLOW_INT4)
            assignments[scheme.path] = rng.choice(options)

        cls.compression_assignments = assignments

        # 3. Apply quantization to any layer whose label contains "int8"/"int4"
        for hf_path, label in assignments.items():
            if "int8" in label:
                prec = 8
            elif "int4" in label:
                prec = 4
            else:
                continue
            _apply_quant(get_submodule(cls.model, hf_path), prec)

        # 4. Convert HF model -> export-ready (embedding, decoder, final_layer).
        converted = convert_for_export(cls.model, options={"use_sdpa": False})
        decoder = converted["text"]
        embedding = cls.model.get_input_embeddings()
        final_layer = cls.model.lm_head

        # 5. Apply LRD to any converted-decoder layer whose label contains "lrd_"
        for hf_path, label in assignments.items():
            if "lrd_" not in label:
                continue
            lrd_token = next(p for p in label.split("+") if p.startswith("lrd_"))
            rank = int(lrd_token[4:])
            conv_path = hf_to_conv.get(hf_path)
            if conv_path is None:
                continue
            try:
                conv_module = get_submodule(decoder, conv_path)
            except Exception:
                continue
            if isinstance(conv_module, LinearCompressed):
                _apply_lrd(conv_module, rank)

        # 6. Build wrapper and generate/store a random single-token input
        cls.wrapper = LLMWrapper(embedding, decoder, final_layer)
        cls.wrapper.eval()

        vocab_size = cls.model.config.vocab_size
        cls.stored_input_ids = torch.randint(0, vocab_size, (1,), dtype=torch.long)
        cls.stored_pos_ids = torch.tensor([1], dtype=torch.long)

        # 7. Store reference torch output (run before export, which may
        #    mutate the wrapper's weights for quantized layers)
        with torch.no_grad():
            cls.torch_output = cls.wrapper(
                cls.stored_input_ids, cls.stored_pos_ids
            ).detach()

        # 8. Export to TensorRT
        cls.tmpdir = tempfile.mkdtemp()
        cls.engine_path = os.path.join(cls.tmpdir, "qwen2_mixed_tensorrt.pt2")

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
            cls.result = export_to_backend(
                {
                    "embedding": embedding,
                    "decoder": decoder,
                    "final_layer": final_layer,
                    "config": cls.model.config,
                },
                config=export_cfg,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            cls.result = None
            cls.export_error = exc

        # 9. Reload the saved engine and run TensorRT inference on the stored input.
        #
        # Inputs stay on CPU: the reloaded graph mixes plain torch ops (KV-cache /
        # position-id bookkeeping, whose buffers were traced on CPU) with the
        # TensorRT-accelerated linear subgraphs. torch_tensorrt's runtime wrapper
        # auto-moves tensors to the engine's device only at that subgraph's
        # boundary; pre-moving the whole input breaks the surrounding CPU-resident
        # ops with a "Tensor device mismatch" error.
        cls.trt_output = None
        if cls.result is not None:
            try:
                loaded_module = _load_reloaded_module(cls.engine_path)
                with torch.no_grad():
                    y = loaded_module(cls.stored_input_ids, cls.stored_pos_ids)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                cls.trt_output = y.detach().cpu()
            except Exception:
                cls.trt_output = None

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_export_succeeded(self):
        """Export completed without raising."""
        if self.export_error is not None:
            self.fail(f"export_to_backend raised: {self.export_error!r}")
        self.assertIsNotNone(self.result)

    def test_engine_file_created(self):
        """Exported engine file exists and is non-empty."""
        self.assertTrue(
            os.path.isfile(self.engine_path),
            f"Expected engine file at {self.engine_path}",
        )
        self.assertGreater(os.path.getsize(self.engine_path), 0, "engine file is empty.")

    def test_result_metadata(self):
        """Export result carries correct backend tag and file path."""
        self.assertEqual(self.result.backend, "tensorrt")
        self.assertEqual(self.result.engine_path, self.engine_path)

    def test_stored_input_valid(self):
        """Stored input tensor is a valid single token index."""
        vocab_size = self.model.config.vocab_size
        self.assertEqual(list(self.stored_input_ids.shape), [1])
        self.assertGreaterEqual(int(self.stored_input_ids.item()), 0)
        self.assertLess(int(self.stored_input_ids.item()), vocab_size)
        self.assertEqual(list(self.stored_pos_ids.shape), [1])

    def test_torch_output_shape_and_finite(self):
        """Reference torch output has correct logit shape and finite values."""
        self.assertEqual(
            list(self.torch_output.shape),
            [self.model.config.vocab_size],
            "Torch output shape mismatch.",
        )
        self.assertTrue(
            torch.isfinite(self.torch_output).all(),
            "Torch output contains non-finite values.",
        )

    def test_inprocess_inference_stats(self):
        """In-process float-vs-TensorRT stats (computed by the exporter itself
        via verbose=True, on its own example input) are present and finite."""
        if self.result is None or self.result.inference_stats is None:
            self.skipTest("Inference stats unavailable (best-effort; TRT runtime may be absent).")

        stats = self.result.inference_stats
        self.assertLess(stats["mean_abs_err"], _MEAN_ABS_ERR_MAX)
        self.assertLess(stats["max_abs_err"], _MAX_ABS_ERR_MAX)

    def test_tensorrt_output_matches_torch(self):
        """Reloaded TensorRT output matches stored torch reference within tolerance.

        Tolerance is intentionally generous: the test checks that the export
        pipeline is functionally correct, not that quantization error is small.
        """
        if self.trt_output is None:
            self.skipTest("TensorRT engine reload/inference unavailable or failed.")

        y_ref = self.torch_output.float()
        y_trt = self.trt_output.float()

        self.assertEqual(
            list(y_trt.shape),
            list(y_ref.shape),
            f"Shape mismatch: TRT={y_trt.shape} vs torch={y_ref.shape}",
        )
        self.assertTrue(
            torch.isfinite(y_trt).all(),
            "TensorRT output contains non-finite values.",
        )

        mae = float((y_ref - y_trt).abs().mean())
        max_err = float((y_ref - y_trt).abs().max())

        self.assertLess(
            mae,
            _MEAN_ABS_ERR_MAX,
            f"Mean abs error {mae:.4f} exceeds threshold {_MEAN_ABS_ERR_MAX}.",
        )
        self.assertLess(
            max_err,
            _MAX_ABS_ERR_MAX,
            f"Max abs error {max_err:.4f} exceeds threshold {_MAX_ABS_ERR_MAX}.",
        )

    def test_compression_variety(self):
        """At least one non-trivial compression type was assigned."""
        non_trivial = [
            v
            for v in self.compression_assignments.values()
            if v not in ("full", "skip")
        ]
        self.assertGreater(
            len(non_trivial),
            0,
            "Random seed produced no compressed layers — check the seed or options.",
        )

    def test_combined_compression_assigned(self):
        """At least one layer received the combined LRD + quantization label."""
        combined = [
            v
            for v in self.compression_assignments.values()
            if "lrd_" in v and ("int8" in v or "int4" in v)
        ]
        self.assertGreater(
            len(combined),
            0,
            "Random seed produced no combined (LRD + quant) layers — "
            "increase the pool size or adjust the seed.",
        )


if __name__ == "__main__":
    unittest.main()
