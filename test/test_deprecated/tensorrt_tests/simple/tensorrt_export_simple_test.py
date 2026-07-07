"""
Minimal smoke test for the torch.export -> torch-tensorrt pipeline.

Mirrors ``test/executorch_tests/simple/executorch_export_simple_test.py`` but
lowers to a TensorRT engine instead of an ExecuTorch program. Kept independent
of transformer-surgeon's model/compression code so it can run as a fast sanity
check that the local torch / torch-tensorrt / CUDA stack can actually export
and execute a compiled graph, before running the heavier Qwen2 pipeline tests
in ``test/tensorrt_tests/exporter_function``.

Requires a CUDA device and the ``torch_tensorrt`` package; skips otherwise.

Run:
    python -m pytest test/tensorrt_tests/simple/tensorrt_export_simple_test.py -v
"""

import os
import tempfile
import unittest

import torch
from torch.export import Dim, export

try:
    import torch_tensorrt
    _HAS_TORCH_TRT = True
except ImportError:
    _HAS_TORCH_TRT = False

_HAS_CUDA = torch.cuda.is_available()

data_type = torch.float32

INPUT_SIZE = (16, 16)
EMBEDDING_SIZE = (32,)
OUTPUT_SIZE = (16,)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1a = torch.nn.Linear(INPUT_SIZE[0], EMBEDDING_SIZE[0], dtype=data_type)
        self.layer1b = torch.nn.Linear(INPUT_SIZE[1], EMBEDDING_SIZE[0], dtype=data_type)
        self.layer2 = torch.nn.Linear(EMBEDDING_SIZE[0], OUTPUT_SIZE[0], dtype=data_type)

    def forward(self, inputA: torch.Tensor, inputB: torch.Tensor) -> torch.Tensor:
        output = self.layer1a(inputA) + self.layer1b(inputB)
        return self.layer2(output)


@unittest.skipUnless(
    _HAS_TORCH_TRT and _HAS_CUDA,
    "torch-tensorrt and a CUDA device are required for TensorRT export tests",
)
class TestSimpleTensorRTExport(unittest.TestCase):
    """End-to-end sanity check: export -> torch_tensorrt.dynamo.compile -> save -> reload -> compare."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        cls.device = torch.device("cuda:0")
        cls.model = SimpleModel().to(cls.device)
        cls.model.eval()

        example_bs = 16
        cls.example_inputs = (
            torch.randn(example_bs, INPUT_SIZE[0], dtype=data_type, device=cls.device),
            torch.randn(example_bs, INPUT_SIZE[1], dtype=data_type, device=cls.device),
        )

        batch = Dim("batch", max=256)
        dynamic_shapes = {
            "inputA": (batch, INPUT_SIZE[0]),
            "inputB": (batch, INPUT_SIZE[1]),
        }

        exported_program = export(cls.model, cls.example_inputs, dynamic_shapes=dynamic_shapes)

        cls.trt_module = torch_tensorrt.dynamo.compile(
            exported_program,
            arg_inputs=list(cls.example_inputs),
            enabled_precisions={torch.float32},
            min_block_size=1,
            device=str(cls.device),
        )

        cls.tmpdir = tempfile.mkdtemp()
        cls.engine_path = os.path.join(cls.tmpdir, "simple_model_trt.pt2")
        torch_tensorrt.save(
            cls.trt_module,
            cls.engine_path,
            arg_inputs=list(cls.example_inputs),
            output_format="exported_program",
        )

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_engine_file_created(self):
        self.assertTrue(os.path.isfile(self.engine_path))
        self.assertGreater(os.path.getsize(self.engine_path), 0)

    def test_trt_output_matches_torch_dynamic_batch(self):
        """In-process compiled module honors the dynamic batch Dim used at compile time."""
        bs = 4  # != the batch size (16) used to build example_inputs
        eval_inputs = (
            torch.randn(bs, INPUT_SIZE[0], dtype=data_type, device=self.device),
            torch.randn(bs, INPUT_SIZE[1], dtype=data_type, device=self.device),
        )

        with torch.no_grad():
            ref_output = self.model(*eval_inputs)
            trt_output = self.trt_module(*eval_inputs)

        max_err = torch.max(torch.abs(ref_output - trt_output)).item()
        self.assertLess(max_err, 1e-2, f"torch vs. in-process TRT max error too high: {max_err}")

    def test_reloaded_trt_output_matches_torch(self):
        """Reloaded module matches torch at the exact shape used to compile/save.

        Unlike the in-process module, the reloaded ``ExportedProgram`` does not
        reliably re-derive the dynamic batch Dim guard on this torch-tensorrt
        version, so this only exercises the save/reload round-trip itself (the
        actual Qwen2 TensorRT exporter never uses dynamic shapes — every input
        is a fixed single-token shape — so a fixed-shape reload check matches
        what that pipeline relies on).
        """
        eval_inputs = self.example_inputs

        with torch.no_grad():
            ref_output = self.model(*eval_inputs)

        loaded_module = torch_tensorrt.load(self.engine_path).module()
        with torch.no_grad():
            reloaded_output = loaded_module(*eval_inputs)

        max_err_reloaded = torch.max(torch.abs(ref_output - reloaded_output)).item()
        self.assertLess(max_err_reloaded, 1e-2, f"torch vs. reloaded TRT max error too high: {max_err_reloaded}")


if __name__ == "__main__":
    unittest.main()
