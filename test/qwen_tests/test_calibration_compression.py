"""
Unit tests for Qwen2 compression schemes.

Covers:
  - SVD-LLM-v2 compression (standard calibration)
  - AA-SVD compression    (cascade calibration)
  - Covariance equivalence: standard vs. cascade calibration modes

All tests use WikiText-2 for calibration data.

Run:
    python -m pytest test_qwen_compression.py -v
    python test_qwen_compression.py [gpu_id]
"""
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformersurgeon import (
    Qwen2CompressionSchemesManager,
    Qwen2ForCausalLMCompress,
)

# ---------------------------------------------------------------------------
# Global test settings
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2-0.5B"
GPU_NUM = 0

# WikiText-2 calibration settings (kept small for fast test runs)
SEQ_LEN = 2048
NUM_CALIBRATION_SAMPLES = 8
RANDOM_SEED = 42

# Compression rank config (mirrors the original svd-llm-v2 tests)
MLP_RANK_OVERRIDES = {
    "mlp.up_proj": 497,
    "mlp.gate_proj": 497,
    "mlp.down_proj": 497,
}
ATTN_RANK = 64


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _device():
    return torch.device(f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu")


def _load_model(device):
    model = Qwen2ForCausalLMCompress.from_pretrained(MODEL_NAME, torch_dtype="auto")
    return model.to(device).eval()


def _build_wikitext_calibration_loader(tokenizer, num_samples=NUM_CALIBRATION_SAMPLES, seq_len=SEQ_LEN):
    """Return a DataLoader of fixed-length token chunks from WikiText-2."""
    raw = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1",
        split="train",
    )
    texts = [t for t in raw["page"] if isinstance(t, str) and len(t) > 0]
    corpus = "\n\n".join(texts)

    token_ids = tokenizer(corpus, truncation=False, padding=False,
                          return_attention_mask=False)["input_ids"]

    max_samples = len(token_ids) // seq_len
    actual = min(num_samples, max_samples)

    rng = random.Random(RANDOM_SEED)
    examples = []
    for _ in range(actual):
        start = rng.randint(0, len(token_ids) - seq_len - 1)
        examples.append({
            "input_ids": torch.tensor(token_ids[start:start + seq_len], dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
        })

    return DataLoader(examples, batch_size=1, shuffle=False)


def _configure_svd_llm_v2(manager):
    manager.set("lrd", "method", "svd-llm-v2", criteria="all", verbose=False)
    for scheme in manager.iter_filtered(criteria="all"):
        if hasattr(scheme.get_module(), "weight"):
            manager.set("lrd", "rank", 64, criteria=scheme.path, verbose=False)
    for layer_key, rank in MLP_RANK_OVERRIDES.items():
        manager.set("lrd", "rank", rank, criteria=layer_key, verbose=False)


def _configure_aa_svd(manager):
    manager.set("lrd", "method", "aa-svd", verbose=False)
    manager.set("lrd", "rank", ATTN_RANK, criteria=["attn"], verbose=False)
    for layer_key, rank in MLP_RANK_OVERRIDES.items():
        manager.set("lrd", "rank", rank, criteria=layer_key, verbose=False)


def _param_count(model):
    return sum(p.numel() for p in model.parameters())


def _run_forward(model, tokenizer, device):
    inputs = tokenizer("The quick brown fox", return_tensors="pt").to(device)
    with torch.no_grad():
        return model(**inputs)


# ---------------------------------------------------------------------------
# Test: SVD-LLM-v2 (fuses svd_llm_v2_test.py + qwen2_0_5b_wikitext_svd_llm_v2_test.py)
# ---------------------------------------------------------------------------
class TestSvdLlmV2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = _device()
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        cls.model = _load_model(cls.device)
        cls.params_before = _param_count(cls.model)

        loader = _build_wikitext_calibration_loader(cls.tokenizer)
        manager = Qwen2CompressionSchemesManager(cls.model)
        _configure_svd_llm_v2(manager)
        manager.set_calibration_data(loader)
        manager.apply(hard=False, criteria="all", verbose=False,
                      device=cls.device, offload_to_cpu=False)

        cls.params_after = _param_count(cls.model)

    def test_compression_reduces_params(self):
        self.assertLess(self.params_after, self.params_before)

    def test_forward_pass(self):
        out = _run_forward(self.model, self.tokenizer, self.device)
        self.assertIsNotNone(out.logits)
        self.assertEqual(out.logits.dim(), 3)

    def test_logits_finite(self):
        out = _run_forward(self.model, self.tokenizer, self.device)
        self.assertTrue(torch.isfinite(out.logits).all(),
                        "Logits contain NaN or Inf after compression.")


# ---------------------------------------------------------------------------
# Test: AA-SVD with cascade calibration
# ---------------------------------------------------------------------------
class TestAASvdCascade(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = _device()
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        cls.model = _load_model(cls.device)
        cls.params_before = _param_count(cls.model)

        loader = _build_wikitext_calibration_loader(cls.tokenizer)
        manager = Qwen2CompressionSchemesManager(cls.model)
        _configure_aa_svd(manager)
        manager.set_calibration_mode(mode="cascade")
        manager.set_calibration_data(loader)
        manager.apply(hard=False, criteria="all", verbose=False,
                      device=cls.device, offload_to_cpu=False)

        cls.params_after = _param_count(cls.model)

    def test_compression_reduces_params(self):
        self.assertLess(self.params_after, self.params_before)

    def test_forward_pass(self):
        out = _run_forward(self.model, self.tokenizer, self.device)
        self.assertIsNotNone(out.logits)
        self.assertEqual(out.logits.dim(), 3)

    def test_logits_finite(self):
        out = _run_forward(self.model, self.tokenizer, self.device)
        self.assertTrue(torch.isfinite(out.logits).all(),
                        "Logits contain NaN or Inf after AA-SVD cascade compression.")


# ---------------------------------------------------------------------------
# Test: covariance equivalence — standard vs. cascade calibration
# ---------------------------------------------------------------------------
class TestCovarianceCascadeCompare(unittest.TestCase):
    """
    Checks that covariance summaries produced by the 'standard' and 'cascade'
    calibration modes are numerically close (max absolute error < atol).
    """

    ATOL = 1e-4

    @classmethod
    def _run_and_collect(cls, mode):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = _load_model(cls.device)
        loader = _build_wikitext_calibration_loader(tokenizer)

        dump_dir = Path(cls.tmp_dir) / mode
        dump_dir.mkdir(parents=True, exist_ok=True)

        manager = Qwen2CompressionSchemesManager(model)
        _configure_svd_llm_v2(manager)
        manager.set_calibration_mode(mode=mode)
        manager.set_calibration_data(loader)
        manager.apply(
            hard=False,
            criteria="all",
            verbose=False,
            device=cls.device,
            offload_to_cpu=False,
            summary_dump_dir=str(dump_dir),
            summary_dump_names=("covariance",),
        )
        return dump_dir

    @classmethod
    def _load_covariances(cls, root):
        cov_dir = root / "covariance"
        if not cov_dir.exists():
            raise RuntimeError(f"Covariance directory not found: {cov_dir}")
        result = {}
        for f in sorted(cov_dir.glob("*.pt")):
            payload = torch.load(f, map_location="cpu")
            result[payload["scheme_path"]] = payload["value"].float()
        if not result:
            raise RuntimeError(f"No covariance files found in {cov_dir}")
        return result

    @classmethod
    def setUpClass(cls):
        cls.device = _device()
        cls.tmp_dir = tempfile.mkdtemp(prefix="cov_compare_")

        standard_root = cls._run_and_collect("standard")
        cascade_root = cls._run_and_collect("cascade")

        cls.standard_map = cls._load_covariances(standard_root)
        cls.cascade_map = cls._load_covariances(cascade_root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_same_layer_keys(self):
        self.assertEqual(
            set(self.standard_map.keys()),
            set(self.cascade_map.keys()),
            "Standard and cascade produced different covariance layer sets.",
        )

    def test_same_shapes(self):
        for key in self.standard_map:
            with self.subTest(key=key):
                self.assertEqual(
                    self.standard_map[key].shape,
                    self.cascade_map[key].shape,
                    f"Shape mismatch for '{key}'.",
                )

    def test_covariance_values_close(self):
        max_err = 0.0
        worst = None
        for key in self.standard_map:
            err = float((self.standard_map[key] - self.cascade_map[key]).abs().max())
            if err > max_err:
                max_err, worst = err, key

        self.assertLessEqual(
            max_err, self.ATOL,
            f"Covariance mismatch too large: max_abs_error={max_err:.3e} > atol={self.ATOL:.3e} "
            f"(worst layer: {worst})",
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            GPU_NUM = int(sys.argv.pop(1))
        except ValueError:
            pass
    unittest.main(verbosity=2)
