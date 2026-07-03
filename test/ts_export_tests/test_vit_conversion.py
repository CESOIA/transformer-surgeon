"""
Unit test: ViT conversion numerical equivalence after convert_for_export.

Run:
    python -m pytest test_vit_conversion.py -v
    python test_vit_conversion.py
"""
import unittest
import torch
from _common import VIT_MODEL, VIT_HIDDEN_ATOL, VIT_LOGIT_ATOL, get_device


class TestViTConversion(unittest.TestCase):
    """Exported ViT encoder must produce hidden states / logits close to HF's native encoder."""

    @classmethod
    def setUpClass(cls):
        from transformersurgeon import ViTForImageClassificationCompress
        from transformersurgeon.utils import convert_for_export

        cls.device = get_device()
        model = ViTForImageClassificationCompress.from_pretrained(VIT_MODEL)
        model.eval().to(cls.device)

        converted = convert_for_export(model, options={"use_sdpa": True}, verbose=False)
        encoder = converted["vit"].to(cls.device).eval()

        pixel_values = torch.randn(1, 3, 224, 224, device=cls.device)

        with torch.no_grad():
            embeddings = model.vit.embeddings(pixel_values)

            hf_hidden = model.vit.encoder(embeddings).last_hidden_state
            hf_hidden = model.vit.layernorm(hf_hidden)
            ts_hidden = encoder(embeddings)

            cls.hf_logits = model.classifier(hf_hidden[:, 0, :])
            cls.ts_logits = model.classifier(ts_hidden[:, 0, :])

        cls.hidden_diff = (hf_hidden - ts_hidden).abs()
        cls.logit_diff = (cls.hf_logits - cls.ts_logits).abs()

    def test_hidden_max_abs_diff(self):
        max_diff = float(self.hidden_diff.max())
        self.assertLessEqual(
            max_diff, VIT_HIDDEN_ATOL,
            f"Hidden state max abs diff {max_diff:.3e} exceeds atol {VIT_HIDDEN_ATOL:.3e}",
        )

    def test_hidden_mean_abs_diff(self):
        mean_diff = float(self.hidden_diff.mean())
        self.assertLessEqual(
            mean_diff, VIT_HIDDEN_ATOL,
            f"Hidden state mean abs diff {mean_diff:.3e} exceeds atol {VIT_HIDDEN_ATOL:.3e}",
        )

    def test_logits_max_abs_diff(self):
        max_diff = float(self.logit_diff.max())
        self.assertLessEqual(
            max_diff, VIT_LOGIT_ATOL,
            f"Logit max abs diff {max_diff:.3e} exceeds atol {VIT_LOGIT_ATOL:.3e}",
        )

    def test_logits_same_argmax(self):
        self.assertTrue(
            torch.equal(
                self.hf_logits.argmax(dim=-1),
                self.ts_logits.argmax(dim=-1),
            ),
            "HF and exported ViT predict different classes.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
