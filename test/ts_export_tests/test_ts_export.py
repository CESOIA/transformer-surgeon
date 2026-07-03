"""
Unit tests for transformer-surgeon export utilities.

Covers:
  - HF-native generation (baseline decode loop with KV cache)
  - Exported-decoder generation (convert_for_export text path, pos_id interface)
  - ViT conversion numerical equivalence (hidden states and logits)

Run:
    python -m pytest test_ts_export.py -v
    python test_ts_export.py
"""
import os
import sys
import unittest

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
QWEN_MODEL = "Qwen/Qwen2-0.5B-Instruct"
VIT_MODEL = "google/vit-base-patch16-224"
MAX_NEW_TOKENS = 32          # short enough for fast tests
TEMPERATURE = 0.0            # greedy — deterministic

CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
    "<|im_start|>user\n{instruction}\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)
TEST_PROMPT = "Tell me one short fact about France."

VIT_HIDDEN_ATOL = 1e-4
VIT_LOGIT_ATOL = 1e-3


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _logits_to_id(logits, temperature=0.0):
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Test: HF-native generation (generation_hf_test.py)
# ---------------------------------------------------------------------------
class TestGenerationHF(unittest.TestCase):
    """Decode loop using model.model (HF backbone) with past_key_values."""

    @classmethod
    def setUpClass(cls):
        from transformers import Qwen2TokenizerFast
        from transformersurgeon import Qwen2ForCausalLMCompress

        cls.device = _device()
        cls.tokenizer = Qwen2TokenizerFast.from_pretrained(QWEN_MODEL)

        full_model = Qwen2ForCausalLMCompress.from_pretrained(QWEN_MODEL, dtype=torch.float16)
        cls.embedding = full_model.get_input_embeddings().to(cls.device, dtype=torch.float16)
        cls.decoder = full_model.model.to(cls.device, dtype=torch.float16)
        cls.lm_head = full_model.lm_head.to(cls.device, dtype=torch.float16)

        cls.output_ids, cls.generated_text = cls._generate()

    @classmethod
    def _generate(cls):
        prompt_text = CHAT_TEMPLATE.format(instruction=TEST_PROMPT)
        input_ids = cls.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].to(cls.device)
        inputs_embeds = cls.embedding(input_ids)

        # Prefill
        past_key_values = None
        for i in range(inputs_embeds.size(0)):
            out = cls.decoder(
                inputs_embeds=inputs_embeds[i:i + 1].unsqueeze(0),
                position_ids=torch.tensor([[i]], device=cls.device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
        logits = cls.lm_head(out.last_hidden_state[0, -1, :])
        next_id = _logits_to_id(logits)
        output_ids = torch.cat([input_ids, next_id], dim=0)
        next_embed = cls.embedding(next_id)

        # Decode
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS - 1):
                pos_id = torch.tensor([[output_ids.size(0) - 1]], device=cls.device)
                out = cls.decoder(
                    inputs_embeds=next_embed[-1:, :].unsqueeze(0),
                    position_ids=pos_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out.past_key_values
                logits = cls.lm_head(out.last_hidden_state[0, -1, :])
                next_id = _logits_to_id(logits)
                output_ids = torch.cat([output_ids, next_id], dim=0)
                next_embed = cls.embedding(next_id)
                if next_id.item() == cls.tokenizer.eos_token_id:
                    break

        text = cls.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return output_ids, text

    def test_generates_non_empty_text(self):
        self.assertGreater(len(self.generated_text.strip()), 0,
                           "HF generation produced empty output.")

    def test_output_length_bounded(self):
        prompt_len = len(self.tokenizer(
            CHAT_TEMPLATE.format(instruction=TEST_PROMPT))["input_ids"])
        generated_len = self.output_ids.size(0) - prompt_len
        self.assertGreater(generated_len, 0, "No tokens were generated.")
        self.assertLessEqual(generated_len, MAX_NEW_TOKENS + 1)

    def test_output_ids_are_integers(self):
        self.assertEqual(self.output_ids.dtype, torch.long)

    def test_no_nan_in_output(self):
        # Sanity: output_ids must be valid token IDs (non-negative)
        self.assertTrue((self.output_ids >= 0).all())


# ---------------------------------------------------------------------------
# Test: Exported-decoder generation (generation_exported_test.py)
# ---------------------------------------------------------------------------
class TestGenerationExported(unittest.TestCase):
    """Decode loop using the exported decoder from convert_for_export (pos_id interface)."""

    @classmethod
    def setUpClass(cls):
        from transformers import Qwen2TokenizerFast
        from transformersurgeon import Qwen2ForCausalLMCompress
        from transformersurgeon.utils import convert_for_export

        cls.device = _device()
        cls.tokenizer = Qwen2TokenizerFast.from_pretrained(QWEN_MODEL)

        full_model = Qwen2ForCausalLMCompress.from_pretrained(QWEN_MODEL, dtype=torch.float16)
        cls.embedding = full_model.get_input_embeddings().to(cls.device, dtype=torch.float16)
        cls.lm_head = full_model.lm_head.to(cls.device, dtype=torch.float16)

        converted = convert_for_export(
            full_model,
            options={"use_sdpa": True, "max_cache_len": MAX_NEW_TOKENS + 64},
            verbose=False,
        )
        cls.decoder = converted["text"].to(cls.device, dtype=torch.float16)

        cls.output_ids, cls.generated_text = cls._generate()

    @classmethod
    def _generate(cls):
        prompt_text = CHAT_TEMPLATE.format(instruction=TEST_PROMPT)
        input_ids = cls.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].to(cls.device)
        inputs_embeds = cls.embedding(input_ids)

        # Prefill (iterative, one token at a time — matches the original script)
        for i in range(inputs_embeds.size(0)):
            output_embeds = cls.decoder(
                inputs_embeds[i:i + 1],
                pos_id=torch.tensor([i]),
            )
        logits = cls.lm_head(output_embeds[-1])
        next_id = _logits_to_id(logits)
        output_ids = torch.cat([input_ids, next_id], dim=0)
        next_embed = cls.embedding(next_id)

        # Decode
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS - 1):
                pos_id = torch.tensor([output_ids.size(0) - 1])
                output_embeds = cls.decoder(next_embed[-1:, :], pos_id=pos_id)
                logits = cls.lm_head(output_embeds[-1, :])
                next_id = _logits_to_id(logits)
                output_ids = torch.cat([output_ids, next_id], dim=0)
                next_embed = cls.embedding(next_id)
                if next_id.item() == cls.tokenizer.eos_token_id:
                    break

        text = cls.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return output_ids, text

    def test_generates_non_empty_text(self):
        self.assertGreater(len(self.generated_text.strip()), 0,
                           "Exported-decoder generation produced empty output.")

    def test_output_length_bounded(self):
        prompt_len = len(self.tokenizer(
            CHAT_TEMPLATE.format(instruction=TEST_PROMPT))["input_ids"])
        generated_len = self.output_ids.size(0) - prompt_len
        self.assertGreater(generated_len, 0, "No tokens were generated.")
        self.assertLessEqual(generated_len, MAX_NEW_TOKENS + 1)

    def test_output_ids_are_integers(self):
        self.assertEqual(self.output_ids.dtype, torch.long)

    def test_no_negative_ids(self):
        self.assertTrue((self.output_ids >= 0).all())


# ---------------------------------------------------------------------------
# Test: ViT conversion numerical equivalence (vit_conversion_inference_test.py)
# ---------------------------------------------------------------------------
class TestViTConversion(unittest.TestCase):
    """Exported ViT encoder must produce hidden states / logits close to HF's native encoder."""

    @classmethod
    def setUpClass(cls):
        from transformersurgeon import ViTForImageClassificationCompress
        from transformersurgeon.utils import convert_for_export

        cls.device = _device()
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


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
