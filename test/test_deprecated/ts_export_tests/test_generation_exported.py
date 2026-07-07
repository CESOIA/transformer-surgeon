"""
Unit test: exported-decoder decode loop via convert_for_export (pos_id interface).

Run:
    python -m pytest test_generation_exported.py -v
    python test_generation_exported.py
"""
import unittest
import torch
from _common import (
    QWEN_MODEL, MAX_NEW_TOKENS, CHAT_TEMPLATE, TEST_PROMPT,
    get_device, logits_to_id,
)


class TestGenerationExported(unittest.TestCase):
    """Decode loop using the exported decoder from convert_for_export (pos_id interface)."""

    @classmethod
    def setUpClass(cls):
        from transformers import Qwen2TokenizerFast
        from transformersurgeon import Qwen2ForCausalLMCompress
        from transformersurgeon.utils import convert_for_export

        cls.device = get_device()
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

        for i in range(inputs_embeds.size(0)):
            output_embeds = cls.decoder(
                inputs_embeds[i:i + 1],
                pos_id=torch.tensor([i]),
            )
        logits = cls.lm_head(output_embeds[-1])
        next_id = logits_to_id(logits)
        output_ids = torch.cat([input_ids, next_id], dim=0)
        next_embed = cls.embedding(next_id)

        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS - 1):
                pos_id = torch.tensor([output_ids.size(0) - 1])
                output_embeds = cls.decoder(next_embed[-1:, :], pos_id=pos_id)
                logits = cls.lm_head(output_embeds[-1, :])
                next_id = logits_to_id(logits)
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
        prompt_len = len(self.tokenizer(CHAT_TEMPLATE.format(instruction=TEST_PROMPT))["input_ids"])
        generated_len = self.output_ids.size(0) - prompt_len
        self.assertGreater(generated_len, 0, "No tokens were generated.")
        self.assertLessEqual(generated_len, MAX_NEW_TOKENS + 1)

    def test_output_ids_are_integers(self):
        self.assertEqual(self.output_ids.dtype, torch.long)

    def test_no_negative_ids(self):
        self.assertTrue((self.output_ids >= 0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
