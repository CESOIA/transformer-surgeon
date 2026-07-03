# Calibrated LRD (SVD-LLM-v2)

Loads Qwen2-0.5B, builds a small WikiText-2 calibration DataLoader, and applies
SVD-LLM-v2 — a calibration-aware variant of LRD that uses activation covariance
statistics to find a better low-rank factorization than plain SVD.

**Run it:**

```bash
python examples/02_calibrated_lrd_svd_llm_v2.py
python examples/02_calibrated_lrd_svd_llm_v2.py --lrd-rank 128 --num-calibration-samples 4
```

**Source** (`examples/02_calibrated_lrd_svd_llm_v2.py`):

```python
import argparse
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress

MLP_RANK_OVERRIDES = {"mlp.up_proj": 497, "mlp.gate_proj": 497, "mlp.down_proj": 497}


def _build_wikitext_calibration_loader(tokenizer, num_samples, seq_len):
    """Return a DataLoader of fixed-length token chunks from WikiText-2."""
    raw = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="train")
    texts = [t for t in raw["page"] if isinstance(t, str) and len(t) > 0]
    token_ids = tokenizer("\n\n".join(texts), truncation=False,
                          padding=False, return_attention_mask=False)["input_ids"]
    actual = min(num_samples, len(token_ids) // seq_len)
    rng = random.Random(42)
    examples = []
    for _ in range(actual):
        start = rng.randint(0, len(token_ids) - seq_len - 1)
        examples.append({
            "input_ids": torch.tensor(token_ids[start:start + seq_len], dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
        })
    return DataLoader(examples, batch_size=1, shuffle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--lrd-rank", type=int, default=64)
    parser.add_argument("--num-calibration-samples", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = Qwen2ForCausalLMCompress.from_pretrained(
        args.model_name, torch_dtype="auto"
    ).to(device).eval()

    params_before = sum(p.numel() for p in model.parameters())
    print(f"Parameters before: {params_before / 1e6:.2f} M")

    calibration_loader = _build_wikitext_calibration_loader(
        tokenizer, args.num_calibration_samples, args.seq_len
    )

    manager = Qwen2CompressionSchemesManager(model)
    manager.set("lrd", "method", "svd-llm-v2", criteria="all")
    for scheme in manager.iter_filtered(criteria="all"):
        if hasattr(scheme.get_module(), "weight"):
            manager.set("lrd", "rank", args.lrd_rank, criteria=scheme.path)
    for layer_key, rank in MLP_RANK_OVERRIDES.items():
        manager.set("lrd", "rank", rank, criteria=layer_key)

    manager.set_calibration_data(calibration_loader)
    manager.apply(hard=False, criteria="all", device=device, offload_to_cpu=True)

    params_after = sum(p.numel() for p in model.parameters())
    print(f"Parameters after:  {params_after / 1e6:.2f} M")

    inputs = tokenizer("The quick brown fox", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    print(f"Logits finite: {torch.isfinite(out.logits).all().item()}")


if __name__ == "__main__":
    main()
```
