# Basic LRD

Loads Qwen2-0.5B-Instruct, applies reversible low-rank decomposition to two
specific projection layers using the HuggingFace manager, runs a generation,
then restores the model to its original weights.

**Run it:**

```bash
python examples/01_basic_lrd.py
python examples/01_basic_lrd.py --model-name Qwen/Qwen2-1.5B-Instruct --lrd-rank 320
```

**Source** (`examples/01_basic_lrd.py`):

```python
import argparse

import torch
from transformers import AutoTokenizer

from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress


def parse_args():
    parser = argparse.ArgumentParser(description="Basic LRD example")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--prompt", type=str, default="Write one short sentence about model compression.")
    parser.add_argument("--lrd-rank", type=int, default=640)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = Qwen2ForCausalLMCompress.from_pretrained(
        args.model_name, torch_dtype="auto"
    ).to(device).eval()

    params_before = param_count(model)
    print(f"Parameters before compression: {params_before / 1e6:.2f} M")

    manager = Qwen2CompressionSchemesManager(model)
    manager.set("lrd", "rank", args.lrd_rank, criteria=[[0, "self_attn.q_proj"]])
    manager.set("lrd", "rank", args.lrd_rank, criteria=[[1, "mlp.up_proj"]])
    manager.apply(hard=False)   # hard=False keeps compression reversible

    params_after = param_count(model)
    print(f"Parameters after  compression: {params_after / 1e6:.2f} M")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print(f"Generated: {generated}")

    manager.restore()
    print(f"Parameters after restore: {param_count(model) / 1e6:.2f} M")


if __name__ == "__main__":
    main()
```
