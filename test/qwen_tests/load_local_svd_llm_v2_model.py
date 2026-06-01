import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from transformersurgeon import Qwen2ForCausalLMCompress


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_model_dir = (
        script_dir
        / "artifacts"
        / "qwen2-0.5b-instruct-svd-llm-v2-local"
    )
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_model_dir

    if not model_dir.exists():
        raise FileNotFoundError(f"Local model directory not found: {model_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Loading compressed model from: {model_dir}")
    model = Qwen2ForCausalLMCompress.from_pretrained(model_dir, torch_dtype="auto").to(device)
    model.eval()

    prompt = "List 3 safety checks before buying a used car."
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("Reload generation output:")
    print(output[0])


if __name__ == "__main__":
    main()
