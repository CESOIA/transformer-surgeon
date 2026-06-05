import torch
import json
import sys
from transformers import AutoTokenizer
from test_cars_qa_msgs import messages
from pathlib import Path
from typing import Dict
from torch.utils.data import DataLoader, Dataset
from transformersurgeon.hf import export_to_hf

### DATATEST BUILD ###
class JsonlMessagesDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def build_dialog_collate_fn(tokenizer, max_length=1024):
    def collate(samples):
        texts = []
        for sample in samples:
            messages = sample.get("messages", [])
            text = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    text.append(f"User: {content}")
                elif role == "assistant":
                    text.append(f"Assistant: {content}")
                else:
                    text.append(f"{role.capitalize()}: {content}")
            texts.append("\n".join(text))
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return collate

### TEST CONFIGURATION ###
hard_mode = True
use_vcon = True  # Whether to use VCON blocks
vcon_beta = 0.5  # Beta value for VCON blocks (between 0 and 1)
VERBOSE = True   # Whether to print detailed information during compression
DO_COMPRESSION = True  # Whether to apply compression
##########################

from transformersurgeon import (
    Qwen2ForCausalLMCompress,
    Qwen2ConfigCompress,
    Qwen2CompressionSchemesManager,
)

modelClass = Qwen2ForCausalLMCompress
configClass = Qwen2ConfigCompress
managerClass = Qwen2CompressionSchemesManager

# Model name
model_name = "Qwen/Qwen2-0.5B-Instruct"

# Local export configuration
default_export_root = Path(__file__).resolve().parent / "artifacts"
export_repo_id = "qwen2-0.5b-instruct-aa-svd"
export_root = Path(sys.argv[2]) if len(sys.argv) > 2 else default_export_root

# Device
# Get GPU number from command line arguments
gpu_num = 0  # Default GPU
if len(sys.argv) > 1:
    try:
        gpu_num = int(sys.argv[1])
    except ValueError:
        print(f"Invalid GPU number '{sys.argv[1]}', using default GPU 0.")

device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
# N.B.: using torch_dtype="auto" to automatically use bfloat16 if supported by the GPU (e.g., A100, H100)
# Not using it results in loading the model in float32, which requires more memory
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
model = modelClass.from_pretrained(model_name, torch_dtype="auto").to(device)
print(model)

### WSVD COMPRESSION CONFIGURATION AND APPLICATION ###
dataset = JsonlMessagesDataset("../../experiments/llm_export/qwen2_compressed/automotive_990_examples.jsonl")
collate_fn = build_dialog_collate_fn(tokenizer)
calibration_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
)

manager = Qwen2CompressionSchemesManager(model)
manager.set("lrd", "method", "aa-svd", verbose=True)
manager.set("lrd", "rank", 64, criteria=["attn"], verbose=True)
manager.set("lrd", "rank", 496, criteria=["mlp"], verbose=True)

manager.set_calibration_mode(
    mode="cascade",
)

manager.set_calibration_data(calibration_loader)

manager.apply(
    hard=False,
    criteria="all",
    verbose=True,
    device=device,
    offload_to_cpu=False,
)
##########################

### SAVE COMPRESSED MODEL LOCALLY ###
export_root.mkdir(parents=True, exist_ok=True)
saved_model_dir = export_to_hf(
    model=model,
    repo_id=export_repo_id,
    base_model=model_name,
    out_dir=str(export_root),
)
print(f"Saved compressed model to: {saved_model_dir}")
##########################

### COMPRESSION CONFIGURATION AND APPLICATION ###

# Example usage of CompressionSchemesManager
"""
    SchemesManager supports filtering layers using criteria such as:
    - Block ID (int)
    - Layer name or path (str, supports partial matching)
    - Lists of the above (OR logic within the list)
    - List inside the criteria list (AND logic between criteria)

    Examples of criteria:
    criteria = [3, ["mlp", "down_proj"]]  # apply to blocks with ID 3 OR containing "mlp" AND "down_proj"
"""

print("Model after compression")
print(model)

### INFERENCE AND TESTING ###
print("Generating text...")

for i, message in enumerate(messages):
    print(f"\nProcessing message {i + 1}...")
    print("Input:", message["content"])

    # Format as chat using apply_chat_template
    chat = [message]
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("Output:", output[0])
    print("-" * 40)

