import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM
from test_messages import messages
import sys

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
print(model.config)
print(model)

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
