import torch
import sys
from transformers import AutoTokenizer
from test_cars_qa_msgs import messages

### TEST CONFIGURATION ###
hard_mode = True
use_vcon = True  # Whether to use VCON blocks
vcon_beta = 0.5  # Beta value for VCON blocks (between 0 and 1)
VERBOSE = True   # Whether to print detailed information during compression
DO_COMPRESSION = False  # Keep model uncompressed
##########################

from transformersurgeon import (
    LlamaForCausalLMCompress,
    LlamaConfigCompress,
    LlamaCompressionSchemesManager,
)

modelClass = LlamaForCausalLMCompress
configClass = LlamaConfigCompress
managerClass = LlamaCompressionSchemesManager

# Model name
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Device
# Get GPU number from command line arguments
gpu_num = 0  # Default GPU
if len(sys.argv) > 1:
    try:
        gpu_num = int(sys.argv[1])
    except ValueError:
        print(f"Invalid GPU number '{sys.argv[1]}', using default GPU 0.")

device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model using tsurgeon model classes, without applying compression
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
model = modelClass.from_pretrained(model_name, torch_dtype="auto").to(device)
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

    # Inference: generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("Output:", output[0])
    print("-" * 40)
