import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)
from test_messages import messages
from qwen_vl_utils import process_vision_info
import sys

### TEST CONFIGURATION ###
model_type = "qwen2_5_vl_c" 
hard_mode = False
use_vcon = False  # Whether to use VCON blocks
vcon_beta = 0.5  # Beta value for VCON blocks (between 0 and 1)
restore_topology = False  # Whether to restore the original model topology only
VERBOSE = False  # Whether to print detailed information during compression
DO_COMPRESSION = True  # Whether to apply compression
USE_ORIGINAL_MODEL = False  # Whether to load the original model without compression for comparison
##########################

if USE_ORIGINAL_MODEL:
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLConfig,
    )
    modelClass = Qwen2_5_VLForConditionalGeneration
    configClass = Qwen2_5_VLConfig
else:
    from transformersurgeon import (
        Qwen2_5_VLForConditionalGenerationCompress,
        Qwen2_5_VLConfigCompress,
        Qwen2_5_VLCompressionSchemesManager,
    )
    modelClass = Qwen2_5_VLForConditionalGenerationCompress
    configClass = Qwen2_5_VLConfigCompress
    managerClass = Qwen2_5_VLCompressionSchemesManager

# Model name
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor, model and tokenizer
# N.B.: using torch_dtype="auto" to automatically use bfloat16 if supported by the GPU (e.g., A100, H100)
# Not using it results in loading the model in float32, which requires more memory
processor = AutoProcessor.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
model = modelClass.from_pretrained(model_name, torch_dtype="auto").to(device)
print(model.config)

### COMPRESSION CONFIGURATION AND APPLICATION ###

# Example usage of CompressionSchemesManager
"""
    SchemesManager supports filtering layers using criteria such as:
    - Block ID (int)
    - Layer name or path (str, supports partial matching)
    - Lists of the above (OR logic within the list)
    - List inside the criteria list (AND logic between criteria)

    Examples of criteria:
    criteria = [3, ["mlp", "vision"]]  apply to all blocks with ID 3 OR containing "mlp" AND "vision" in their name/path
"""

filter = [
    ["visual", "mlp.up_proj"],                # Apply to all "mlp.up_proj" layers in vision_config
    ["visual", "mlp.down_proj", 0],           # Apply to the first "mlp.down_proj" layer in vision_config
    ["visual", "mlp.down_proj", 1],           # Apply to the second "mlp.down_proj" layer in vision_config
    ["language_model", "mlp.down_proj", 27],  # Apply to the last "mlp.down_proj" layer in text_config 
]

if not USE_ORIGINAL_MODEL:
    manager = managerClass(model)
    # manager.set("pruning", "mode", "unstructured", filter, verbose=VERBOSE)
    # manager.set("pruning", "criterion", "magnitude", filter, verbose=VERBOSE)
    # manager.set("pruning", "ratio", 0.5, filter, verbose=VERBOSE)
    # manager.set("quantization", "precision", "binary", filter, verbose=VERBOSE)
    # manager.set("quantization", "sparsity", 0.5, filter, verbose=VERBOSE)
    # manager.set("quantization", "sparse_criterion", "magnitude", filter, verbose=VERBOSE)
    # manager.set("quantization", "sparse_reverse", True, filter, verbose=VERBOSE)
    manager.set("lrd", "rank", 64, filter, verbose=VERBOSE)

    if use_vcon:
        manager.init_vcon(filter, verbose=VERBOSE)
        manager.set_vcon_beta(filter, vcon_beta)

    # Print the compression schemes manager
    # print(manager)

    # Apply all compression schemes to the model
    manager.apply(hard=hard_mode, verbose=VERBOSE)

# Print the model architecture
print(model)

### INFERENCE AND TESTING ###
print("Generating text with compressed model...")

for i, message in enumerate(messages):
    print(f"\nProcessing message {i + 1}...")
    print("Input:", message["content"][1]["text"])
    
    # Preparation for inference for this single message
    single_message = [message]
    text = processor.apply_chat_template(
        single_message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(single_message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output for this message
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    message_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Display result immediately
    print("Output:", message_output[0])
    print("-" * 40)

if USE_ORIGINAL_MODEL:
    exit() # If we are only testing the original model, we can exit here without restoring

# Cancel vcon
if use_vcon:
    manager.cancel_vcon()
# Restore the model to its original state and test again
manager.restore(topology=restore_topology)

print("Generating text with restored model...")

for i, message in enumerate(messages):
    print(f"\nProcessing message {i + 1}...")
    print("Input:", message["content"][1]["text"])
    
    # Preparation for inference for this single message
    single_message = [message]
    text = processor.apply_chat_template(
        single_message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(single_message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output for this message
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    message_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Display result immediately
    print("Output:", message_output[0])
    print("-" * 40)