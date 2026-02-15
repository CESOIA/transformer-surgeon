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
hard_mode = True
use_vcon = True  # Whether to use VCON blocks
vcon_beta = 0.5  # Beta value for VCON blocks (between 0 and 1)
VERBOSE = True  # Whether to print detailed information during compression
DO_COMPRESSION = True  # Whether to apply compression
##########################

if model_type == "qwen2_vl_c":
    from transformersurgeon import (
        Qwen2VLForConditionalGenerationCompress,
        Qwen2VLConfigCompress,
        Qwen2VLCompressionSchemesManager,
    )

    modelClass = Qwen2VLForConditionalGenerationCompress
    configClass = Qwen2VLConfigCompress
    managerClass = Qwen2VLCompressionSchemesManager

    # Model name
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
elif model_type == "qwen2_5_vl_c":
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
else:
    raise ValueError(f"Unsupported model_type '{model_type}'")

# Device
# Get GPU number from command line arguments
gpu_num = 0  # Default GPU
if len(sys.argv) > 1:
    try:
        gpu_num = int(sys.argv[1])
    except ValueError:
        print(f"Invalid GPU number '{sys.argv[1]}', using default GPU 0.")

device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")

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

print("Number of parameters before compression:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

if DO_COMPRESSION:
    manager = managerClass(model)
    manager.set_lrd_rank(512,
        [
            ["visual", "mlp.up_proj"],                # Apply to all "mlp.up_proj" layers in vision_config
            ["visual", "mlp.down_proj", 0],           # Apply to the first "mlp.down_proj" layer in vision_config
            ["visual", "mlp.down_proj", 1],           # Apply to the second "mlp.down_proj" layer in vision_config
            ["language_model", "mlp.down_proj", 27],  # Apply to the last "mlp.down_proj" layer in text_config
        ], verbose=VERBOSE)
    # manager.set_lrd_rank_all(32)

    if use_vcon:
        manager.init_vcon_all(verbose=VERBOSE)
        # manager.init_vcon(criteria=[3, "mlp"], verbose=VERBOSE)  # Initialize VCON for only specific layers
        manager.set_vcon_beta_all(vcon_beta)
        # manager.set_vcon_beta(vcon_beta, criteria=[3, "mlp"])  # Set beta for only specific layers

    # Optionally print the full compression configuration as a table
    # print(manager)

    # Apply all compression schemes to the model
    manager.apply_all(hard=hard_mode, verbose=VERBOSE)

    # Update in-place the compressed model configuration from the manager
    compress_config = manager.update_config()

    # Optionally print the model architecture
    # print(model)

print("Number of parameters after compression:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

# After applying compression with soft mode, you can revert the model to its original state if needed
# manager.restore_all()  # Uncomment this line if you want to restore the model to its original state

### INFERENCE AND TESTING ###
print("Generating text...")

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