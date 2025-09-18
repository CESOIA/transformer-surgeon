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
model_type = "qwen2_vl_c" 
hard_mode = False
use_vcon = True  # Whether to use VCON blocks
vcon_beta = 0.5  # Beta value for VCON blocks (between 0 and 1)
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
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = modelClass.from_pretrained(model_name).to(device)

# Example usage of CompressionSchemesManager
config_dict = model.config.to_dict()
config_dict["vision_config"]["lrd_rank_lists"]["mlp_up"] = 512
# config_dict["vision_config"]["lrd_rank_lists"]["mlp_up"] = [512, 256] * 16 # you can define lrd rank for each block
config_dict["text_config"]["lrd_rank_lists"]["mlp_up"] = "full"

# Apply updated configuration to the model and update dict
compress_config = configClass.from_dict(config_dict)

# Initialize the CompressionSchemesManager with the model and configuration
manager = managerClass(compress_config.to_dict(), model)

print(manager) # print the full compression configuration

if use_vcon:
    # Initialize VCON blocks for all modules
    # manager.init_vcon_all(verbose=True)
    manager.init_vcon(criteria=[3, "mlp"], verbose=True)  # Initialize VCON for specific layers

    # Set the beta value for all VCON blocks
    manager.set_vcon_beta_all(vcon_beta)

# Apply all compression schemes to the model (soft mode)
manager.apply_all(hard=hard_mode, verbose=True)

if hard_mode:
    # Apply configuration to the model - needed for hard mode
    model.config = compress_config

print(model) # print the model architecture

# After applying compression with soft mode, you can revert the model to its original state if needed
# manager.restore_all()  # Uncomment this line if you want to restore the model to its original state

# Process each message separately
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