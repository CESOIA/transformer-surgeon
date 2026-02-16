import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModel,  # Added to test AutoModel loading (with trust_remote_code=True)
)
from test.qwen_tests.test_messages import messages
from qwen_vl_utils import process_vision_info
import sys

### TEST CONFIGURATION ###
model_type = "qwen2_5_vl_c" 
hard_mode = True
use_vcon = False  # Whether to use VCON blocks
vcon_beta = 0.5  # Beta value for VCON blocks (between 0 and 1)
VERBOSE = True  # Whether to print detailed information during compression
USE_AUTOMODEL = True  # Whether to load the model with AutoModel (trust_remote_code=True) or with the custom class directly
LOAD_REMOTE = False  # Whether to load the model from the Hugging Face Hub (True) or from a local path (False)
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
if LOAD_REMOTE:
    model_name = "prolucio/Qwen2.5-VL-compress-custom"  # Hugging Face Hub repo with the exported compressed model
else:
    model_name = "./models/Qwen2.5-VL-compress-custom"  # Local path with the exported compressed model

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
if not USE_AUTOMODEL:
    model = modelClass.from_pretrained(model_name, torch_dtype="auto").to(device)
else:
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto").to(device)

### COMPRESSION AND HUGGING FACE EXPORT TEST ###

def sizeof_dtype(dtype):
    if dtype == torch.float32:
        return 4
    elif dtype == torch.float16:
        return 2
    elif dtype == torch.bfloat16:
        return 2
    elif dtype == torch.int8:
        return 1
    elif dtype == torch.int4:
        return 0.5
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

print("Number of parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
print("Estimated size on disk", sum(p.numel()*sizeof_dtype(p.dtype) for p in model.parameters()) / 2**30, "GB")
print("Number of parameters of the vision tower:", sum(p.numel() for n, p in model.named_parameters() if "visual" in n) / 1e6, "M")
print("Number of parameters of the text tower:", sum(p.numel() for n, p in model.named_parameters() if "language_model" in n) / 1e6, "M")
print("Number of parameters of the output layers:", sum(p.numel() for n, p in model.named_parameters() if "lm_head" in n) / 1e6, "M")

# Check model compression configuration
manager = managerClass(model)
print(manager)

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