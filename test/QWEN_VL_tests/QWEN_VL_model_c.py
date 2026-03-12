import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)
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

model_name = "/ibex/user/antonic/CESOIA/transformer-surgeon/qwen-vl-finetune/checkpoints"

model = modelClass.from_pretrained(model_name, torch_dtype="auto").to(device)

# Load processor, model and tokenizer
# N.B.: using torch_dtype="auto" to automatically use bfloat16 if supported by the GPU (e.g., A100, H100)
# Not using it results in loading the model in float32, which requires more memory
processor = AutoProcessor.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")

model = modelClass.from_pretrained(model_name, torch_dtype="auto").to(device)
print(model.config)

print(model)