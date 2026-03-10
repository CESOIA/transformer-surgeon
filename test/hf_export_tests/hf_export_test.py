import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)
import sys
sys.path.append("../test_data")  # Add the path to the parent directory to import from it
from test_messages import messages  # Import the messages defined in test_messages.py

### TEST CONFIGURATION ###
model_type = "qwen2_5_vl_c" 
hard_mode = True
VERBOSE = True  # Whether to print detailed information during compression
DO_COMPRESSION = True  # Whether to apply compression
DO_EXPORT = True  # Whether to export the model to Hugging Face Hub
LOCAL_ONLY = True  # Whether to avoid any interaction with Hugging Face Hub (e.g., for testing purposes without credentials)
REPO_ID = "prolucio/Qwen2.5-VL-compress-custom"  # Repository ID for Hugging Face Hub (e.g., "username/repo_name")
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
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
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
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

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
processor = AutoProcessor.from_pretrained(model_name, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, dtype="auto")
model = modelClass.from_pretrained(model_name, dtype="auto").to(device)


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

print("Number of parameters before compression:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
print("Estimated size on disk", sum(p.numel()*sizeof_dtype(p.dtype) for p in model.parameters()) / 2**30, "GB")
print("Number of parameters of the vision tower:", sum(p.numel() for n, p in model.named_parameters() if "visual" in n) / 1e6, "M")
print("Number of parameters of the text tower:", sum(p.numel() for n, p in model.named_parameters() if "language_model" in n) / 1e6, "M")
print("Number of parameters of the output layers:", sum(p.numel() for n, p in model.named_parameters() if "lm_head" in n) / 1e6, "M")

if DO_COMPRESSION:
    manager = managerClass(model)
    # manager.set("lrd", "rank", 128,
    #     [
    #         ["visual", "mlp.up_proj"],                # Apply to all "mlp.up_proj" layers in vision_config
    #         ["visual", "mlp.down_proj", 0],           # Apply to the first "mlp.down_proj" layer in vision_config
    #         ["visual", "mlp.down_proj", 1],           # Apply to the second "mlp.down_proj" layer in vision_config
    #         ["language_model", "mlp.down_proj", 27],  # Apply to the last "mlp.down_proj" layer in text_config
    #     ], verbose=VERBOSE)
    manager.set("lrd", "rank", 128, [["visual", "mlp.up_proj", 2]])
    manager.set("lrd", "rank", 128, [
        ["language_model", "mlp.down_proj", 26],
        ["language_model", "mlp.down_proj", 27]
        ])

    # Optionally print the full compression configuration as a table
    # print(manager)  

    # Apply all compression schemes to the model
    manager.apply(hard=hard_mode, verbose=VERBOSE)

    # Update in-place the compressed model configuration from the manager
    compress_config = manager.update_config(verbose=VERBOSE)

    print(model.config)

    # Optionally print the model architecture
    # print(model)

if DO_EXPORT:
    # Export model to hugging face
    from transformersurgeon.export import export_to_hf
    print("Exporting model to Hugging Face...")
    token = None
    repo_id = REPO_ID
    if not LOCAL_ONLY:
        with open("./hf_token.txt") as f: token = f.read().strip()
    export_to_hf(
        model,
        repo_id=repo_id,
        base_model=model_name,
        out_dir="models",
        readme="This is a compressed version of Qwen2.5-VL-7B-Instruct using custom compression schemes.",
        embed_code=True,
        token=token,
        private=True,
        exist_ok=True,
    )
    print("Model exported.")

print("Number of parameters after compression:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
print("Estimated size on disk", sum(p.numel()*sizeof_dtype(p.dtype) for p in model.parameters()) / 2**30, "GB")