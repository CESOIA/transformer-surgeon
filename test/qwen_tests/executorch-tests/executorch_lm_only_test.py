import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
)

from test_messages import messages
from qwen_vl_utils import process_vision_info
import sys

### TEST CONFIGURATION ###
PARTITIONER = XnnpackPartitioner()
# PARTITIONER = CoreMLPartitioner()
EXPORT_TO_EXECUTORCH = True

hard_mode = True
data_type = torch.float16  # Data type for model weights
use_vcon = False  # Whether to use VCON blocks
vcon_beta = 0.5  # Beta value for VCON blocks (between 0 and 1)
VERBOSE = True  # Whether to print detailed information during compression
DO_COMPRESSION = True  # Whether to apply compression
USE_STANDARD_MODEL = False  # Whether to use standard model instead of compressed version
##########################

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

if USE_STANDARD_MODEL:
    # Use standard model class instead of compressed version
    modelClass = AutoModel

# Device
# Get GPU number from command line arguments
gpu_num = 0  # Default GPU
if len(sys.argv) > 1:
    try:
        gpu_num = int(sys.argv[1])
    except ValueError:
        print(f"Invalid GPU number '{sys.argv[1]}', using default GPU 0.")

# device = torch.device(f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Load processor, model and tokenizer
processor = AutoProcessor.from_pretrained(model_name, torch_dtype=data_type)
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=data_type)
model = modelClass.from_pretrained(model_name, torch_dtype=data_type).to(device)

model.config.use_cache = False  # Disable cache for generation

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
            # ["visual", "mlp.up_proj"],                # Apply to all "mlp.up_proj" layers in vision_config
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

msg_index = 0
i, message = msg_index, messages[msg_index]

# Preparation of example inputs
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

# Definition of wrapper module for export - export only the language decoder

class EmbeddingWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

class LMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.language_model
        model.config._attn_implementation = "eager" # required for coreml export

    def forward(self,
                position_ids: torch.LongTensor,
                inputs_embeds: torch.Tensor,
                cache_position: torch.LongTensor,
                ):
        outputs = self.model(
            attention_mask={"full_attention": None},
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
        )
        return outputs.last_hidden_state

class FinalLayerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.lm_head

    def forward(self, hidden_states: torch.Tensor):
        logits = self.model(hidden_states[:, -1, :])
        return logits

# Instantiate the wrappers of all the blocks to be exported
wrapped_embedding = EmbeddingWrapper(model)
wrapped_model = LMWrapper(model)
wrapped_final_layer = FinalLayerWrapper(model)

# Prepare example inputs for each wrapper
batch_size = 1
seq_len = 30
max_seq_len = 1217
embed_dim = 3584
vocab_size = 152064
# embedding
example_inputs_embedding = (
    torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
)
# language decoder
example_inputs = (
    torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).to(device),
    torch.randn(batch_size, seq_len, embed_dim, dtype=data_type),
    torch.arange(seq_len, dtype=torch.long).to(device),
)
# final layer
example_inputs_final_layer = (
    torch.randn(batch_size, seq_len, embed_dim, dtype=data_type),
)

# Prepare dynamic shapes for the inputs of all the blocks
dyn_seq_len = Dim("seq_len", min=1, max=max_seq_len)
# embedding
dynamic_shapes_embedding = {
    "input_ids": (batch_size, dyn_seq_len),
}
# language decoder
dynamic_shapes = {
    "position_ids": (batch_size, dyn_seq_len),
    "inputs_embeds": (batch_size, dyn_seq_len, embed_dim),
    "cache_position": (dyn_seq_len,),
}
# final layer
dynamic_shapes_final_layer = {
    "hidden_states": (batch_size, dyn_seq_len, embed_dim),
}

# Test the wrappers in PyTorch - use the outputs of each block as inputs for the next one where suited
print("Test wrapped model inference in pytorch")
wrapped_embedding.eval()
wrapped_model.eval()
wrapped_final_layer.eval()
example_outputs_embedding = wrapped_embedding(*example_inputs_embedding)
# example_outputs = wrapped_model(example_inputs[0], example_outputs_embedding, example_inputs[2])
example_outputs = wrapped_model(*example_inputs)
# example_outputs_final_layer = wrapped_final_layer(example_outputs)
example_outputs_final_layer = wrapped_final_layer(*example_inputs_final_layer)

if EXPORT_TO_EXECUTORCH:
    def write_to_file(filename: str, data: bytes):
        from tqdm import tqdm
        chunk_size = 1024*1024  # 1 MB
        with open(filename, "wb") as file:
            for i in tqdm(range(0, len(data), chunk_size), desc=f"Writing {filename}"):
                file.write(data[i:i+chunk_size])
                file.flush()
                os.fsync(file.fileno()) # Ensure data is written to disk

    print("EXPORTING EMBEDDING LAYER")
    print("Exporting to graph format...")
    exported_program_embedding = export(wrapped_embedding, example_inputs_embedding, dynamic_shapes=dynamic_shapes_embedding)
    print("Exporting to Executorch format...")
    executorch_program_embedding = to_edge_transform_and_lower(
        exported_program_embedding,
        partitioner=[PARTITIONER],
    ).to_executorch()
    write_to_file("embedding.pte", executorch_program_embedding.buffer)

    print("EXPORTING LANGUAGE DECODER")
    print("Exporting to graph format...")
    exported_program = export(wrapped_model, example_inputs, dynamic_shapes=dynamic_shapes)
    print("Exporting to Executorch format...")
    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[PARTITIONER],
    ).to_executorch()
    write_to_file("lm_decoder.pte", executorch_program.buffer)

    print("EXPORTING FINAL LAYER")
    print("Exporting final layer to graph format...")
    exported_program_final_layer = export(wrapped_final_layer, example_inputs_final_layer, dynamic_shapes=dynamic_shapes_final_layer)
    print("Exporting final layer to Executorch format...")
    executorch_program_final_layer = to_edge_transform_and_lower(
        exported_program_final_layer,
        partitioner=[PARTITIONER],
    ).to_executorch()
    write_to_file("final_layer.pte", executorch_program_final_layer.buffer)

print("Loading and running Executorch program...")
from executorch.runtime import Runtime

runtime = Runtime.get()
# load program
program_embedding = runtime.load_program("embedding.pte")
program_lm_decoder = runtime.load_program("lm_decoder.pte")
program_final_layer = runtime.load_program("final_layer.pte")
# extract method
method_embedding = program_embedding.load_method("forward")
method_lm = program_lm_decoder.load_method("forward")
method_final_layer = program_final_layer.load_method("forward")
# run method
outputs_embedding = method_embedding.execute(list(example_inputs_embedding))
# outputs = method_lm.execute([example_inputs[0], outputs_embedding[0], example_inputs[2]])
outputs = method_lm.execute(list(example_inputs))
# outputs_final_layer = method_final_layer.execute(list(outputs))
outputs_final_layer = method_final_layer.execute(list(example_inputs_final_layer))
# here the next token is evaluated as follow
# temp = 1.0 (< 0 -> more greedy, > 1 -> more random)
# logits = torch.softmax(outputs[0] / temp, dim=-1)
# next_token = torch.multinomial(logits, num_samples=1)

if type(example_outputs_embedding) is tuple or type(example_outputs_embedding) is list:
    example_output_embedding = example_outputs_embedding[0]
if type(example_outputs) is tuple or type(example_outputs) is list:
    example_output = example_outputs[0]
if type(example_outputs_final_layer) is tuple or type(example_outputs_final_layer) is list:
    example_output_final_layer = example_outputs_final_layer[0]

print("Maximum error between original and executorch program (embedding):", torch.max(torch.abs(example_outputs_embedding - outputs_embedding[0])).item())
print("Maximum error between original and executorch program (lm):", torch.max(torch.abs(example_outputs - outputs[0])).item())
print("Maximum error between original and executorch program (final layer):", torch.max(torch.abs(example_outputs_final_layer - outputs_final_layer[0])).item())


print("Loading and running Executorch program with a different sequence length...")
# Test with a different sequence length
new_seq_len = 50
# Prepare new example inputs
# embedding
new_example_inputs_embedding = (
    torch.randint(0, vocab_size, (batch_size, new_seq_len), dtype=torch.long),
)
# language decoder
new_example_inputs = (
    torch.arange(0, new_seq_len, dtype=torch.long, device=device).unsqueeze(0).to(device),
    torch.randn(1, new_seq_len, embed_dim, dtype=data_type),
    torch.arange(new_seq_len, dtype=torch.long).to(device),
)
# final layer
new_example_inputs_final_layer = (
    torch.randn(1, new_seq_len, embed_dim, dtype=data_type),
)
# run method
new_outputs = method_embedding.execute(list(new_example_inputs_embedding))
new_outputs = method_lm.execute([new_example_inputs[0], new_outputs[0], new_example_inputs[2]])
new_outputs = method_final_layer.execute(list(new_outputs))
print("Test with different sequence length completed successfully.")