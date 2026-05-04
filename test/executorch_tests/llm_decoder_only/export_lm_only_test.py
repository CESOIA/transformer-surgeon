import torch
import os
from copy import deepcopy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_AVAILABLE_THREADS = torch.get_num_threads()
print(f"Available CPU threads: {MAX_AVAILABLE_THREADS}")
torch.set_num_threads(min(MAX_AVAILABLE_THREADS, 32))  # Limit to 4 threads for better performance in some cases
print(f"Using {torch.get_num_threads()} CPU threads for PyTorch operations")

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import Dim, export, draft_export

from transformers import Qwen2TokenizerFast

from transformersurgeon import (
    # Qwen2_5_VLForConditionalGenerationCompress,
    # Qwen2_5_VLConfigCompress,
    # Qwen2_5_VLCompressionSchemesManager,
    Qwen2ForCausalLMCompress,
    Qwen2ConfigCompress,
    Qwen2CompressionSchemesManager,
    convert_for_export,
)

### DEFAULT TEST CONFIGURATION ###
EXPORT_TO_EXECUTORCH = False
EXPORT_EMBEDDING, EXPORT_LM_DECODER, EXPORT_FINAL_LAYER = False, False, False
CHECK_IR_VALIDITY = False

### TEST CONFIGURATION ###
PARTITIONER = XnnpackPartitioner()
EXPORT_TO_EXECUTORCH = True
# EXPORT_EMBEDDING = True
EXPORT_LM_DECODER = True
# EXPORT_FINAL_LAYER = True
CHECK_IR_VALIDITY = True

data_type = torch.float32  # Data type for model weights
device = torch.device("cpu")
VERBOSE = False  # Whether to print detailed information during compression

# modelClass = Qwen2_5_VLForConditionalGenerationCompress
# configClass = Qwen2_5_VLConfigCompress
# managerClass = Qwen2_5_VLCompressionSchemesManager
modelClass = Qwen2ForCausalLMCompress
configClass = Qwen2ConfigCompress
managerClass = Qwen2CompressionSchemesManager

# Model name
# model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model_name = "Qwen/Qwen2-0.5B-Instruct"
max_seq_len = 2048

# Load processor, model and tokenizer
tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
model = modelClass.from_pretrained(model_name, torch_dtype="auto").to(device)

# Extract embedding layer (token id to features vector)
embedding = model.get_input_embeddings()

# Extract final layer (last feature to token logits)
final_layer = model.lm_head

# Convert model to export-compatible modules
convert_options = {
    "use_sdpa": False,  # Whether to use SDPA or regular MHA in the decoder wrapper
}
converted_models = convert_for_export(model, options=convert_options, verbose=VERBOSE)
decoder = converted_models['text']

# Put all modules on the same device and data type
embedding = embedding.to(device, dtype=data_type)
decoder = decoder.to(device, dtype=data_type)
final_layer = final_layer.to(device, dtype=data_type)

### COMPRESSION CONFIGURATION AND APPLICATION ###
#####################################
### This will be integrated later ###
#####################################

# Definition of wrapper modules for export

class EmbeddingWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.module(input_ids)

class DecoderWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self,
               inputs_embeds: torch.Tensor,
               cache_len_tensor: torch.Tensor,
               ):
        cache_len = cache_len_tensor.size(0)
        output_embed = self.module(
            inputs_embeds,
            cache_len=cache_len,
            )
        return output_embed

class FinalLayerWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, hidden_states: torch.Tensor):
        logits = self.module(hidden_states[:, -1, :])
        return logits

# Instantiate the wrappers of all the blocks to be exported
wrapped_embedding = EmbeddingWrapper(embedding)
wrapped_decoder = DecoderWrapper(decoder)
wrapped_final_layer = FinalLayerWrapper(final_layer)

# Prepare example inputs for each wrapper
batch_size = 1
seq_len = 2
cache_len = 5
if hasattr(model.config, "text_config"):
    embed_dim = model.config.text_config.hidden_size
else:
    embed_dim = model.config.hidden_size
vocab_size = 152064

# embedding
example_inputs_embedding = (
    torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long).to(device),
)
# language decoder
example_inputs_decoder = (
    torch.randn(batch_size, seq_len, embed_dim, dtype=data_type).to(device), # inputs_embeds
    torch.ones(cache_len), # cache_len_tensor
)
# final layer
example_inputs_final_layer = (
    torch.randn(batch_size, seq_len, embed_dim, dtype=data_type).to(device),
)

# Store example inputs on disk for testing the executorch program
torch.save(example_inputs_embedding, "example_inputs_embedding.pt")
torch.save(example_inputs_decoder, "example_inputs_decoder.pt")
torch.save(example_inputs_final_layer, "example_inputs_final_layer.pt")

# Prepare dynamic shapes for the inputs of all the blocks
dyn_seq_len = Dim("dyn_seq_len", min=1, max=max_seq_len-1)
dyn_cache_len = Dim("dyn_cache_len", min=1, max=max_seq_len-1)

# embedding
dynamic_shapes_embedding = {
    "input_ids": (Dim.STATIC, dyn_seq_len),
}

# language decoder
dynamic_shapes = {
    "inputs_embeds": (Dim.STATIC, dyn_seq_len, Dim.STATIC),
    "cache_len_tensor": (dyn_cache_len,),
}

# final layer
dynamic_shapes_final_layer = {
    "hidden_states": (Dim.STATIC, dyn_seq_len, Dim.STATIC),
}

# Test the wrappers in PyTorch and store the outputs for reference
print("Test wrapped model inference in pytorch and store outputs for reference...")
wrapped_embedding.eval()
wrapped_decoder.eval()
wrapped_final_layer.eval()
example_outputs_embedding = wrapped_embedding(*example_inputs_embedding)
example_outputs = wrapped_decoder(*example_inputs_decoder)
example_outputs_final_layer = wrapped_final_layer(*example_inputs_final_layer)
torch.save(example_outputs_embedding, "example_outputs_embedding.pt")
torch.save(example_outputs, "example_outputs_decoder.pt")
torch.save(example_outputs_final_layer, "example_outputs_final_layer.pt")

if EXPORT_TO_EXECUTORCH:
    def write_to_file(filename: str, data: bytes):
        from tqdm import tqdm
        chunk_size = 1024*1024*10  # 10 MB
        with open(filename, "wb") as file:
            for i in tqdm(range(0, len(data), chunk_size), desc=f"Writing {filename}"):
                file.write(data[i:i+chunk_size])
                file.flush()
                os.fsync(file.fileno()) # Ensure data is written to disk

    if EXPORT_EMBEDDING:
        print("EXPORTING EMBEDDING LAYER")
        print("Exporting to ATen graph...")
        # exported_module = export(wrapped_embedding, example_inputs_embedding, dynamic_shapes=dynamic_shapes_embedding)
        exported_module = draft_export(wrapped_embedding, example_inputs_embedding)
        print("Lowering to edge model...")
        edge_module = to_edge_transform_and_lower(
            exported_module,
            compile_config=EdgeCompileConfig(_check_ir_validity=CHECK_IR_VALIDITY),
            partitioner=[PARTITIONER],
        )
        print("Exporting to Executorch format...")
        executorch_module = edge_module.to_executorch()
        print("Writing to file...")
        write_to_file("embedding.pte", executorch_module.buffer)

    if EXPORT_LM_DECODER:
        print("EXPORTING LANGUAGE DECODER")
        print("Exporting to ATen graph...")
        exported_module = export(wrapped_decoder, example_inputs_decoder, dynamic_shapes=dynamic_shapes)
        print("Lowering to edge model...")
        edge_module = to_edge_transform_and_lower(
            exported_module,
            compile_config=EdgeCompileConfig(_check_ir_validity=CHECK_IR_VALIDITY),
            partitioner=[PARTITIONER],
        )
        open("lm_decoder.log", "w").write(edge_module.exported_program().graph_module.print_readable(print_output=False))
        print("Exporting to Executorch format...")
        executorch_module = edge_module.to_executorch()
        print("Writing to file...")
        write_to_file("lm_decoder.pte", executorch_module.buffer)

    if EXPORT_FINAL_LAYER:
        print("EXPORTING FINAL LAYER")
        print("Exporting final layer to graph format...")
        # exported_program_final_layer = export(wrapped_final_layer, example_inputs_final_layer, dynamic_shapes=dynamic_shapes_final_layer)
        exported_program_final_layer = draft_export(wrapped_final_layer, example_inputs_final_layer)
        print("Exporting final layer to Executorch format...")
        executorch_program_final_layer = to_edge_transform_and_lower(
            exported_program_final_layer,
            compile_config=EdgeCompileConfig(_check_ir_validity=CHECK_IR_VALIDITY),
            partitioner=[PARTITIONER],
        ).to_executorch()
        write_to_file("final_layer.pte", executorch_program_final_layer.buffer)