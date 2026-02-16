import torch
import os
from copy import deepcopy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import Dim, export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer
)
from executorch.backends.apple.coreml.quantizer.coreml_quantizer import (
    CoreMLQuantizer,
)

from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
)
from transformersurgeon import (
    Qwen2_5_VLForConditionalGenerationCompress,
    Qwen2_5_VLConfigCompress,
    Qwen2_5_VLCompressionSchemesManager,
)

from transformersurgeon.utils import convert_for_export
from transformersurgeon.blocks.rope import precompute_rope_inv_freqs

from test_messages import messages
from qwen_vl_utils import process_vision_info
import sys

### TEST CONFIGURATION ###
PARTITIONER = XnnpackPartitioner()
# PARTITIONER = CoreMLPartitioner()
QUANTIZER = XNNPACKQuantizer
EXPORT_TO_EXECUTORCH = True
EXPORT_EMBEDDING = True
EXPORT_LM_DECODER = True
EXPORT_FINAL_LAYER = True

hard_mode = True
data_type = torch.float16  # Data type for model weights
VERBOSE = True  # Whether to print detailed information during compression
DO_COMPRESSION = False  # Whether to apply compression
USE_STANDARD_MODEL = False  # Whether to use standard model instead of compressed version

Q_GROUP_SIZE = 32  # Quantization group size for weight-only quantization
##########################


modelClass = Qwen2_5_VLForConditionalGenerationCompress
configClass = Qwen2_5_VLConfigCompress
managerClass = Qwen2_5_VLCompressionSchemesManager

# Quantizer function

def quantize(model, example_inputs):
    # model.print_readable()
    quantizer = QUANTIZER()
    qparams = get_symmetric_quantization_config(is_per_channel=True)
    quantizer.set_global(qparams)
    m = prepare_pt2e(model, quantizer)
    # calibration
    m(*example_inputs)
    m = convert_pt2e(m)
    # m.print_readable()
    return m

# Model name
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

device = torch.device("cpu")

# Load processor, model and tokenizer
processor = AutoProcessor.from_pretrained(model_name, torch_dtype=data_type)
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=data_type)
model = modelClass.from_pretrained(model_name, torch_dtype=data_type).to(device)

### COMPRESSION CONFIGURATION AND APPLICATION ###

if DO_COMPRESSION:
    manager = managerClass(model)
    manager.set_lrd_rank(512,
        [
            # ["visual", "mlp.up_proj"],              # Apply to all "mlp.up_proj" layers in vision_config
            ["visual", "mlp.down_proj", 0],           # Apply to the first "mlp.down_proj" layer in vision_config
            ["visual", "mlp.down_proj", 1],           # Apply to the second "mlp.down_proj" layer in vision_config
            ["language_model", "mlp.down_proj", 27],  # Apply to the last "mlp.down_proj" layer in text_config
        ], verbose=VERBOSE)
    # manager.set_lrd_rank_all(32)

    # Optionally print the full compression configuration as a table
    # print(manager)

    # Apply all compression schemes to the model
    manager.apply_all(hard=hard_mode, verbose=VERBOSE)

    # Update in-place the compressed model configuration from the manager
    compress_config = manager.update_config()

    # Optionally print the model architecture
    # print(model)

# Convert to export-compatible modules
converted_models = convert_for_export(model, verbose=False)
decoder = converted_models['text']
decoder = decoder.to(device, dtype=data_type)

# Preparation of example inputs
msg_index = 0
i, message = msg_index, messages[msg_index]

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

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.model(input_ids)

class LMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.inv_freqs = precompute_rope_inv_freqs(
            head_dim=128,
            base=1e6,
            device=device,
            ).to(data_type)
    
    def forward(self,
               inputs_embeds: torch.Tensor,
               key_cache: torch.Tensor,
               value_cache: torch.Tensor,
               ):
        output_embed, key_cache, value_cache = self.model(
            inputs_embeds,
            inv_freq=self.inv_freqs,
            key_cache=key_cache,
            value_cache=value_cache,
            )
        return output_embed, key_cache, value_cache

class FinalLayerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.lm_head

    def forward(self, hidden_states: torch.Tensor):
        logits = self.model(hidden_states[:, -1, :])
        return logits

# Instantiate the wrappers of all the blocks to be exported
wrapped_embedding = EmbeddingWrapper(model)
wrapped_model = LMWrapper(decoder)
wrapped_final_layer = FinalLayerWrapper(model)

# Prepare example inputs for each wrapper
batch_size = 1
seq_len = 20
cache_len = 50
max_seq_len = 1217
embed_dim = 3584
vocab_size = 152064
# embedding
example_inputs_embedding = (
    torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long).to(device),
)
# language decoder
example_inputs = (
    torch.randn(batch_size, seq_len, embed_dim, dtype=data_type).to(device),
    torch.randn(batch_size, cache_len, decoder.total_cache_length, dtype=data_type).to(device),  # key_cache
    torch.randn(batch_size, cache_len, decoder.total_cache_length, dtype=data_type).to(device),  # value_cache
)
# final layer
example_inputs_final_layer = (
    torch.randn(batch_size, seq_len, embed_dim, dtype=data_type).to(device),
)

# Prepare dynamic shapes for the inputs of all the blocks
dyn_seq_len = Dim("seq_len", min=1, max=max_seq_len)
dyn_cache_len = Dim("cache_len", min=1, max=max_seq_len)
# embedding
dynamic_shapes_embedding = {
    "input_ids": (batch_size, dyn_seq_len),
}
# language decoder
dynamic_shapes = {
    "inputs_embeds": (batch_size, dyn_seq_len, embed_dim),
    "key_cache": (batch_size, dyn_cache_len, decoder.total_cache_length),
    "value_cache": (batch_size, dyn_cache_len, decoder.total_cache_length),
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
example_outputs = wrapped_model(*example_inputs)
example_outputs_final_layer = wrapped_final_layer(*example_inputs_final_layer)

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
        exported_module = export(wrapped_embedding, example_inputs_embedding, dynamic_shapes=dynamic_shapes_embedding)
        print("Quantizing graph module...")
        # quantized_module = quantize(exported_module, example_inputs_embedding)
        quantized_module = exported_module
        print("Lowering to edge model...")
        edge_module = to_edge_transform_and_lower(
            # export(quantized_module, example_inputs_embedding),
            exported_module,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[PARTITIONER],
        )
        print("Exporting to Executorch format...")
        executorch_module = edge_module.to_executorch()
        print("Writing to file...")
        write_to_file("embedding.pte", executorch_module.buffer)

    if EXPORT_LM_DECODER:
        print("EXPORTING LANGUAGE DECODER")
        print("Exporting to ATen graph...")
        exported_module = export(wrapped_model, example_inputs, dynamic_shapes=dynamic_shapes)
        print("Quantizing graph module...")
        # quantized_module = quantize(exported_module, example_inputs)
        quantized_module = exported_module
        print("Lowering to edge model...")
        edge_module = to_edge_transform_and_lower(
            # export(quantized_module, example_inputs),
            exported_module,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[PARTITIONER],
        )
        print("Exporting to Executorch format...")
        executorch_module = edge_module.to_executorch()
        print("Writing to file...")
        write_to_file("lm_decoder.pte", executorch_module.buffer)

    if EXPORT_FINAL_LAYER:
        print("EXPORTING FINAL LAYER")
        print("Exporting final layer to graph format...")
        exported_program_final_layer = export(wrapped_final_layer, example_inputs_final_layer, dynamic_shapes=dynamic_shapes_final_layer)
        print("Exporting final layer to Executorch format...")
        executorch_program_final_layer = to_edge_transform_and_lower(
            exported_program_final_layer,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=[PARTITIONER],
        ).to_executorch()
        write_to_file("final_layer.pte", executorch_program_final_layer.buffer)

print("Loading and running Executorch program...")
from executorch.runtime import Runtime

runtime = Runtime.get()
# load program
program_embedding = runtime.load_program("embedding.pte")
program_lm_decoder = runtime.load_program("lm_decoder.pte")
# program_final_layer = runtime.load_program("final_layer.pte")
# extract method
method_embedding = program_embedding.load_method("forward")
method_lm = program_lm_decoder.load_method("forward")
# method_final_layer = program_final_layer.load_method("forward")
# run method
outputs_embedding = method_embedding.execute(list(example_inputs_embedding))
outputs = method_lm.execute(list(example_inputs))
# outputs_final_layer = method_final_layer.execute(list(example_inputs_final_layer))

if type(example_outputs_embedding) is tuple or type(example_outputs_embedding) is list:
    example_output_embedding = example_outputs_embedding[0]
if type(example_outputs) is tuple or type(example_outputs) is list:
    example_output = example_outputs[0]
# if type(example_outputs_final_layer) is tuple or type(example_outputs_final_layer) is list:
#     example_output_final_layer = example_outputs_final_layer[0]

print("Maximum error between original and executorch program (embedding):", torch.max(torch.abs(example_outputs_embedding - outputs_embedding[0])).item())
print("Maximum error between original and executorch program (lm):", torch.max(torch.abs(example_outputs[0] - outputs[0])).item())
exit()
# print("Maximum error between original and executorch program (final layer):", torch.max(torch.abs(example_outputs_final_layer - outputs_final_layer[0])).item())


print("Loading and running Executorch program with a different sequence length...")
# Test with a different sequence length
seq_len = 26
# Prepare new example input
# embedding
example_inputs_embedding = (
    torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
)
# language decoder
example_inputs = (
    torch.randn(batch_size, 1, embed_dim, dtype=data_type).to(device),
    torch.randn(batch_size, seq_len, decoder.total_cache_length, dtype=data_type).to(device),  # key_cache
    torch.randn(batch_size, seq_len, decoder.total_cache_length, dtype=data_type).to(device),  # value_cache
)
# final layer
example_inputs_final_layer = (
    torch.randn(1, seq_len, embed_dim, dtype=data_type),
)
# run torch method for reference
example_outputs_embedding = wrapped_embedding(*example_inputs_embedding)
example_outputs = wrapped_model(*[example_inputs[0], example_inputs[1], example_outputs_embedding, example_inputs[3]])
example_outputs_final_layer = wrapped_final_layer(example_outputs[0])
# run executorch methods
outputs_embedding = method_embedding.execute(list(example_inputs_embedding))
outputs = method_lm.execute([example_inputs[0], example_inputs[1], outputs_embedding[0], example_inputs[3]])
outputs_final_layer = method_final_layer.execute(list(outputs))

print("Maximum error between original and executorch program (embedding):", torch.max(torch.abs(example_outputs_embedding - outputs_embedding[0])).item())
print("Maximum error between original and executorch program (lm):", torch.max(torch.abs(example_outputs[0] - outputs[0])).item())
print("Maximum error between original and executorch program (final layer):", torch.max(torch.abs(example_outputs_final_layer - outputs_final_layer[0])).item())
