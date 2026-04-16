import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_AVAILABLE_THREADS = torch.get_num_threads()
print(f"Available CPU threads: {MAX_AVAILABLE_THREADS}")
torch.set_num_threads(min(MAX_AVAILABLE_THREADS, 32))
print(f"Using {torch.get_num_threads()} CPU threads for PyTorch operations")

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import Dim, export, draft_export
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

from torchao.quantization import (
    quantize_,
    Int4WeightOnlyConfig,
)

from transformers import Qwen2TokenizerFast

from transformersurgeon import (
    Qwen2ForCausalLMCompress,
    Qwen2ConfigCompress,
    Qwen2CompressionSchemesManager,
    convert_for_export,
)

### TEST CONFIGURATION ###
PARTITIONER = XnnpackPartitioner()
EXPORT_TO_EXECUTORCH = True
EXPORT_EMBEDDING = True
EXPORT_LM_DECODER = True
EXPORT_FINAL_LAYER = True
CHECK_IR_VALIDITY = True

data_type = torch.float16
device = torch.device("cpu")
VERBOSE = False

modelClass = Qwen2ForCausalLMCompress
model_name = "Qwen/Qwen2-0.5B-Instruct"
max_seq_len = 2048

# Load model and tokenizer
tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
model = modelClass.from_pretrained(model_name, torch_dtype="auto").to(device)

embedding = model.get_input_embeddings()
final_layer = model.lm_head

convert_options = {
    "use_sdpa": False,
}
converted_models = convert_for_export(model, options=convert_options, verbose=VERBOSE)
decoder = converted_models['text']

embedding = embedding.to(device, dtype=data_type)
decoder = decoder.to(device, dtype=data_type)
final_layer = final_layer.to(device, dtype=data_type)

# Wrapper modules

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
        return self.module(inputs_embeds, cache_len=cache_len)

class FinalLayerWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, hidden_states: torch.Tensor):
        return self.module(hidden_states[:, -1, :])

wrapped_embedding = EmbeddingWrapper(embedding)
wrapped_decoder = DecoderWrapper(decoder)
wrapped_final_layer = FinalLayerWrapper(final_layer)

# Example inputs
batch_size = 1
seq_len = 2
cache_len = 5
if hasattr(model.config, "text_config"):
    embed_dim = model.config.text_config.hidden_size
else:
    embed_dim = model.config.hidden_size
vocab_size = 152064

example_inputs_embedding = (
    torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
)
example_inputs_decoder = (
    torch.randn(batch_size, seq_len, embed_dim, dtype=data_type),
    torch.ones(cache_len),
)
example_inputs_final_layer = (
    torch.randn(batch_size, seq_len, embed_dim, dtype=data_type),
)

# Dynamic shapes
dyn_seq_len = Dim("dyn_seq_len", min=1, max=max_seq_len - 1)
dyn_cache_len = Dim("dyn_cache_len", min=1, max=max_seq_len - 1)

dynamic_shapes_embedding = {"input_ids": (Dim.STATIC, dyn_seq_len)}
dynamic_shapes_decoder = {
    "inputs_embeds": (Dim.STATIC, dyn_seq_len, Dim.STATIC),
    "cache_len_tensor": (dyn_cache_len,),
}
dynamic_shapes_final_layer = {"hidden_states": (Dim.STATIC, dyn_seq_len, Dim.STATIC)}

# INT4 quantizer (per-channel weights, per-tensor activations — standard XNNPACK config)
quantizer = XNNPACKQuantizer().set_global(
    get_symmetric_quantization_config(
        is_per_channel=True,
        is_dynamic=True,
        weight_qmin=-8,
        weight_qmax=7,)
)

def quantize_module(module, example_inputs, dynamic_shapes=None, use_torchao=False):
    module.eval()
    if use_torchao:
        quantize_(module, Int4WeightOnlyConfig(group_size=64))
        if dynamic_shapes is not None:
            quantized = export(module, example_inputs, dynamic_shapes=dynamic_shapes)
        else:
            quantized = export(module, example_inputs)
        prepared = None
    else:
        if dynamic_shapes is not None:
            exported = export(module, example_inputs, dynamic_shapes=dynamic_shapes)
        else:
            exported = export(module, example_inputs)
        prepared = prepare_pt2e(exported.module(), quantizer)
        # Run one forward pass so weight observers see the weight tensors.
        # For proper activation calibration, replace example_inputs with representative data.
        with torch.no_grad():
            prepared(*example_inputs)
        converted = convert_pt2e(prepared)
        if dynamic_shapes is not None:
            quantized = export(converted, example_inputs, dynamic_shapes=dynamic_shapes)
        else:
            quantized = export(converted, example_inputs)
    return prepared, quantized

def write_to_file(filename: str, data: bytes):
    from tqdm import tqdm
    chunk_size = 1024 * 1024 * 10
    with open(filename, "wb") as f:
        for i in tqdm(range(0, len(data), chunk_size), desc=f"Writing {filename}"):
            f.write(data[i:i + chunk_size])
            f.flush()
            os.fsync(f.fileno())

if EXPORT_TO_EXECUTORCH:

    if EXPORT_EMBEDDING:
        print("EXPORTING EMBEDDING LAYER (INT4)")
        prepared_embedding, quantized = quantize_module(wrapped_embedding, example_inputs_embedding, dynamic_shapes_embedding)
        edge = to_edge_transform_and_lower(
            quantized,
            compile_config=EdgeCompileConfig(_check_ir_validity=CHECK_IR_VALIDITY),
            partitioner=[PARTITIONER],
        )
        write_to_file("embedding_int4.pte", edge.to_executorch().buffer)

    if EXPORT_LM_DECODER:
        print("EXPORTING LANGUAGE DECODER (INT4)")
        prepared_decoder, quantized = quantize_module(wrapped_decoder, example_inputs_decoder, dynamic_shapes_decoder)
        edge = to_edge_transform_and_lower(
            quantized,
            compile_config=EdgeCompileConfig(_check_ir_validity=CHECK_IR_VALIDITY),
            partitioner=[PARTITIONER],
        )
        open("lm_decoder_int4.log", "w").write(
            edge.exported_program().graph_module.print_readable(print_output=False)
        )
        write_to_file("lm_decoder_int4.pte", edge.to_executorch().buffer)

    if EXPORT_FINAL_LAYER:
        print("EXPORTING FINAL LAYER (INT4)")
        prepared_final_layer, quantized = quantize_module(wrapped_final_layer, example_inputs_final_layer, dynamic_shapes_final_layer)
        edge = to_edge_transform_and_lower(
            quantized,
            compile_config=EdgeCompileConfig(_check_ir_validity=CHECK_IR_VALIDITY),
            partitioner=[PARTITIONER],
        )
        write_to_file("final_layer_int4.pte", edge.to_executorch().buffer)
