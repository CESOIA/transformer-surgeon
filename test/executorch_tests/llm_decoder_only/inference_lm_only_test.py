import torch
import os
from copy import deepcopy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_AVAILABLE_THREADS = torch.get_num_threads()
SELECTED_THREADS = 32
print(f"Available CPU threads: {MAX_AVAILABLE_THREADS}")
torch.set_num_threads(min(MAX_AVAILABLE_THREADS, SELECTED_THREADS))  # Limit threads
print(f"Using {torch.get_num_threads()} CPU threads for PyTorch operations")

from executorch.runtime import Runtime

device = torch.device("cpu")
data_type = torch.float32  # Data type for model weights

### LOAD INPUT AND OUTPUT EXAMPLES FOR TESTING ###
print("Load total cache length parameter to shape the input tensors for the decoder...")
total_cache_length = torch.load("total_cache_length.pt")

print("Loading example input tensors...")
example_inputs_embedding = torch.load("example_inputs_embedding.pt")
example_inputs_decoder = torch.load("example_inputs_decoder.pt")
example_inputs_final_layer = torch.load("example_inputs_final_layer.pt")

print("Loading example output tensors for reference...")
example_outputs_embedding = torch.load("example_outputs_embedding.pt")
example_outputs = torch.load("example_outputs_decoder.pt")
example_outputs_final_layer = torch.load("example_outputs_final_layer.pt")

### LOAD THE EXECUTORCH PROGRAMS AND RUN TESTS ###
print("Loading and running Executorch program with loaded example inputs...")
runtime = Runtime.get()
# load program
program_embedding = runtime.load_program("embedding.pte")
program_decoder = runtime.load_program("lm_decoder.pte")
program_final_layer = runtime.load_program("final_layer.pte")
# extract method
method_embedding = program_embedding.load_method("forward")
method_decoder = program_decoder.load_method("forward")
method_final_layer = program_final_layer.load_method("forward")
# run method
outputs_embedding = method_embedding.execute(list(example_inputs_embedding))
outputs = method_decoder.execute(list(example_inputs_decoder))
outputs_final_layer = method_final_layer.execute(list(example_inputs_final_layer))

print("Maximum error between original and executorch program (embedding):", torch.max(torch.abs(example_outputs_embedding[0] - outputs_embedding[0])).item())
print("Maximum error between original and executorch program (lm):", torch.max(torch.abs(example_outputs[0] - outputs[0])).item())
print("Maximum error between original and executorch program (final layer):", torch.max(torch.abs(example_outputs_final_layer[0] - outputs_final_layer[0])).item())

### TEST FULL PIPELINE WITH A DIFFERENT SEQUENCE LENGTH ###
print("Test full executorch pipeline with a different sequence length...")
# Test with a different sequence length
batch_size = 1
seq_len = 10
cache_len = 50
max_seq_len = 1217
embed_dim = 3584
vocab_size = 152064

example_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
example_key_cache = torch.randn(batch_size, cache_len, total_cache_length, dtype=data_type).to(device)
example_value_cache = torch.randn(batch_size, cache_len, total_cache_length, dtype=data_type).to(device)

# run executorch methods
outputs_embedding = method_embedding.execute([example_input_ids])
outputs_decoder = method_decoder.execute([outputs_embedding[0], example_key_cache, example_value_cache])
outputs_final_layer = method_final_layer.execute([outputs_decoder[0]])

# if the program runs without error, we can consider the test successful. We can also check the output shapes
print("Output shape from embedding program:", outputs_embedding[0].shape)
print("Output shape from decoder program:", outputs[0].shape)
print("Output shape from final layer program:", outputs_final_layer[0].shape)

print("Full pipeline executorch test completed successfully.")
