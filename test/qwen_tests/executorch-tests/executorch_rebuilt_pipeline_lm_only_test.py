import torch
import os
import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from executorch.runtime import Runtime

from transformers import Qwen2TokenizerFast

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
data_type = torch.float16

TOTAL_CACHE_LENGTH = 14336  # Predefined total cache length for the model

tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL_NAME)

# Load model runtime
runtime = Runtime.get()
# Load executorch model
program_embedding = runtime.load_program("embedding.pte")
program_lm_decoder = runtime.load_program("lm_decoder.pte")
program_final_layer = runtime.load_program("final_layer.pte")
# extract method
method_embedding = program_embedding.load_method("forward")
method_lm_decoder = program_lm_decoder.load_method("forward")
method_final_layer = program_final_layer.load_method("forward")

device = torch.device("cpu")

template = (
    "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
    "<|im_start|>user\n{instruction}\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)
prompt = "What's the capital of France?"

# Tokenize inputs (string to input_ids)
inputs = tokenizer(
    template.format(instruction=prompt),
    return_tensors="pt",
).to(device)

input_ids = inputs["input_ids"]

# Logits to token sampling function
def logits_to_input_id(logits, temperature=1.0):
    # Greedy decoding for temperature <= 0
    if temperature <= 0.0:
        # Greedy decoding
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # Scale by temperature
    logits = logits/float(temperature)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    return torch.multinomial(probs, num_samples=1)

# Decode loop
max_new_tokens = 64
temperature = 0.6

# Initialize embeddings sequence
inputs_embeds = method_embedding.execute([input_ids])[0]

# Initialize cache
key_cache = torch.empty((inputs_embeds.size(0), 1, TOTAL_CACHE_LENGTH), device=device, dtype=data_type)
value_cache = torch.empty_like(key_cache)

# Prefill
output_embed, key_cache, value_cache = method_lm_decoder.execute([
    inputs_embeds,
    key_cache,
    value_cache,
    ])

# Extract logits from output embed
logits = method_final_layer.execute([output_embed[:, -1:, :]])[0]
# Sample next token from logits
output_id = logits_to_input_id(logits, temperature=temperature)
# Concatenate to the sequence
output_ids = torch.cat([input_ids, output_id], dim=1)
# Get next input embeddings
inputs_embeds = method_embedding.execute([output_id])[0]

position = output_embed.size(1)
temperature = 0.0
output_ids = input_ids
for i in tqdm.tqdm(range(max_new_tokens), "Generating"):
    # Decode
    output_embed, key_cache, value_cache = method_lm_decoder.execute([
        inputs_embeds[:, -1:, :],
        key_cache,
        value_cache,
        ])
    
    # Extract logits from output embed
    logits = method_final_layer.execute([output_embed[:, -1:, :]])[0]
    # Sample next token from logits
    output_id = logits_to_input_id(logits, temperature=temperature)
    # Concatenate to the sequence
    output_ids = torch.cat([output_ids, output_id], dim=1)
    # Get next input embeddings
    inputs_embeds = method_embedding.execute([output_id])[0]
    # Check for end-of-sequence token
    if output_id.item() == tokenizer.eos_token_id:
        break

print(output_ids)

# Detokenize output_ids to string
generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

print("Generated text:", generated_text)
