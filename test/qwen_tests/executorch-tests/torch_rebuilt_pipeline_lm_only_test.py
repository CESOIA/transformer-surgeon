import torch
import os
import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformersurgeon import Qwen2_5_VLForConditionalGenerationCompress
from transformersurgeon.utils import convert_for_export
from transformersurgeon import precompute_rope_inv_freqs
from transformers import Qwen2TokenizerFast

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL_NAME)
model = Qwen2_5_VLForConditionalGenerationCompress.from_pretrained(MODEL_NAME)

# Extract language backbone only
embedding = model.get_input_embeddings()
decoder = model.language_model
final_layer = model.lm_head

# Convert to export-compatible modules
converted_models = convert_for_export(model, verbose=False)
decoder = converted_models['text']
print("Total cache length:", decoder.total_cache_length)

# Set device and data type
# device = torch.device("cpu")
device = torch.device("cuda")
data_type = torch.float16

# Put all modules on the same device
embedding = embedding.to(device, dtype=data_type)
decoder = decoder.to(device, dtype=data_type)
final_layer = final_layer.to(device, dtype=data_type)

# Prepare prompt
template = (
    "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
    "<|im_start|>user\n{instruction}\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)
prompt = """What is the capital of France?"""

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

# Precompute RoPE inverse frequencies
inv_freq = precompute_rope_inv_freqs(
    head_dim=128,
    base=1e6,
    device=device,
    ).to(data_type)

# Decode loop
max_new_tokens = 512
temperature = 0.0

# Initialize embeddings sequence
inputs_embeds = embedding(input_ids)

# Prefill phase
key_cache = torch.zeros((inputs_embeds.size(0), 1, decoder.total_cache_length), device=device, dtype=data_type)
value_cache = torch.zeros_like(key_cache)
output_embed, key_cache, value_cache = decoder(
    inputs_embeds,
    inv_freq=inv_freq,
    key_cache=key_cache,
    value_cache=value_cache,
    )
# Extract logits from output embed
logits = final_layer(output_embed[:, -1, :])
# Sample next token from logits
output_id = logits_to_input_id(logits, temperature=temperature)
# Concatenate to the sequence
output_ids = torch.cat([input_ids, output_id], dim=1)
# Get next input embeddings
inputs_embeds = embedding(output_id)

with torch.no_grad():
    for i in tqdm.tqdm(range(max_new_tokens), "Generating"):
        output_embed, key_cache, value_cache = decoder(
            inputs_embeds[:, -1:, :],
            inv_freq=inv_freq,
            key_cache=key_cache,
            value_cache=value_cache,
            )
        
        # Extract logits from output embed
        logits = final_layer(output_embed[:, -1, :])
        
        # Sample next token from logits
        output_id = logits_to_input_id(logits, temperature=temperature)        

        # Append to sequence
        output_ids = torch.cat([output_ids, output_id], dim=1)

        # Get next input embeddings
        inputs_embeds = embedding(output_id)

        # Check for end-of-sequence token
        if output_id.item() == tokenizer.eos_token_id:
            break

# Detokenize output_ids to string
generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

print("Generated text:", generated_text)
