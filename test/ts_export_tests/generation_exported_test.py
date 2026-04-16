import torch
import os
import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from transformersurgeon import Qwen2_5_VLForConditionalGenerationCompress
from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.utils import convert_for_export
from transformers import Qwen2TokenizerFast

# MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL_NAME)
# model = Qwen2_5_VLForConditionalGenerationCompress.from_pretrained(MODEL_NAME)
model = Qwen2ForCausalLMCompress.from_pretrained(MODEL_NAME)

# Extract language backbone only
embedding = model.get_input_embeddings()
final_layer = model.lm_head

# Convert to export-compatible modules
max_context_len = 512
convert_options = {
    "use_sdpa": True,  # Whether to use SDPA or regular MHA in the decoder wrapper
    "max_cache_len": max_context_len,  # Maximum cache length for the decoder wrapper
}
converted_models = convert_for_export(model, options=convert_options, verbose=True)
decoder = converted_models['text']

# Set device and data type
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
prompt = """Tell me a story."""

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
temperature = 0.2

# Initialize embeddings sequence
inputs_embeds = embedding(input_ids)

# Prefill phase
output_embed = decoder(
    inputs_embeds,
    last_pos=inputs_embeds.size(1),
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

    start_context_len = output_ids.size(1)
    for i in tqdm.tqdm(range(start_context_len, max_context_len), "Generating"):
        last_pos = output_ids.size(1) + 1

        output_embed = decoder(
            inputs_embeds[:, -1:, :],
            last_pos=last_pos,
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
            print("End-of-sequence token generated, stopping generation.")
            break

# Detokenize output_ids to string
generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

print("Generated text:", generated_text)
