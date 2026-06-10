import torch
import os
import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformersurgeon import Qwen2ForCausalLMCompress
from transformers import Qwen2TokenizerFast

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

tokenizer = Qwen2TokenizerFast.from_pretrained(MODEL_NAME)
model = Qwen2ForCausalLMCompress.from_pretrained(MODEL_NAME, dtype=torch.float16)

# Extract language backbone only
embedding = model.get_input_embeddings()
decoder = model.model
final_layer = model.lm_head

# Set device and data type
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_type = torch.float16
max_context_len = 128

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
prompt = """Tell me one short fact about France."""

# Tokenize inputs (string to input_ids)
inputs = tokenizer(
    template.format(instruction=prompt),
    return_tensors="pt",
).to(device)

input_ids = inputs["input_ids"][0]  # Remove batch dimension

# Logits to token sampling function
def logits_to_input_id(logits, temperature=1.0):
    # Greedy decoding for temperature <= 0
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Scale by temperature
    logits = logits / float(temperature)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1)

# Decode loop
temperature = 0.0

# Initialize embeddings sequence
inputs_embeds = embedding(input_ids)

# Prefill phase - perform decode iteratively (TEMP)
past_key_values = None
for i in range(inputs_embeds.size(0)):
    output = decoder(
        inputs_embeds=inputs_embeds[i:i+1].unsqueeze(0),
        position_ids=torch.tensor([[i]], device=device),
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = output.past_key_values
output_embeds = output.last_hidden_state[0]  # Remove batch dim -> (1, hidden)

# Extract logits from output embed
logits = final_layer(output_embeds[-1])
# Sample next token from logits
output_id = logits_to_input_id(logits, temperature=temperature)
# Concatenate to the sequence
output_ids = torch.cat([input_ids, output_id], dim=0)
# Get next input embeddings
inputs_embeds = embedding(output_id)

with torch.no_grad():

    start_context_len = output_ids.size(0)
    for i in tqdm.tqdm(range(start_context_len, max_context_len), "Generating"):
        pos_id = torch.tensor([[output_ids.size(0) - 1]], device=device)

        output = decoder(
            inputs_embeds=inputs_embeds[-1:, :].unsqueeze(0),
            position_ids=pos_id,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = output.past_key_values

        # Extract logits from output embed
        logits = final_layer(output.last_hidden_state[0, -1, :])

        # Sample next token from logits
        output_id = logits_to_input_id(logits, temperature=temperature)

        # Append to sequence
        output_ids = torch.cat([output_ids, output_id], dim=0)

        # Get next input embeddings
        inputs_embeds = embedding(output_id)

        # Check for end-of-sequence token
        if output_id.item() == tokenizer.eos_token_id:
            print("End-of-sequence token generated, stopping generation.")
            break

# Detokenize output_ids to string
generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

print("Generated text:", generated_text)
