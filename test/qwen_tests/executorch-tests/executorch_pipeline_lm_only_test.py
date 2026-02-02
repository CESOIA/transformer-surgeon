import torch
import os
import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from executorch.runtime import Runtime

from transformers import Qwen2TokenizerFast

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
data_type = torch.float32

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
print(input_ids)

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

# Decode loop (no caching)
max_new_tokens = 64

# Initialize embeddings sequence
inputs_embeds = method_embedding.execute([input_ids])[0]
prompt_length = input_ids.size(1)
attention_mask = torch.triu(torch.full(size=(1, 1, prompt_length, prompt_length), fill_value=torch.finfo(data_type).min, device=device), diagonal=1)
cache_position = torch.arange(0, prompt_length, device=device)
position_ids = cache_position.unsqueeze(0)

# Here insert prefill phase (if cache is used)
temperature = 0.6
output_ids = input_ids
for i in tqdm.tqdm(range(max_new_tokens), "Generating"):
    # Decode
    output_embed = method_lm_decoder.execute([attention_mask, position_ids, inputs_embeds, cache_position])[0]
    
    # Extract logits from output embed
    logits = method_final_layer.execute([output_embed])[0]
    
    # Sample next token from logits
    output_id = logits_to_input_id(logits, temperature=temperature)

    # Append to sequence
    output_ids = torch.cat([output_ids, output_id], dim=1)
    inputs_embeds = torch.cat([inputs_embeds, method_embedding.execute([output_id])[0]], dim=1)
    prompt_length = output_ids.size(1)
    attention_mask = torch.triu(torch.full(size=(1, 1, prompt_length, prompt_length), fill_value=torch.finfo(data_type).min, device=device), diagonal=1)
    cache_position = torch.arange(0, prompt_length, device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)

    # Check for end-of-sequence token
    if output_id.item() == tokenizer.eos_token_id:
        break

# Detokenize output_ids to string
generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

print("Generated text:", generated_text)
