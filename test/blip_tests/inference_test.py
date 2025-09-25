
"""
BLIP/BLIP-2 Unconditional Image Captioning Script

This script generates captions for images using either BLIP or BLIP-2 models.
No text input or VQA is supported—only pure image captioning.
"""


import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)
from test_messages import messages


### CONFIGURATION ###
# Choose either "blip" or "blip2"
model_type = "blip2"  # "blip" or "blip2"

# BLIP-2 Model Options (uncomment to override default):
# model_name = "Salesforce/blip2-opt-2.7b"      # Smaller, faster (default)
# model_name = "Salesforce/blip2-flan-t5-xl"    # Better instruction following
# model_name = "Salesforce/blip2-opt-6.7b"      # Larger, more capable
##########################

if model_type == "blip":
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
elif model_type == "blip2":
    if 'model_name' not in locals():
        model_name = "Salesforce/blip2-opt-2.7b"
    print(f"Loading BLIP-2 model: {model_name}")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


print(f"Using model: {model_name}")
print(f"Device: {device}")
print(f"Model type: {model_type}")
print("-" * 50)

print("Generating captions...")
for i, message in enumerate(messages):
    print(f"\nProcessing message {i + 1}...")
    # Find image path only
    image_path = None
    for content in message["content"]:
        if content["type"] == "image":
            image_path = content["image"]
            break
    if image_path is None:
        print("No image found in message, skipping...")
        continue
    print("Image:", image_path)
    try:
        image = Image.open(image_path).convert('RGB')
        if model_type == "blip":
            inputs = processor(image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_length=50, num_beams=5)
            caption = processor.decode(out[0], skip_special_tokens=True)
            print("Generated caption:", caption)
        elif model_type == "blip2":
            inputs = processor(image, return_tensors="pt").to(device)
            generated_ids = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                length_penalty=1.0
            )
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            print("Generated caption:", caption)
        print("-" * 40)
    except FileNotFoundError:
        print(f"Image file '{image_path}' not found, skipping...")
        continue
    except Exception as e:
        print(f"Error processing message: {e}")
        continue
print("\nInference completed!")
