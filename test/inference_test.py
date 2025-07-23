import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from qwen2_vl_c.modeling_qwen2_vl_c import Qwen2VLForConditionalGenerationCompress

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model name
model_name = "Qwen/Qwen2-VL-7B-Instruct"

# Load processor, model and tokenizer
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2VLForConditionalGenerationCompress.from_pretrained(model_name).to(device)

print(processor)
print(tokenizer)
print(model)

# Message
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "example_image.jpg",
            },
            {
                "type": "text",
                "text": "What kind of apples are these?"
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "example_image.jpg",
            },
            {
                "type": "text",
                "text": "Describe the picture in a sentence."
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "example_image.jpg",
            },
            {
                "type": "text",
                "text": "How many apples are there in the image?"
            },
        ],
    }
]

image = Image.open("example_image.jpg").convert("RGB")

# Process each message separately
print("Generating text...")

for i, message in enumerate(messages):
    print(f"\nProcessing message {i + 1}...")
    print("Input:", message["content"][1]["text"])
    
    # Preparation for inference for this single message
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

    # Inference: Generation of the output for this message
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    message_output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Display result immediately
    print("Output:", message_output[0])
    print("-" * 40)