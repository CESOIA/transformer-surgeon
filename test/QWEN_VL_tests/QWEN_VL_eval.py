import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoProcessor,  Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def build_qwen_messages(img_path, user_text):
    """
    Convert one sample from your dataset format into Qwen-VL message format.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def clean_user_prompt(text):
    """
    Remove the <image> token from the original annotation prompt,
    since Qwen uses structured image inputs instead.
    """
    if text is None:
        return ""
    return text.replace("<image>", "").strip()


def model_eval(eval_data_path, img_folder, output_path, model_name_or_path, max_samples=None, max_new_tokens=128):
    # === Load dataset ===
    with open(eval_data_path, "r") as f:
        eval_dataset = json.load(f)

    if max_samples is not None:
        eval_dataset = eval_dataset[:max_samples]

    # === Load Qwen-VL model and processor ===
    model =  Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)

    model.eval()

    outputs = []

    # === Evaluate ===
    for example in tqdm(eval_dataset, desc="Processing"):
        img_fname = example["image"]
        full_image_path = os.path.join(img_folder, img_fname)

        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            continue

        conversations = example.get("conversations", [])
        if len(conversations) < 2:
            print(f"Skipping malformed example for image {img_fname}")
            continue

        # Your format:
        # conversations[0] -> human question
        # conversations[1] -> gpt reference answer
        user_prompt = clean_user_prompt(conversations[0].get("value", ""))
        ref_answer = conversations[1].get("value", "")

        # === Convert to Qwen message format ===
        messages = build_qwen_messages(full_image_path, user_prompt)

        # === Prepare text prompt ===
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # === Prepare image/video inputs ===
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # === Run inference ===
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        # Remove prompt tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        outputs.append({
            "img_fname": img_fname,
            "ref_answer": ref_answer,
            "model_ut_response": output_text,
        })

    # === Save results ===
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(outputs)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen-VL on a LLaVA-style JSON dataset.")

    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Path to the JSON annotation file"
    )
    parser.add_argument(
        "--img_folder",
        type=str,
        required=True,
        help="Folder containing the images"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="eval.res",
        help="Path to save the output JSON"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Qwen-VL model name or local path"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional number of samples to evaluate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens"
    )

    args = parser.parse_args()

    model_eval(
        eval_data_path=args.eval_data_path,
        img_folder=args.img_folder,
        output_path=args.output_path,
        model_name_or_path=args.model_name_or_path,
        max_new_tokens=args.max_new_tokens
    )

if __name__ == "__main__":
    main()