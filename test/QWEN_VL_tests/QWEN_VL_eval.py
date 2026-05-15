import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


### TEST CONFIGURATION ###
model_type = "qwen2_5_vl_c"
##########################


if model_type == "qwen2_vl_c":
    from transformersurgeon import Qwen2VLForConditionalGenerationCompress
    modelClass = Qwen2VLForConditionalGenerationCompress
    DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-3B-Instruct"

elif model_type == "qwen2_5_vl_c":
    from transformersurgeon import Qwen2_5_VLForConditionalGenerationCompress
    modelClass = Qwen2_5_VLForConditionalGenerationCompress
    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

else:
    raise ValueError(f"Unsupported model_type '{model_type}'")


def get_model_input_device(model):
    """
    Get a reasonable device for placing inputs.
    Works better than model.device when device_map='auto' is used.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_qwen_messages(img_path, user_text):
    """
    Convert one sample from dataset format into Qwen-VL message format.
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


def model_eval(
    eval_data_path,
    img_folder,
    output_path,
    model_name_or_path,
    max_samples=100,
    max_new_tokens=128,
):
    # === Load dataset ===
    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    if not isinstance(eval_dataset, list):
        raise ValueError(f"Expected eval dataset to be a list, got {type(eval_dataset)}")

    if max_samples is not None:
        eval_dataset = eval_dataset[:max_samples]

    # === Load model and processor ===
    model = modelClass.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
    )

    # temprorary fix
    model = model.to(torch.float32)

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model.eval()

    input_device = get_model_input_device(model)
    outputs = []

    # === Evaluate ===
    for idx, example in enumerate(tqdm(eval_dataset, desc="Processing")):
        #try:
        img_fname = example["image"]
        full_image_path = os.path.join(img_folder, img_fname)

        if not os.path.exists(full_image_path):
            print(f"[Warning] Image not found: {full_image_path}")
            continue

        conversations = example.get("conversations", [])
        if len(conversations) < 2:
            print(f"[Warning] Skipping malformed example for image {img_fname}")
            continue

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
            add_generation_prompt=True,
        )

        # === Prepare vision inputs ===
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move tensor inputs to the device used for the first model shard
        inputs = {
            k: v.to(input_device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        # === Run inference ===
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Remove prompt tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        outputs.append({
            "user_prompt": user_prompt,
            "img_fname": img_fname,
            "ref_answer": ref_answer,
            "model_ut_response": output_text,
        })

        #except Exception as e:
        #    print(f"[Error] Failed on sample index {idx}: {e}")
        #    continue

    # === Save results ===
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(outputs)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen-VL on a LLaVA-style JSON dataset."
    )

    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Path to the JSON annotation file",
    )
    parser.add_argument(
        "--img_folder",
        type=str,
        required=True,
        help="Folder containing the images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="eval_results.json",
        help="Path to save the output JSON",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Qwen-VL model name or local path",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Optional number of samples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens",
    )

    args = parser.parse_args()

    model_eval(
        eval_data_path=args.eval_data_path,
        img_folder=args.img_folder,
        output_path=args.output_path,
        model_name_or_path=args.model_name_or_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()