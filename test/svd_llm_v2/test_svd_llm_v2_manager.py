import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer

from transformersurgeon import (  # noqa: E402
    Qwen2CompressionSchemesManager,
    Qwen2ForCausalLMCompress,
    Qwen2_5_VLCompressionSchemesManager,
    Qwen2_5_VLForConditionalGenerationCompress,
)


DEFAULT_VL_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TARGETS = "0:self_attn.q_proj"


class DummyQwen25VLDataset(Dataset):
    def __init__(self, num_samples, image_size):
        self.num_samples = num_samples
        self.image_size = image_size
        self.prompts = [
            "Describe the geometric pattern in one short sentence.",
            "Which colors are most visible in this image?",
            "Count the large colored blocks in the picture.",
            "Summarize the image content briefly.",
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {
            "prompt": self.prompts[index % len(self.prompts)],
            "image": self._make_image(index),
        }

    def _make_image(self, index):
        width, height = self.image_size, self.image_size
        base_color = (
            40 + (index * 37) % 160,
            60 + (index * 53) % 150,
            80 + (index * 29) % 140,
        )
        image = Image.new("RGB", (width, height), base_color)
        draw = ImageDraw.Draw(image)
        step = max(width // 4, 1)
        for i in range(4):
            color = (
                (base_color[0] + 40 * i) % 255,
                (base_color[1] + 70 * i) % 255,
                (base_color[2] + 90 * i) % 255,
            )
            left = (i * step) // 2
            top = (i * step) // 3
            right = min(width - 1, left + step)
            bottom = min(height - 1, top + step)
            draw.rectangle([left, top, right, bottom], fill=color)
        return image


class DummyQADataset(Dataset):
    def __init__(self, num_samples):
        self.samples = [
            {
                "question": "What is low-rank decomposition useful for?",
                "answer": "It can reduce model parameter count and compute by approximating large matrices.",
            },
            {
                "question": "Why do calibration activations help compression?",
                "answer": "They show which input directions matter most for the target layer.",
            },
            {
                "question": "What does a covariance matrix summarize?",
                "answer": "It summarizes second-order relationships among activation features.",
            },
            {
                "question": "When should the calibration pass run?",
                "answer": "It should run before applying the activation-aware low-rank decomposition.",
            },
        ]
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.samples[index % len(self.samples)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="SVD-LLM-v2 calibration test for Qwen2.5-VL 3B and LLM-only Qwen."
    )
    parser.add_argument("--case", choices=["vl", "llm"], default="vl")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Defaults to Qwen2.5-VL-3B for --case vl and Qwen2.5-3B for --case llm.",
    )
    parser.add_argument(
        "--dataset",
        choices=["vl", "qa"],
        default=None,
        help="Defaults to vl for --case vl and qa for --case llm.",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--targets", type=str, default=DEFAULT_TARGETS)
    parser.add_argument("--target-scope", choices=["text", "vision"], default="text")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=56 * 56)
    parser.add_argument("--max-pixels", type=int, default=64 * 64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--apply-with-calibration",
        action="store_true",
        help="Use manager.apply(calibration_data=...) instead of explicit calibrate_lrd() then apply().",
    )
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype):
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def default_model_name(case):
    if case == "llm":
        return DEFAULT_LLM_MODEL_NAME
    return DEFAULT_VL_MODEL_NAME


def parse_target_criteria(targets, target_scope, case):
    if case == "llm":
        scope_token = "model"
    else:
        scope_token = "language_model" if target_scope == "text" else "visual"
    criteria = []
    for spec in targets.split(","):
        spec = spec.strip()
        if not spec:
            continue
        block_id, path = spec.split(":", 1)
        criteria.append([int(block_id), scope_token, path])
    if not criteria:
        raise ValueError("At least one target must be provided.")
    return criteria


def build_qa_prompt(sample):
    return (
        "Answer the following question clearly and concisely.\n"
        f"Question: {sample['question']}\n"
        f"Answer: {sample['answer']}"
    )


def build_collate_fn(processor, dataset_type, text_only):
    def collate(samples):
        texts = []
        images = []
        for sample in samples:
            content = []
            if dataset_type == "qa":
                prompt = build_qa_prompt(sample)
            else:
                prompt = sample["prompt"]

            if dataset_type == "vl" and not text_only:
                content.append({"type": "image"})
                images.append(sample["image"])
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            texts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        processor_kwargs = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }
        if dataset_type == "vl" and not text_only:
            processor_kwargs["images"] = images
        return processor(**processor_kwargs)

    return collate


def build_llm_collate_fn(tokenizer):
    def collate(samples):
        texts = []
        for sample in samples:
            messages = [
                {
                    "role": "user",
                    "content": build_qa_prompt(sample),
                }
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                texts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                texts.append(build_qa_prompt(sample))
        return tokenizer(texts, padding=True, return_tensors="pt")

    return collate


def move_to_device(batch, device):
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def forward_for_calibration(model, batch):
    return model(**batch, use_cache=False)


def get_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    return output[0]


def assert_selected_targets(manager, criteria):
    selected = list(manager.iter_filtered(criteria=criteria))
    if len(selected) != len(criteria):
        paths = [scheme.path for scheme in selected]
        raise RuntimeError(
            f"Expected {len(criteria)} target layers but matched {len(selected)}: {paths}"
        )
    return selected


def assert_rank_is_valid(selected, rank):
    for scheme in selected:
        weight = scheme.get_module().weight
        if rank >= min(weight.size()):
            raise ValueError(
                f"Rank {rank} is invalid for {scheme.path} with weight shape {tuple(weight.shape)}."
            )


def assert_covariances(selected):
    for scheme in selected:
        compressor = scheme.compressors["lrd"]
        module = scheme.get_module()
        in_features = module.weight.shape[1]
        assert compressor.covariance is not None, f"Missing covariance for {scheme.path}"
        assert tuple(compressor.covariance.shape) == (in_features, in_features)
        assert torch.isfinite(compressor.covariance).all(), f"Non-finite covariance for {scheme.path}"
        assert not hasattr(compressor, "_covariance_sum")
        assert not hasattr(compressor, "_covariance_tokens")


def assert_lrd_applied(selected, rank):
    for scheme in selected:
        module = scheme.get_module()
        assert module.rank == rank, f"{scheme.path} rank is {module.rank}, expected {rank}"
        assert module.weight_2 is not None, f"{scheme.path} missing weight_2"
        assert module.weight.shape[1] == rank, f"{scheme.path} left factor has wrong rank"
        assert module.weight_2.shape[0] == rank, f"{scheme.path} right factor has wrong rank"
        assert torch.isfinite(module.weight).all(), f"{scheme.path} has non-finite left factor"
        assert torch.isfinite(module.weight_2).all(), f"{scheme.path} has non-finite right factor"


def main():
    args = parse_args()
    args.model_name = args.model_name or default_model_name(args.case)
    args.dataset = args.dataset or ("qa" if args.case == "llm" else "vl")
    if args.case == "llm" and args.dataset == "vl":
        raise ValueError("--case llm requires --dataset qa.")
    if args.case == "llm" and args.target_scope == "vision":
        raise ValueError("--case llm only supports text targets.")
    if args.text_only and args.target_scope == "vision":
        raise ValueError("--target-scope vision requires image inputs; remove --text-only.")
    if args.dataset == "qa" and args.target_scope == "vision":
        raise ValueError("--dataset qa is text-only and cannot calibrate vision layers.")

    device = resolve_device(args.device)

    if args.case == "llm":
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = Qwen2ForCausalLMCompress.from_pretrained(
            args.model_name,
            torch_dtype=resolve_dtype(args.torch_dtype),
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        manager_class = Qwen2CompressionSchemesManager
        collate_fn = build_llm_collate_fn(tokenizer)
        dataset = DummyQADataset(num_samples=args.num_samples)
    else:
        processor = AutoProcessor.from_pretrained(
            args.model_name,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            trust_remote_code=args.trust_remote_code,
        )
        model = Qwen2_5_VLForConditionalGenerationCompress.from_pretrained(
            args.model_name,
            torch_dtype=resolve_dtype(args.torch_dtype),
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        manager_class = Qwen2_5_VLCompressionSchemesManager
        if args.dataset == "qa":
            dataset = DummyQADataset(num_samples=args.num_samples)
        else:
            dataset = DummyQwen25VLDataset(
                num_samples=args.num_samples,
                image_size=args.image_size,
            )
        collate_fn = build_collate_fn(processor, args.dataset, args.text_only)
    model.eval()

    calibration_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    manager = manager_class(model)
    criteria = parse_target_criteria(args.targets, args.target_scope, args.case)
    selected = assert_selected_targets(manager, criteria)
    assert_rank_is_valid(selected, args.rank)

    print("Selected layers:")
    for scheme in selected:
        print(f"  - {scheme.path}")

    manager.set("lrd", "rank", args.rank, criteria=criteria, verbose=True)
    manager.set("lrd", "method", "svd_llm_v2", criteria=criteria, verbose=True)

    first_batch = move_to_device(next(iter(calibration_loader)), device)
    with torch.no_grad():
        reference_logits = get_logits(forward_for_calibration(model, first_batch)).detach().float().cpu()

    if args.apply_with_calibration:
        manager.apply(
            hard=False,
            criteria=criteria,
            verbose=True,
            calibration_data=calibration_loader,
            calibration_kwargs={
                "max_batches": args.max_batches,
                "device": device,
                "forward_fn": forward_for_calibration,
            },
        )
    else:
        calibrated = manager.calibrate_lrd(
            calibration_data=calibration_loader,
            criteria=criteria,
            max_batches=args.max_batches,
            device=device,
            forward_fn=forward_for_calibration,
            verbose=True,
        )
        assert calibrated == len(selected), f"Calibrated {calibrated}, expected {len(selected)}"
        assert_covariances(selected)
        manager.apply(hard=False, criteria=criteria, verbose=True)

    assert_covariances(selected)
    assert_lrd_applied(selected, args.rank)

    with torch.no_grad():
        compressed_logits = get_logits(forward_for_calibration(model, first_batch)).detach().float().cpu()

    assert tuple(compressed_logits.shape) == tuple(reference_logits.shape)
    assert torch.isfinite(compressed_logits).all()

    rel_error = (
        torch.linalg.norm(compressed_logits - reference_logits)
        / torch.linalg.norm(reference_logits).clamp_min(1e-12)
    ).item()

    print("\nSVD-LLM-v2 manager calibration test passed.")
    print(f"Case: {args.case}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Compressed layers: {len(selected)}")
    print(f"Rank: {args.rank}")
    print(f"Logits relative error: {rel_error:.6f}")


if __name__ == "__main__":
    main()
