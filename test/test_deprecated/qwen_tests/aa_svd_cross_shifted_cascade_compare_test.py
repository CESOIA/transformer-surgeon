import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

from transformersurgeon.models.qwen2_c.define_qwen2_c import (
    Qwen2CompressionSchemesManager,
    Qwen2ForCausalLMCompress,
)


class JsonlMessagesDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def build_dialog_collate_fn(tokenizer, max_length: int = 1024):
    def collate(samples):
        texts = []
        for sample in samples:
            messages = sample.get("messages", [])
            text = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    text.append(f"User: {content}")
                elif role == "assistant":
                    text.append(f"Assistant: {content}")
                else:
                    text.append(f"{role.capitalize()}: {content}")
            texts.append("\n".join(text))

        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return collate


def set_qwen_ranks(manager):
    manager.set("lrd", "rank", 64, criteria=["attn"], verbose=True)
    manager.set("lrd", "rank", 496, criteria=["mlp"], verbose=True)


def build_calibration_loader(tokenizer, dataset_path: Path, max_samples: int, max_length: int):
    dataset = JsonlMessagesDataset(dataset_path)
    if max_samples is not None and max_samples > 0:
        capped = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(capped)))

    # Deterministic order is required by the sanity check.
    collate_fn = build_dialog_collate_fn(tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )


def run_aa_svd_cascade(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    device: torch.device,
    max_samples: int,
    max_batches: int,
    max_length: int,
):
    print("Step 1/3: AA-SVD cascade -> dump shifted_covariance")

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
    model = Qwen2ForCausalLMCompress.from_pretrained(model_name, torch_dtype="auto").to(device)

    calibration_loader = build_calibration_loader(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        max_samples=max_samples,
        max_length=max_length,
    )

    manager = Qwen2CompressionSchemesManager(model)
    manager.set("lrd", "method", "aa-svd", verbose=True)
    set_qwen_ranks(manager)
    manager.set_calibration_mode(mode="cascade")
    manager.set_calibration_data(calibration_loader)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager.apply(
        hard=False,
        criteria="all",
        verbose=True,
        max_batches=max_batches,
        device=device,
        offload_to_cpu=False,
        summary_dump_dir=str(output_dir),
        summary_dump_names=("shifted_covariance",),
    )

    return model, tokenizer


def run_svdllm_standard_calibration_on_compressed_model(
    compressed_model,
    tokenizer,
    dataset_path: Path,
    output_dir: Path,
    device: torch.device,
    max_samples: int,
    max_batches: int,
    max_length: int,
):
    print("Step 2/3: SVD-LLM-v2 standard calibration on compressed model -> dump covariance")

    calibration_loader = build_calibration_loader(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        max_samples=max_samples,
        max_length=max_length,
    )

    manager = Qwen2CompressionSchemesManager(compressed_model)
    manager.set("lrd", "method", "svd-llm-v2", verbose=True)
    set_qwen_ranks(manager)
    manager.set_calibration_mode(mode="standard")
    manager.set_calibration_data(calibration_loader)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager.run_calibration(
        criteria="all",
        max_batches=max_batches,
        device=device,
        offload_to_cpu=False,
        verbose=True,
        show_progress=True,
        summary_dump_dir=str(output_dir),
        summary_dump_names=("covariance",),
    )


def load_summary_map(summary_root: Path, summary_name: str) -> Dict[str, torch.Tensor]:
    summary_dir = summary_root / summary_name
    if not summary_dir.exists():
        raise RuntimeError(f"Summary folder not found: {summary_dir}")

    result = {}
    for file in sorted(summary_dir.glob("*.pt")):
        payload = torch.load(file, map_location="cpu")
        scheme_path = payload["scheme_path"]
        value = payload["value"]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{summary_name} value is not a tensor for {scheme_path}: {type(value)}")
        result[scheme_path] = value.float()

    if len(result) == 0:
        raise RuntimeError(f"No {summary_name} summaries found in {summary_dir}")

    return result


def compare_shifted_vs_covariance(
    shifted_root: Path,
    covariance_root: Path,
):
    print("Step 3/3: Compare shifted_covariance vs covariance")

    shifted_map = load_summary_map(shifted_root, "shifted_covariance")
    covariance_map = load_summary_map(covariance_root, "covariance")

    shifted_keys = set(shifted_map.keys())
    covariance_keys = set(covariance_map.keys())

    missing = sorted(shifted_keys - covariance_keys)
    extra = sorted(covariance_keys - shifted_keys)

    if missing:
        raise AssertionError(
            f"Missing covariance entries for shifted_covariance keys. count={len(missing)} preview={missing[:5]}"
        )
    if extra:
        raise AssertionError(
            f"Extra covariance entries not present in shifted_covariance. count={len(extra)} preview={extra[:5]}"
        )

    max_abs = 0.0
    mean_abs_sum = 0.0
    elem_count = 0
    worst_key = None

    print("Per-layer report: shifted_covariance (AA-SVD cascade) vs covariance (SVD-LLM-v2 standard on compressed model)")
    for key in sorted(shifted_keys):
        shifted = shifted_map[key]
        cov = covariance_map[key]

        if shifted.shape != cov.shape:
            raise AssertionError(
                f"Shape mismatch for {key}: shifted={tuple(shifted.shape)} covariance={tuple(cov.shape)}"
            )

        abs_diff = (shifted - cov).abs()
        local_max = float(abs_diff.max().item())
        local_mean = float(abs_diff.mean().item())

        if local_max > max_abs:
            max_abs = local_max
            worst_key = key

        mean_abs_sum += float(abs_diff.sum().item())
        elem_count += int(abs_diff.numel())

        print(
            f"  {key} | shape={tuple(shifted.shape)} | "
            f"mean_abs_error={local_mean:.6e} | max_abs_error={local_max:.6e}"
        )

    mean_abs = mean_abs_sum / max(elem_count, 1)
    print("Summary: shifted_covariance vs covariance")
    print(f"  compared_layers={len(shifted_keys)}")
    print(f"  global_mean_abs_error={mean_abs:.6e}")
    print(f"  global_max_abs_error={max_abs:.6e}")
    print(f"  worst_layer={worst_key}")

    return {
        "compared_layers": len(shifted_keys),
        "global_mean_abs_error": mean_abs,
        "global_max_abs_error": max_abs,
        "worst_layer": worst_key,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Requested sanity flow: AA-SVD cascade shifted_covariance, then SVD-LLM-v2 standard covariance "
            "on the same compressed model and same dataset order."
        )
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/lucianoprono/private/transformer-surgeon/experiments/llm_export/qwen2_compressed/automotive_990_examples.jsonl",
    )
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "artifacts" / "aa_svd_shifted_vs_covariance"),
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_root = Path(args.output_root).resolve()
    shifted_dir = output_root / "aa_svd_cascade"
    covariance_dir = output_root / "svdllm_standard_on_cascade_model"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    cascade_model, tokenizer = run_aa_svd_cascade(
        model_name=args.model,
        dataset_path=dataset_path,
        output_dir=shifted_dir,
        device=device,
        max_samples=args.max_samples,
        max_batches=args.max_batches,
        max_length=args.max_length,
    )

    run_svdllm_standard_calibration_on_compressed_model(
        compressed_model=cascade_model,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        output_dir=covariance_dir,
        device=device,
        max_samples=args.max_samples,
        max_batches=args.max_batches,
        max_length=args.max_length,
    )

    stats = compare_shifted_vs_covariance(
        shifted_root=shifted_dir,
        covariance_root=covariance_dir,
    )

    if stats["global_max_abs_error"] > args.atol:
        raise AssertionError(
            "Shifted-vs-covariance sanity check failed: "
            f"max_abs_error={stats['global_max_abs_error']:.6e} > atol={args.atol:.6e}"
        )

    print("Requested sanity check passed.")


if __name__ == "__main__":
    main()
