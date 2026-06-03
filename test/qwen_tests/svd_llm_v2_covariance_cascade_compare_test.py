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


def set_min_rank_for_all_lrd_layers(manager):
    updated = 0
    for scheme in manager.iter_filtered(criteria="all"):
        module = scheme.get_module()
        if not hasattr(module, "weight"):
            continue
        scheme.set("lrd", "rank", 64, verbose=False)
        updated += 1
    return updated


def set_ranks_for_selected_linear_layers(manager, ranks_by_layer: Dict[str, int] = None):
    if ranks_by_layer is None:
        ranks_by_layer = {
            "mlp.up_proj": 497,
            "mlp.gate_proj": 497,
            "mlp.down_proj": 497,
        }

    for layer_name, rank in ranks_by_layer.items():
        manager.set("lrd", "rank", rank, criteria=layer_name, verbose=False)


def configure_manager(manager):
    manager.set("lrd", "method", "svd-llm-v2", verbose=True)
    count = set_min_rank_for_all_lrd_layers(manager)
    set_ranks_for_selected_linear_layers(manager)
    print(f"Configured LRD rank for {count} candidate schemes.")


def build_calibration_loader(tokenizer, dataset_path: Path, max_samples: int, max_length: int):
    dataset = JsonlMessagesDataset(dataset_path)
    if max_samples is not None and max_samples > 0:
        capped = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(capped)))

    collate_fn = build_dialog_collate_fn(tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )


def run_mode(
    mode: str,
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    device: torch.device,
    max_samples: int,
    max_batches: int,
    max_length: int,
):
    print(f"Running mode={mode}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
    model = Qwen2ForCausalLMCompress.from_pretrained(model_name, torch_dtype="auto").to(device)

    calibration_loader = build_calibration_loader(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        max_samples=max_samples,
        max_length=max_length,
    )

    manager = Qwen2CompressionSchemesManager(model)
    configure_manager(manager)
    manager.set_calibration_mode(mode=mode)
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
        summary_dump_names=("covariance",),
    )


def load_covariance_map(summary_root: Path):
    cov_dir = summary_root / "covariance"
    if not cov_dir.exists():
        raise RuntimeError(f"Covariance summary folder not found: {cov_dir}")

    result = {}
    for file in sorted(cov_dir.glob("*.pt")):
        payload = torch.load(file, map_location="cpu")
        scheme_path = payload["scheme_path"]
        value = payload["value"]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Covariance value is not a tensor for {scheme_path}: {type(value)}")
        result[scheme_path] = value.float()

    if len(result) == 0:
        raise RuntimeError(f"No covariance summaries found in {cov_dir}")

    return result


def compare_covariances(standard_root: Path, cascade_root: Path):
    standard_map = load_covariance_map(standard_root)
    cascade_map = load_covariance_map(cascade_root)

    standard_keys = set(standard_map.keys())
    cascade_keys = set(cascade_map.keys())

    missing = sorted(standard_keys - cascade_keys)
    extra = sorted(cascade_keys - standard_keys)

    if missing:
        raise AssertionError(f"Cascade is missing covariance entries. count={len(missing)} preview={missing[:5]}")
    if extra:
        raise AssertionError(f"Cascade has extra covariance entries. count={len(extra)} preview={extra[:5]}")

    max_abs = 0.0
    mean_abs_sum = 0.0
    elem_count = 0
    worst_key = None
    per_layer_rows = []

    for key in sorted(standard_keys):
        std = standard_map[key]
        cas = cascade_map[key]

        if std.shape != cas.shape:
            raise AssertionError(
                f"Shape mismatch for {key}: standard={tuple(std.shape)} cascade={tuple(cas.shape)}"
            )

        abs_diff = (std - cas).abs()
        local_max = float(abs_diff.max().item())
        local_mean = float(abs_diff.mean().item())
        if local_max > max_abs:
            max_abs = local_max
            worst_key = key

        mean_abs_sum += float(abs_diff.sum().item())
        elem_count += int(abs_diff.numel())
        per_layer_rows.append((key, local_mean, local_max, tuple(std.shape)))

    mean_abs = mean_abs_sum / max(elem_count, 1)

    print("Per-layer covariance error report")
    for layer_name, layer_mean, layer_max, layer_shape in per_layer_rows:
        print(
            f"  {layer_name} | shape={layer_shape} | "
            f"mean_abs_error={layer_mean:.6e} | max_abs_error={layer_max:.6e}"
        )

    print("Covariance comparison summary")
    print(f"  compared_layers={len(standard_keys)}")
    print(f"  global_mean_abs_error={mean_abs:.6e}")
    print(f"  global_max_abs_error={max_abs:.6e}")
    print(f"  worst_layer={worst_key}")

    return {
        "compared_layers": len(standard_keys),
        "global_mean_abs_error": mean_abs,
        "global_max_abs_error": max_abs,
        "worst_layer": worst_key,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare covariance summaries between standard and cascade calibration in svd-llm-v2 mode."
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
        default=str(Path(__file__).resolve().parent / "artifacts" / "svd_llm_v2_covariance_compare"),
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_root = Path(args.output_root).resolve()
    standard_dir = output_root / "standard"
    cascade_dir = output_root / "cascade"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    run_mode(
        mode="standard",
        model_name=args.model,
        dataset_path=dataset_path,
        output_dir=standard_dir,
        device=device,
        max_samples=args.max_samples,
        max_batches=args.max_batches,
        max_length=args.max_length,
    )

    run_mode(
        mode="cascade",
        model_name=args.model,
        dataset_path=dataset_path,
        output_dir=cascade_dir,
        device=device,
        max_samples=args.max_samples,
        max_batches=args.max_batches,
        max_length=args.max_length,
    )

    stats = compare_covariances(standard_root=standard_dir, cascade_root=cascade_dir)
    if stats["global_max_abs_error"] > args.atol:
        raise AssertionError(
            "Covariance sanity check failed: "
            f"max_abs_error={stats['global_max_abs_error']:.6e} > atol={args.atol:.6e}"
        )

    print("Covariance sanity check passed.")


if __name__ == "__main__":
    main()
