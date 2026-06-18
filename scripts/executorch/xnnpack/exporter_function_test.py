import argparse
import os
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.executorch_exporters.xnnpack import XNNPACKExportConfig
from transformersurgeon.utils import convert_for_export


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test library ExecuTorch exporter function")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="HF model identifier",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="xnnpack",
        choices=["xnnpack"],
        help="Target backend for lowering",
    )
    parser.add_argument(
        "--float-type",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Cast all model components to this dtype before export. Default None preserves existing dtypes.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=1024,
        help="Maximum sequence length for export (used for cache size)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hf",
        choices=["hf", "direct"],
        help=(
            "hf: pass full model and let exporter call convert_for_export; "
            "direct: pre-convert and pass embedding/decoder/final_layer directly"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts",
        help="Output directory for exported pte",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print simple inference error statistics",
    )
    # --- MLP quantization via manager ---
    parser.add_argument(
        "--quant-mlp",
        action="store_true",
        help="Quantize MLP layers using Qwen2CompressionSchemesManager before export",
    )
    parser.add_argument(
        "--quant-precision",
        type=int,
        default=8,
        choices=[4, 8],
        help="Integer bit-width for MLP quantization (4 or 8)",
    )
    parser.add_argument(
        "--quant-mode",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="hard: torchao native quantization; soft: fake-quant (reversible)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="per_channel",
        choices=["per_channel", "per_tensor"],
        help="Weight quantization granularity for MLP layers",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable static activation quantization calibrated on WikiText-2 (requires --quant-mlp)",
    )
    parser.add_argument(
        "--act-precision",
        type=int,
        default=8,
        choices=[4, 8],
        help="Integer bit-width for activation quantization (used with --calibrate)",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=128,
        help="Number of WikiText-2 token chunks to use for activation calibration",
    )
    parser.add_argument(
        "--calibration-seq-len",
        type=int,
        default=512,
        help="Sequence length of each calibration chunk",
    )
    return parser.parse_args()


def _build_pte_stem(args: argparse.Namespace) -> str:
    if args.quant_mlp:
        act_tag = f"_a{args.act_precision}" if args.calibrate else ""
        return f"qwen2_{args.backend}_w{args.quant_precision}{act_tag}_{args.quant_mode}"
    return f"qwen2_{args.backend}_full"


def _build_wikitext_calibration_loader(
    tokenizer,
    num_samples: int = 128,
    seq_len: int = 512,
    seed: int = 42,
) -> DataLoader:
    raw = load_dataset(
        "EleutherAI/wikitext_document_level",
        "wikitext-2-raw-v1",
        split="train",
    )
    texts = [t for t in raw["page"] if isinstance(t, str) and len(t) > 0]
    corpus = "\n\n".join(texts)

    token_ids = tokenizer(corpus, truncation=False, padding=False,
                          return_attention_mask=False)["input_ids"]

    max_samples = len(token_ids) // seq_len
    actual = min(num_samples, max_samples)

    rng = random.Random(seed)
    examples = []
    for _ in range(actual):
        start = rng.randint(0, len(token_ids) - seq_len - 1)
        examples.append({
            "input_ids": torch.tensor(token_ids[start:start + seq_len], dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
        })

    return DataLoader(examples, batch_size=1, shuffle=False)


def quantize_mlp_with_manager(model, args, calibration_loader=None):
    act_info = f", activation=a{args.act_precision}" if calibration_loader is not None else ""
    print(
        f"Quantizing MLP layers: precision=w{args.quant_precision}, "
        f"mode={args.quant_mode}, granularity={args.granularity}{act_info}"
    )
    manager = Qwen2CompressionSchemesManager(model)

    criteria = ["mlp"]
    manager.set("quantization", "precision", args.quant_precision, criteria=criteria)
    manager.set("quantization", "method", "vanilla", criteria=criteria)
    manager.set("quantization", "granularity", args.granularity, criteria=criteria)
    manager.set("quantization", "sparsity", 0.0, criteria=criteria)
    manager.set("quantization", "sparse_method", "magnitude", criteria=criteria)
    manager.set("quantization", "eps", 1e-6, criteria=criteria)

    if calibration_loader is not None:
        manager.set("quantization", "precision_activation", args.act_precision, criteria=criteria)
        manager.set_calibration_data(calibration_loader)

    hard = args.quant_mode == "hard"
    manager.apply(hard=hard, criteria=criteria, verbose=args.verbose)
    print("MLP quantization applied.")
    return model


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model in float16 by default; float_type cast (if any) happens inside export_to_backend.
    load_dtype = _DTYPE_MAP.get(args.float_type, torch.float16) if args.float_type else torch.float16

    print(f"Loading model: {args.model_name}")
    model = Qwen2ForCausalLMCompress.from_pretrained(
        args.model_name,
        torch_dtype=load_dtype,
    )
    model.eval()

    # --- Optional MLP quantization via manager ---
    if args.quant_mlp:
        calibration_loader = None
        if args.calibrate:
            print(f"Building WikiText-2 calibration loader ({args.num_calibration_samples} samples, seq_len={args.calibration_seq_len})")
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            calibration_loader = _build_wikitext_calibration_loader(
                tokenizer,
                num_samples=args.num_calibration_samples,
                seq_len=args.calibration_seq_len,
            )
        quantize_mlp_with_manager(model, args, calibration_loader=calibration_loader)

    if args.mode == "hf":
        model_input = model
    else:
        converted = convert_for_export(
            model,
            options={"use_sdpa": False},
            verbose=False,
        )
        model_input = {
            "embedding": model.get_input_embeddings(),
            "decoder": converted["text"],
            "final_layer": model.lm_head,
            "config": model.config,
        }

    output_path = os.path.join(args.out_dir, f"{_build_pte_stem(args)}.pte")

    # None = preserve each component's existing dtype (enables mixed float precision).
    export_float_type = _DTYPE_MAP[args.float_type] if args.float_type else None

    export_config = XNNPACKExportConfig(
        output_path=output_path,
        backend=args.backend,
        float_type=export_float_type,
        max_seq_len=args.max_sequence_length,
        convert_options={"use_sdpa": False},
        verbose=args.verbose,
    )

    result = export_to_backend(
        model_input,
        config=export_config,
    )

    print("\nExport result:")
    print(f"  pte_path         : {result.pte_path}")
    print(f"  backend          : {result.backend}")
    print(f"  precision        : {result.precision}")
    print(f"  mismatch_count   : {len(result.weight_mismatches)}")

    if result.weight_mismatches:
        print("  top mismatches:")
        for m in result.weight_mismatches[:5]:
            print(
                "    "
                f"weight={m['matched_weight']} "
                f"max_abs_err={m['max_abs_err']:.6g} "
                f"mean_abs_err={m['mean_abs_err']:.6g}"
            )

    if result.inference_stats is not None:
        print("  inference stats:")
        print(f"    max_abs_err : {result.inference_stats['max_abs_err']:.8f}")
        print(f"    mean_abs_err: {result.inference_stats['mean_abs_err']:.8f}")
        print(f"    mse         : {result.inference_stats['mse']:.10f}")
        print(f"    rmse        : {result.inference_stats['rmse']:.8f}")


if __name__ == "__main__":
    main()
