import argparse
import os

import torch

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.export import export_to_executorch
from transformersurgeon.utils import convert_for_export


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
        choices=["xnnpack", "qnn"],
        help="Target backend for lowering",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="int8",
        choices=["full", "int4", "int8"],
        help="Global quant precision",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=1024,
        help="Maximum sequence length for export (used for cache size)",
    )
    parser.add_argument(
        "--static-seq-len-1",
        action="store_true",
        help="Export static decode graph with fixed input seq_len=1",
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
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable backend fallback (qnn -> xnnpack)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    model = Qwen2ForCausalLMCompress.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
    )
    model.eval()

    # Keep examples tiny for quick smoke runs.
    if args.static_seq_len_1:
        example_input_ids = torch.randint(0, model.config.vocab_size, (1, 1), dtype=torch.long)
        # Static wrapper currently expects 1-based effective length for seq_len=1 decode.
        example_cache_len = torch.tensor([1], dtype=torch.long)
    else:
        example_input_ids = torch.randint(0, model.config.vocab_size, (1, 3), dtype=torch.long)
        example_cache_len = torch.ones(7)

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

    output_path = os.path.join(
        args.out_dir,
        f"export_{args.mode}_{args.backend}_{args.precision}"
        f"{'_static_seq1' if args.static_seq_len_1 else ''}.pte",
    )

    result = export_to_executorch(
        model_input,
        output_path=output_path,
        backend=args.backend,
        precision=args.precision,
        static_seq_len_1=args.static_seq_len_1,
        calibration_data=None,
        example_input_ids=example_input_ids,
        example_cache_len_tensor=example_cache_len,
        max_seq_len=args.max_sequence_length,
        convert_options={"use_sdpa": False},
        verbose=args.verbose,
        allow_backend_fallback=not args.no_fallback,
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
