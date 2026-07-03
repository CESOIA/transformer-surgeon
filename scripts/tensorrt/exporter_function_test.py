"""
Export Qwen2 to TensorRT with a randomly-assigned per-layer compression scheme,
then run in-process inference to compare TensorRT output against the plain
torch reference.

Similar in spirit to scripts/executorch/xnnpack/exporter_function_test.py, but
instead of applying one uniform quantization scheme to all MLP layers via
Qwen2CompressionSchemesManager, every indexed linear layer independently draws
one of the following labels:

  full          - no compression (native dtype)
  int8          - per-channel int8 weight-only quantization (torchao hard)
  int4          - per-channel int4 weight-only quantization (requires --allow-int4)
  lrd_N         - SVD low-rank decomposition to rank N (32/64/128, shape-permitting)
  lrd_N+int8    - LRD (rank N) combined with int8 quantization on the same layer
  lrd_N+int4    - LRD (rank N) combined with int4 quantization (requires --allow-int4)

This exercises the exporter's mixed-precision path (arbitrary per-layer
combinations of quantization and low-rank decomposition in a single engine)
rather than a single fixed scheme. Pass --full-only to skip randomization
entirely and export the unmodified model, as a baseline.

--cache-impl selects how the KV cache is threaded through the exported graph:
"mutable" (default) keeps it as internal module state; "io_scatter" and
"io_concat" expose it as explicit inout graph buffers (fed in and returned
each step) via aten.index_put / positional-mask-where respectively, which is
what a functional runtime like TensorRT requires instead of in-place buffer
mutation.

Requires a CUDA device and the ``torch_tensorrt`` package.
"""

import argparse
import json
import os
import random

import torch
import torch.nn as nn

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager
from transformersurgeon.models.qwen2_c.indexing_qwen2_c import QWEN2_C_INDEXING
from transformersurgeon.blocks import LinearCompressed
from transformersurgeon.compression.quantization import Quantizer
from transformersurgeon.compression.lrd_methods import METHOD_FUNCTIONS as _LRD_SVD
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.tensorrt import TensorRTExportConfig
from transformersurgeon.utils import convert_for_export
from transformersurgeon.utils.utils import flatten_index_paths, get_submodule


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_LRD_RANK_CANDIDATES = [32, 64, 128]

_QUANT_CFG = {
    "method": "vanilla",
    "sparsity": 0.0,
    "sparse_method": "magnitude",
    "eps": 1e-6,
    "granularity": "per_channel",
    "precision_activation": "full",
    "method_activation": "maxmin",
    "scheme_activation": "asymmetric",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export Qwen2 to TensorRT with random per-layer compression")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B-Instruct", help="HF model identifier")
    parser.add_argument(
        "--float-type",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Native dtype to load the model in",
    )
    parser.add_argument("--max-sequence-length", type=int, default=1024, help="Maximum sequence length (KV cache sizing)")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Output directory for the exported engine")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random per-layer compression assignment")
    parser.add_argument("--full-only", action="store_true", help="Disable randomization; export the unmodified model")
    parser.add_argument("--allow-int4", action="store_true", help="Allow int4 in the random label pool (needs int4 kernel support)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target CUDA device for TensorRT compilation")
    parser.add_argument("--use-fp16", action="store_true", default=True, help="Allow FP16 kernels for un-quantized ops")
    parser.add_argument("--output-format", type=str, default="exported_program", choices=["exported_program", "torchscript"])
    parser.add_argument("--verbose", action="store_true", help="Print in-process float-vs-TensorRT inference stats")
    parser.add_argument(
        "--cache-impl",
        type=str,
        default="mutable",
        choices=["mutable", "io_scatter", "io_concat"],
        help="KV-cache implementation for the exported decoder. 'mutable' keeps the "
             "cache as internal graph state (module buffers); 'io_scatter'/'io_concat' "
             "expose it as explicit graph I/O (portable inout buffers, e.g. ScatterND "
             "on TensorRT).",
    )
    return parser.parse_args()


def _write_cache_metadata(output_path: str, model_config, args: argparse.Namespace) -> str:
    """Write a JSON sidecar with the KV-cache geometry next to the exported engine.

    inference_exported_test.py needs num_layers/kv_num_heads/head_dim/max_cache_len
    to build the io_* cache tensors; without this file those have to be typed in
    by hand (and get out of sync with whatever model/--max-sequence-length the
    engine was actually exported with).
    """
    meta = {
        "cache_impl": args.cache_impl,
        "num_layers": int(model_config.num_hidden_layers),
        "kv_num_heads": int(model_config.num_key_value_heads),
        "head_dim": int(model_config.hidden_size // model_config.num_attention_heads),
        "max_cache_len": int(args.max_sequence_length),
    }
    meta_path = os.path.splitext(output_path)[0] + ".cache_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def _build_hf_to_conv_map(num_blocks: int) -> dict[str, str]:
    idx = QWEN2_C_INDEXING["text"]
    source_paths = flatten_index_paths(idx["path_list"])

    flat_matching: list[str] = []
    for _, mapped in idx["layer_matching"].items():
        if isinstance(mapped, str):
            flat_matching.append(mapped)
        else:
            flat_matching.extend(str(m) for m in mapped)

    result: dict[str, str] = {}
    for i in range(num_blocks):
        for src, dst in zip(source_paths, flat_matching):
            hf_path = idx["path_template"].format(block_index=i, path=src)
            result[hf_path] = f"blocks.{i}.{dst}"
    return result


def _compression_options(module: nn.Linear, allow_int4: bool) -> list[str]:
    opts = ["full", "int8"]
    if allow_int4:
        opts.append("int4")
    max_rank = min(module.in_features, module.out_features) - 1
    for r in _LRD_RANK_CANDIDATES:
        if r < max_rank:
            opts.append(f"lrd_{r}")
            opts.append(f"lrd_{r}+int8")
            if allow_int4:
                opts.append(f"lrd_{r}+int4")
    return opts


def _apply_quant(module: nn.Linear, precision: int) -> None:
    Quantizer({**_QUANT_CFG, "precision": precision}).apply(module, hard=True)


def _apply_lrd(module: LinearCompressed, rank: int) -> None:
    cap = min(module.in_features, module.out_features) - 1
    rank = min(rank, cap)
    if rank <= 0:
        return
    with torch.no_grad():
        dtype = module.weight.dtype
        US_r, V_r = _LRD_SVD["svd"](module.weight.detach().float(), rank)
        module.init_lrd(rank)
        module.weight.data.copy_(US_r.to(dtype))
        module.linear_V.weight.data.copy_(V_r.to(dtype))


def assign_random_compression(model, rng: random.Random, allow_int4: bool) -> dict[str, str]:
    """Randomly assign one compression label per linear layer indexed by the manager."""
    manager = Qwen2CompressionSchemesManager(model)
    assignments: dict[str, str] = {}
    for scheme in manager:
        try:
            module = scheme.get_module()
        except Exception:
            assignments[scheme.path] = "skip"
            continue
        if not isinstance(module, nn.Linear):
            assignments[scheme.path] = "skip"
            continue
        options = _compression_options(module, allow_int4)
        assignments[scheme.path] = rng.choice(options)
    return assignments


def apply_quantization(model, assignments: dict[str, str]) -> None:
    """Apply hard int8/int4 quantization to the HF model, in-place.

    Must run *before* convert_for_export: conversion copies weights into new
    decoder modules, so quantizing after conversion would mutate a model that
    is no longer connected to the exported graph.
    """
    for hf_path, label in assignments.items():
        if "int8" in label:
            prec = 8
        elif "int4" in label:
            prec = 4
        else:
            continue
        _apply_quant(get_submodule(model, hf_path), prec)


def apply_lrd(decoder, assignments: dict[str, str], hf_to_conv: dict[str, str]) -> None:
    """Apply SVD LRD to the already-converted decoder's linear layers, in-place.

    Must run *after* convert_for_export, since it targets the converted
    LinearCompressed modules (not the original HF nn.Linear layers).
    """
    for hf_path, label in assignments.items():
        if "lrd_" not in label:
            continue
        lrd_token = next(p for p in label.split("+") if p.startswith("lrd_"))
        rank = int(lrd_token[4:])
        conv_path = hf_to_conv.get(hf_path)
        if conv_path is None:
            continue
        try:
            conv_module = get_submodule(decoder, conv_path)
        except Exception:
            continue
        if isinstance(conv_module, LinearCompressed):
            _apply_lrd(conv_module, rank)


def summarize_assignments(assignments: dict[str, str]) -> None:
    from collections import Counter
    counts = Counter(assignments.values())
    print("Compression assignment summary:")
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<14}: {count}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    load_dtype = _DTYPE_MAP[args.float_type]

    print(f"Loading model: {args.model_name} (dtype={args.float_type})")
    model = Qwen2ForCausalLMCompress.from_pretrained(args.model_name, torch_dtype=load_dtype)
    model.eval()

    num_blocks = model.config.num_hidden_layers
    hf_to_conv = _build_hf_to_conv_map(num_blocks)

    if args.full_only:
        assignments = {path: "full" for path in hf_to_conv}
        stem = f"qwen2_tensorrt_full_seed{args.seed}"
    else:
        assignments = assign_random_compression(model, rng, args.allow_int4)
        stem = f"qwen2_tensorrt_random_seed{args.seed}"

    if args.cache_impl != "mutable":
        stem += f"_{args.cache_impl}"

    summarize_assignments(assignments)
    print(f"KV-cache implementation: {args.cache_impl}")

    if not args.full_only:
        apply_quantization(model, assignments)

    convert_options = {
        "use_sdpa": False,
        "cache_impl": args.cache_impl,
        "max_cache_len": args.max_sequence_length,
    }
    converted = convert_for_export(model, options=convert_options, verbose=False)
    decoder = converted["text"]
    embedding = model.get_input_embeddings()
    final_layer = model.lm_head

    if not args.full_only:
        apply_lrd(decoder, assignments, hf_to_conv)

    output_path = os.path.join(args.out_dir, f"{stem}.pt2")

    export_config = TensorRTExportConfig(
        output_path=output_path,
        backend="tensorrt",
        max_seq_len=args.max_sequence_length,
        convert_options=convert_options,
        device=args.device,
        use_fp16=args.use_fp16,
        output_format=args.output_format,
        verbose=args.verbose,
    )

    result = export_to_backend(
        {
            "embedding": embedding,
            "decoder": decoder,
            "final_layer": final_layer,
            "config": model.config,
        },
        config=export_config,
    )

    meta_path = _write_cache_metadata(result.engine_path, model.config, args)

    print("\nExport result:")
    print(f"  engine_path      : {result.engine_path}")
    print(f"  cache_meta_path  : {meta_path}")
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
        print("  inference stats (torch vs. TensorRT, exporter's own example input):")
        print(f"    max_abs_err : {result.inference_stats['max_abs_err']:.8f}")
        print(f"    mean_abs_err: {result.inference_stats['mean_abs_err']:.8f}")
        print(f"    mse         : {result.inference_stats['mse']:.10f}")
        print(f"    rmse        : {result.inference_stats['rmse']:.8f}")


if __name__ == "__main__":
    main()
