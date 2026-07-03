import argparse
import os
import random
import re
import traceback
from datetime import datetime

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.executorch_exporters.qnn import QNNExportConfig
from transformersurgeon.utils import convert_for_export


ATEN_PATTERN = re.compile(r"aten\.[A-Za-z0-9_\.]+")

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test library ExecuTorch QNN exporter function")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="HF model identifier",
    )
    parser.add_argument(
        "--float-type",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Cast all model components to this dtype before export. Default None preserves existing dtypes.",
    )
    # --- QNN-specific ---
    parser.add_argument(
        "--soc-model",
        type=str,
        default="SM8850",
        help="Qualcomm SoC model used by QNN compiler spec",
    )
    parser.add_argument(
        "--online-prepare",
        action="store_true",
        default=False,
        help="Enable online_prepare in QNN compiler spec (off by default; not supported on host emulator)",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP16 mode on HTP (--fp16 / --no-fp16, default: enabled)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of graph shards for the decoder (1 = no sharding, >1 enables multi-context)",
    )
    # --- Shared with xnnpack ---
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
    parser.add_argument(
        "--cache-impl",
        type=str,
        default="mutable",
        choices=["mutable", "io_scatter", "io_concat"],
        help="KV-cache implementation for the exported model. 'mutable' keeps the "
             "cache as internal graph state (QNN peak); 'io_scatter'/'io_concat' "
             "expose it as explicit graph I/O (portable to TensorRT etc.).",
    )
    # --- ATen diagnostics (QNN-specific; helps debug unsupported ops) ---
    parser.add_argument(
        "--aten-log-file",
        type=str,
        default=None,
        help=(
            "Optional diagnostics log path. If unset, writes to "
            "<out-dir>/aten_op_problems_<stem>.log only on export failure."
        ),
    )
    return parser.parse_args()


def _qairt_version() -> str:
    """Best-effort QAIRT/QNN SDK version for artifact naming.

    Prefers the QNN_SDK_ROOT directory name (full build id, e.g.
    ``2.37.0.250724``); falls back to the ``version:`` field of the SDK's
    ``sdk.yaml``, then to ``"unknown"``.
    """
    sdk_root = os.environ.get("QNN_SDK_ROOT", "").rstrip("/")
    if sdk_root:
        name = os.path.basename(sdk_root)
        if name:
            return name
        sdk_yaml = os.path.join(sdk_root, "sdk.yaml")
        try:
            with open(sdk_yaml) as fh:
                for line in fh:
                    if line.strip().startswith("version:"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return "unknown"


def _build_pte_stem(args: argparse.Namespace) -> str:
    parts = [f"qwen2_qnn_{args.soc_model.lower()}_qairt{_qairt_version()}"]
    if args.quant_mlp:
        act_tag = f"_a{args.act_precision}" if args.calibrate else ""
        # "hard" is the mandatory quant mode for export, so it carries no
        # information in the file name; only tag the non-default "soft" mode.
        mode_tag = "" if args.quant_mode == "hard" else f"_{args.quant_mode}"
        parts.append(f"w{args.quant_precision}{act_tag}{mode_tag}")
    else:
        parts.append("full")
    if args.num_shards > 1:
        parts.append(f"s{args.num_shards}")
    if not args.fp16:
        parts.append("nofp16")
    return "_".join(parts)


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


def _extract_aten_ops_from_text(text: str) -> list[str]:
    return sorted(set(ATEN_PATTERN.findall(text)))


def _write_aten_diagnostics_log(
    *,
    log_path: str,
    args: argparse.Namespace,
    stem: str,
    export_error: str | None,
    export_error_aten_ops: list[str],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    lines = [
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"model_name={args.model_name}",
        f"mode={args.mode}",
        f"soc_model={args.soc_model}",
        f"online_prepare={args.online_prepare}",
        f"fp16={args.fp16}",
        f"num_shards={args.num_shards}",
        f"quant_mlp={args.quant_mlp}",
        f"quant_precision={args.quant_precision}",
        f"max_sequence_length={args.max_sequence_length}",
        f"stem={stem}",
        f"output_path={output_path}",
        "",
        "[export_exception_aten_ops]",
    ]
    if export_error:
        lines.append("status=failed")
        lines.append(f"unique_aten_ops_in_exception={len(export_error_aten_ops)}")
        for op_name in export_error_aten_ops:
            lines.append(op_name)
        lines.append("")
        lines.append("[export_exception_traceback]")
        lines.extend(export_error.rstrip().splitlines())
    else:
        lines.append("status=ok")
        lines.append("unique_aten_ops_in_exception=0")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
            options={"use_sdpa": False, "cache_impl": args.cache_impl},
            verbose=False,
        )
        model_input = {
            "embedding": model.get_input_embeddings(),
            "decoder": converted["text"],
            "final_layer": model.lm_head,
            "config": model.config,
        }

    stem = _build_pte_stem(args)
    output_path = os.path.join(args.out_dir, f"{stem}.pte")

    export_float_type = _DTYPE_MAP[args.float_type] if args.float_type else None

    export_config = QNNExportConfig(
        output_path=output_path,
        backend="qnn",
        float_type=export_float_type,
        soc_model=args.soc_model,
        is_online_prepare=args.online_prepare,
        use_fp16=args.fp16,
        num_shards=args.num_shards,
        max_seq_len=args.max_sequence_length,
        convert_options={"use_sdpa": False, "cache_impl": args.cache_impl},
        verbose=args.verbose,
    )

    export_error = None
    export_error_aten_ops: list[str] = []
    result = None
    try:
        result = export_to_backend(
            model_input,
            config=export_config,
        )
    except Exception as exc:
        export_error = "".join(traceback.format_exception(exc))
        export_error_aten_ops = _extract_aten_ops_from_text(export_error)
    finally:
        # Only write the diagnostics log on failure, or always if --aten-log-file is set.
        log_path = args.aten_log_file or (
            os.path.join(args.out_dir, f"aten_op_problems_{stem}.log")
            if export_error
            else None
        )
        if log_path:
            _write_aten_diagnostics_log(
                log_path=log_path,
                args=args,
                stem=stem,
                export_error=export_error,
                export_error_aten_ops=export_error_aten_ops,
                output_path=output_path,
            )
            print(f"ATen diagnostics log written to: {log_path}")

    if export_error is not None:
        raise RuntimeError(
            f"Export failed. See ATen diagnostics log for op details: {log_path}"
        )

    assert result is not None

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
