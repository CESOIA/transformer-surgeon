import argparse
import collections
import os
import re
import traceback
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import torch

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.export import export_to_executorch
from transformersurgeon.export.executorch_exporters.common import build_example_inputs, build_wrapper
from transformersurgeon.export.executorch_exporters.qnn import QNNExportConfig
from transformersurgeon.export.export import _resolve_model_components_for_export
from transformersurgeon.utils import convert_for_export


ATEN_PATTERN = re.compile(r"aten\.[A-Za-z0-9_\.]+")


def parse_args():
    parser = argparse.ArgumentParser(description="Test library ExecuTorch QNN exporter function")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="HF model identifier",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="qnn",
        choices=["qnn"],
        help="Target backend for lowering",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="full",
        choices=["full", "int8"],
        help="Global quant precision",
    )
    parser.add_argument(
        "--soc-model",
        type=str,
        default="SM8650",
        help="Qualcomm SoC model used by QNN compiler spec",
    )
    parser.add_argument(
        "--online-prepare",
        action="store_true",
        default=False,
        help="Enable online_prepare in QNN compiler spec (recommended off for host emulator)",
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
    parser.add_argument(
        "--aten-log-file",
        type=str,
        default=None,
        help=(
            "Optional diagnostics log path. If unset, writes to "
            "<out-dir>/aten_op_problems_<mode>_<backend>_<precision>.log"
        ),
    )
    return parser.parse_args()


def _extract_aten_ops_from_text(text: str) -> list[str]:
    return sorted(set(ATEN_PATTERN.findall(text)))


def _collect_aten_inventory(
    model_input: Any,
    *,
    convert_options: dict[str, Any],
    max_seq_len: int,
) -> tuple[collections.Counter[str], str | None]:
    try:
        embedding, decoder, final_layer, model_config = _resolve_model_components_for_export(
            model_input,
            convert_options=convert_options,
            verbose=False,
        )

        wrapper = build_wrapper(
            {
                "embedding": embedding,
                "decoder": decoder,
                "final_layer": final_layer,
            },
            model_config=model_config,
        )
        wrapper.eval()

        example_inputs = build_example_inputs(
            model_config,
            config=SimpleNamespace(max_seq_len=max_seq_len),
        )
        exported = torch.export.export(wrapper, example_inputs)
        graph_module = exported.module()

        counter: collections.Counter[str] = collections.Counter()
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            target_name = str(node.target)
            if target_name.startswith("aten."):
                counter[target_name] += 1

        return counter, None
    except Exception as exc:  # best-effort diagnostics only
        return collections.Counter(), "".join(traceback.format_exception(exc))


def _resolve_log_path(args: argparse.Namespace) -> str:
    if args.aten_log_file:
        return args.aten_log_file
    return os.path.join(
        args.out_dir,
        f"aten_op_problems_{args.mode}_{args.backend}_{args.precision}.log",
    )


def _write_aten_diagnostics_log(
    *,
    log_path: str,
    args: argparse.Namespace,
    aten_inventory: collections.Counter[str],
    inventory_error: str | None,
    export_error: str | None,
    export_error_aten_ops: list[str],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    lines = [
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"model_name={args.model_name}",
        f"mode={args.mode}",
        f"backend={args.backend}",
        f"precision={args.precision}",
        f"soc_model={args.soc_model}",
        f"online_prepare={args.online_prepare}",
        f"fp16={args.fp16}",
        f"num_shards={args.num_shards}",
        f"max_sequence_length={args.max_sequence_length}",
        f"output_path={output_path}",
        "",
        "[graph_aten_inventory]",
    ]

    if inventory_error:
        lines.append("status=failed")
        lines.append("error=failed to collect FX graph inventory")
        lines.append("")
        lines.append("[graph_aten_inventory_traceback]")
        lines.extend(inventory_error.rstrip().splitlines())
    else:
        lines.append("status=ok")
        lines.append(f"unique_aten_ops={len(aten_inventory)}")
        lines.append(f"total_aten_nodes={sum(aten_inventory.values())}")
        lines.append("")
        lines.append("[graph_aten_inventory_counts]")
        for op_name, count in sorted(aten_inventory.items()):
            lines.append(f"{op_name}={count}")

    lines.append("")
    lines.append("[export_exception_aten_ops]")
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

    print(f"Loading model: {args.model_name}")
    model = Qwen2ForCausalLMCompress.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
    )
    model.eval()

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
        f"export_{args.mode}_{args.backend}_{args.precision}_s{args.num_shards}.pte",
    )
    log_path = _resolve_log_path(args)

    export_config = QNNExportConfig(
        output_path=output_path,
        backend=args.backend,
        precision=args.precision,
        soc_model=args.soc_model,
        is_online_prepare=args.online_prepare,
        use_fp16=args.fp16,
        num_shards=args.num_shards,
        max_seq_len=args.max_sequence_length,
        convert_options={"use_sdpa": False},
        verbose=args.verbose,
    )

    aten_inventory, inventory_error = _collect_aten_inventory(
        model_input,
        convert_options={"use_sdpa": False},
        max_seq_len=args.max_sequence_length,
    )
    export_error = None
    export_error_aten_ops: list[str] = []
    result = None
    try:
        result = export_to_executorch(
            model_input,
            config=export_config,
        )
    except Exception as exc:
        export_error = "".join(traceback.format_exception(exc))
        export_error_aten_ops = _extract_aten_ops_from_text(export_error)
    finally:
        _write_aten_diagnostics_log(
            log_path=log_path,
            args=args,
            aten_inventory=aten_inventory,
            inventory_error=inventory_error,
            export_error=export_error,
            export_error_aten_ops=export_error_aten_ops,
            output_path=output_path,
        )
        print(f"ATen diagnostics log written to: {log_path}")

    if export_error is not None:
        raise RuntimeError(
            "Export failed. See ATen diagnostics log for op details: "
            f"{log_path}"
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
