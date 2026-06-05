import argparse
import os
import subprocess
import time

import numpy as np
import torch
from transformers import Qwen2TokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _qnn_context_mismatch_hint(stderr_text: str) -> str:
    if "QnnContextCustomProtocol expected magic number" not in stderr_text:
        return ""

    return (
        "\nLikely cause:\n"
        "  QNN context-binary/protocol mismatch between exported .pte and current runtime SDK/build.\n"
        "Suggested checks:\n"
        "  1) Re-export the .pte with the same ExecuTorch build and QNN SDK used by qnn_executor_runner.\n"
        "  2) Keep --qnn-sdk-root consistent between export and inference runs.\n"
        "  3) If mismatch persists, try exporting with --online-prepare enabled for host emulator validation."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run text generation with an exported QNN .pte on host using QNN HTP emulator "
            "via qnn_executor_runner (no adb/device required)."
        )
    )
    parser.add_argument(
        "--pte-path",
        type=str,
        default="artifacts/export_hf_qnn_full.pte",
        help="Path to exported .pte file",
    )
    parser.add_argument(
        "--executorch-root",
        type=str,
        default="/home/luciano/Qualcomm_AI/executorch",
        help="Path to ExecuTorch repository root",
    )
    parser.add_argument(
        "--build-folder",
        type=str,
        default="/home/luciano/Qualcomm_AI/executorch/build-x86",
        help="Path to host build folder containing qnn_executor_runner and libqnn_executorch_backend.so",
    )
    parser.add_argument(
        "--qnn-sdk-root",
        type=str,
        default=os.getenv("QNN_SDK_ROOT", ""),
        help="Path to QNN SDK root (defaults to QNN_SDK_ROOT env var)",
    )
    parser.add_argument(
        "--runner-path",
        type=str,
        default="",
        help="Optional explicit path to qnn_executor_runner. Defaults to <build-folder>/examples/qualcomm/executor_runner/qnn_executor_runner",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts/qnn_emulator_inference",
        help="Local folder for temporary I/O and outputs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HF tokenizer identifier used during export",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Say hi",
        help="Prompt text",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. <= 0 uses greedy decoding.",
    )
    parser.add_argument(
        "--sanity-first",
        action="store_true",
        default=True,
        help="Run one forward sanity step before generation and print diagnostics on failure",
    )
    parser.add_argument(
        "--no-sanity-first",
        dest="sanity_first",
        action="store_false",
        help="Disable preflight sanity step",
    )
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        default=False,
        help="Run only one forward sanity step and exit",
    )
    parser.add_argument(
        "--runner-timeout-sec",
        type=float,
        default=900.0,
        help="Timeout in seconds for each qnn_executor_runner invocation",
    )
    parser.add_argument(
        "--clean-workspace",
        action="store_true",
        default=True,
        help="Delete prior temporary artifacts before running",
    )
    parser.add_argument(
        "--no-clean-workspace",
        dest="clean_workspace",
        action="store_false",
        help="Keep existing temporary artifacts in artifact-dir",
    )
    return parser.parse_args()


def logits_to_next_id(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits / float(temperature)
    probs = torch.nn.functional.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def prepare_workspace(path: str, *, clean: bool) -> None:
    os.makedirs(path, exist_ok=True)

    if clean:
        outputs_path = os.path.join(path, "outputs")
        if os.path.isdir(outputs_path):
            for file_name in os.listdir(outputs_path):
                file_path = os.path.join(outputs_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        for file_name in os.listdir(path):
            if not (
                file_name.startswith("runner_")
                and (file_name.endswith(".stdout.log") or file_name.endswith(".stderr.log"))
            ) and file_name not in {
                "input_0_0.raw",
                "input_0_1.raw",
                "input_list.txt",
                "etdump.etdp",
                "debug_output.bin",
                "inference_speed.txt",
            }:
                continue
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    os.makedirs(os.path.join(path, "outputs"), exist_ok=True)


def run_emulator_step(
    *,
    runner_path: str,
    pte_path: str,
    workspace: str,
    token_id: int,
    effective_len: int,
    env: dict[str, str],
    step_tag: str,
    runner_timeout_sec: float,
) -> torch.Tensor:
    input_token_path = os.path.join(workspace, "input_0_0.raw")
    input_len_path = os.path.join(workspace, "input_0_1.raw")
    input_list_path = os.path.join(workspace, "input_list.txt")
    output_folder_path = os.path.join(workspace, "outputs")
    output_path = os.path.join(output_folder_path, "output_0_0.raw")
    etdump_path = os.path.join(workspace, "etdump.etdp")

    torch.tensor([[token_id]], dtype=torch.long).numpy().tofile(input_token_path)
    torch.tensor([effective_len], dtype=torch.long).numpy().tofile(input_len_path)

    with open(input_list_path, "w", encoding="utf-8") as f:
        f.write("input_0_0.raw input_0_1.raw\n")

    if os.path.exists(output_path):
        os.remove(output_path)

    cmd = [
        runner_path,
        "--model_path",
        pte_path,
        "--input_list_path",
        "input_list.txt",
        "--output_folder_path",
        "outputs",
        "--etdump_path",
        "etdump.etdp",
        "--method_index",
        "0",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=runner_timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_data = exc.stdout if exc.stdout is not None else ""
        stderr_data = exc.stderr if exc.stderr is not None else ""
        stdout_text = (
            stdout_data.decode("utf-8", errors="replace")
            if isinstance(stdout_data, (bytes, bytearray))
            else str(stdout_data)
        )
        stderr_text = (
            stderr_data.decode("utf-8", errors="replace")
            if isinstance(stderr_data, (bytes, bytearray))
            else str(stderr_data)
        )
        stdout_log_path = os.path.join(workspace, f"runner_{step_tag}.stdout.log")
        stderr_log_path = os.path.join(workspace, f"runner_{step_tag}.stderr.log")
        with open(stdout_log_path, "w", encoding="utf-8") as f:
            f.write(stdout_text)
        with open(stderr_log_path, "w", encoding="utf-8") as f:
            f.write(stderr_text)
        context_hint = _qnn_context_mismatch_hint(stderr_text)
        raise TimeoutError(
            "qnn_executor_runner timed out.\n"
            f"timeout_sec={runner_timeout_sec}\n"
            f"step_tag={step_tag}\n"
            f"stdout_log={stdout_log_path}\n"
            f"stderr_log={stderr_log_path}\n"
            f"stdout:\n{stdout_text[-4000:]}\n"
            f"stderr:\n{stderr_text[-4000:]}"
            f"{context_hint}"
        ) from exc

    stdout_log_path = os.path.join(workspace, f"runner_{step_tag}.stdout.log")
    stderr_log_path = os.path.join(workspace, f"runner_{step_tag}.stderr.log")
    with open(stdout_log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    with open(stderr_log_path, "w", encoding="utf-8") as f:
        f.write(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            "qnn_executor_runner failed.\n"
            f"returncode={proc.returncode}\n"
            f"step_tag={step_tag}\n"
            f"stdout_log={stdout_log_path}\n"
            f"stderr_log={stderr_log_path}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )

    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Missing output tensor at '{output_path}'.\n"
            f"step_tag={step_tag}\n"
            f"stdout_log={stdout_log_path}\n"
            f"stderr_log={stderr_log_path}\n"
            f"Runner stdout:\n{proc.stdout[-2000:]}"
        )

    logits = np.fromfile(output_path, dtype=np.float32)
    if logits.size == 0:
        raise RuntimeError("Runner produced an empty output tensor")

    return torch.from_numpy(logits).view(1, -1)


def run_sanity_step(
    *,
    runner_path: str,
    pte_path: str,
    workspace: str,
    first_token_id: int,
    env: dict[str, str],
    runner_timeout_sec: float,
) -> None:
    _ = run_emulator_step(
        runner_path=runner_path,
        pte_path=pte_path,
        workspace=workspace,
        token_id=first_token_id,
        effective_len=1,
        env=env,
        step_tag="sanity",
        runner_timeout_sec=runner_timeout_sec,
    )


def main():
    args = parse_args()

    if not args.qnn_sdk_root:
        raise RuntimeError(
            "QNN SDK root is required. Set QNN_SDK_ROOT or pass --qnn-sdk-root."
        )

    pte_path = os.path.abspath(args.pte_path)
    if not os.path.exists(pte_path):
        raise FileNotFoundError(
            f"PTE file not found at '{pte_path}'. Run exporter_function_test.py first."
        )

    runner_path = args.runner_path
    if not runner_path:
        runner_path = os.path.join(
            args.build_folder,
            "examples/qualcomm/executor_runner/qnn_executor_runner",
        )
    runner_path = os.path.abspath(runner_path)
    if not os.path.exists(runner_path):
        raise FileNotFoundError(
            f"qnn_executor_runner not found at '{runner_path}'. Build host runner first."
        )

    backend_lib_dir = os.path.abspath(os.path.join(args.build_folder, "lib"))
    if not os.path.exists(os.path.join(backend_lib_dir, "libqnn_executorch_backend.so")):
        raise FileNotFoundError(
            "Missing libqnn_executorch_backend.so under build folder lib directory."
        )

    qnn_host_lib_dir = os.path.join(args.qnn_sdk_root, "lib", "x86_64-linux-clang")
    if not os.path.isdir(qnn_host_lib_dir):
        raise FileNotFoundError(
            f"QNN host library directory not found at '{qnn_host_lib_dir}'."
        )

    workspace = os.path.abspath(args.artifact_dir)
    prepare_workspace(workspace, clean=args.clean_workspace)

    env = dict(os.environ)
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{qnn_host_lib_dir}:{backend_lib_dir}:{existing_ld}" if existing_ld else f"{qnn_host_lib_dir}:{backend_lib_dir}"

    tokenizer = Qwen2TokenizerFast.from_pretrained(args.model_name)
    template = (
        "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n"
        "<|im_start|>user\n{instruction}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    input_ids = tokenizer(
        template.format(instruction=args.prompt),
        return_tensors="pt",
    )["input_ids"].long()

    output_ids = input_ids[0].clone()
    generated_tokens = 0

    if args.sanity_first or args.sanity_only:
        try:
            run_sanity_step(
                runner_path=runner_path,
                pte_path=pte_path,
                workspace=workspace,
                first_token_id=int(output_ids[0].item()),
                env=env,
                runner_timeout_sec=args.runner_timeout_sec,
            )
            print("Sanity step passed: one forward call executed successfully.")
        except Exception as exc:
            print("Sanity step failed. Detailed diagnostics:")
            print(f"  runner_path   : {runner_path}")
            print(f"  pte_path      : {pte_path}")
            print(f"  workspace     : {workspace}")
            print(f"  qnn_sdk_root  : {args.qnn_sdk_root}")
            print(f"  build_folder  : {args.build_folder}")
            raise

    if args.sanity_only:
        print("Sanity-only mode complete.")
        return

    t_start = time.perf_counter()

    # Prefill phase -- iterative, non efficient (TEMP)
    logits = None
    for effective_len in range(output_ids.size(0)):
        logits = run_emulator_step(
            runner_path=runner_path,
            pte_path=pte_path,
            workspace=workspace,
            token_id=int(output_ids[effective_len].item()),
            effective_len=effective_len,
            env=env,
            step_tag=f"prefill_{effective_len}",
            runner_timeout_sec=args.runner_timeout_sec,
        )
        print(".", end="", flush=True)
    print(" Prefill complete. Starting generation...")

    for _ in range(args.max_new_tokens):
        if logits is None:
            raise RuntimeError("No logits produced before generation loop")

        next_id = logits_to_next_id(logits, args.temperature)
        output_ids = torch.cat([output_ids, next_id.squeeze(0)], dim=0)
        generated_tokens += 1

        if next_id.item() == tokenizer.eos_token_id:
            break

        logits = run_emulator_step(
            runner_path=runner_path,
            pte_path=pte_path,
            workspace=workspace,
            token_id=int(next_id.item()),
            effective_len=int(output_ids.size(0)),
            env=env,
            step_tag=f"decode_{generated_tokens}",
            runner_timeout_sec=args.runner_timeout_sec,
        )

        print(".", end="", flush=True)
    print(" Generation complete.")

    total_time_s = time.perf_counter() - t_start
    tokens_per_s = generated_tokens / max(total_time_s, 1e-12)
    avg_token_time_ms = (total_time_s / max(generated_tokens, 1)) * 1000.0

    generated_text = tokenizer.batch_decode(output_ids.unsqueeze(0), skip_special_tokens=True)[0]

    print("\nGeneration result")
    print(f"  pte_path            : {pte_path}")
    print(f"  model_name          : {args.model_name}")
    print(f"  prompt              : {args.prompt}")
    print(f"  generated_tokens    : {generated_tokens}")
    print(f"  total_inference_s   : {total_time_s:.6f}")
    print(f"  tokens_per_s        : {tokens_per_s:.2f}")
    print(f"  avg_token_time_ms   : {avg_token_time_ms:.3f}")
    print(f"  output_text         : {generated_text}")


if __name__ == "__main__":
    main()
