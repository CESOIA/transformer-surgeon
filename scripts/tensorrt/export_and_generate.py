"""Export Qwen2 through ONNX to TensorRT, then generate with the engine.

    python scripts/tensorrt/export_and_generate.py                       # fp16
    python scripts/tensorrt/export_and_generate.py --quant 4             # INT4 weight-only
    python scripts/tensorrt/export_and_generate.py --quant 4 --int4-backend edgellm_plugin
    python scripts/tensorrt/export_and_generate.py --no-build            # ONNX only (build on the target)
    python scripts/tensorrt/export_and_generate.py --skip-export         # reuse an existing export

The artifact is out-dir/model.onnx + model.manifest.json (+ model.engine when
built here). On a Jetson, copy the ONNX + manifest and build there with
    python -m transformersurgeon.export.tensorrt.tensorrt_export model.manifest.json
Needs onnx, onnxscript, the `tensorrt` package and a CUDA device; the Edge-LLM
INT4 plugin additionally needs EDGELLM_PLUGIN_PATH (see edgellm_int4.py).
"""

import argparse
import os
import time

import torch
from transformers import AutoTokenizer

from transformersurgeon import Qwen2ForCausalLMCompress
from transformersurgeon.export import export_to_backend
from transformersurgeon.export.tensorrt import TensorRTExportConfig
from transformersurgeon.export.tensorrt.session import TensorRTLLMSession
from transformersurgeon.models.qwen2_c import Qwen2CompressionSchemesManager


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--out-dir", default="artifacts/qwen2_trt")
    parser.add_argument("--max-cache-len", type=int, default=1024)
    parser.add_argument("--max-input-len", type=int, default=512)
    parser.add_argument("--quant", type=int, choices=[4, 8], default=None,
                        help="hard per-channel weight-only quantization of all decoder linears")
    parser.add_argument("--int4-backend", choices=["dequantize", "edgellm_plugin"], default="dequantize")
    parser.add_argument("--no-build", action="store_true", help="export ONNX + manifest only")
    parser.add_argument("--skip-export", action="store_true", help="reuse out-dir/model.manifest.json")
    parser.add_argument("--prompt", default="Tell me one short fact about France.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def export(args) -> str:
    model = Qwen2ForCausalLMCompress.from_pretrained(args.model_name, dtype=torch.float16).eval()
    if args.quant:
        manager = Qwen2CompressionSchemesManager(model)
        criteria = ["self_attn", "mlp"]
        manager.set("quantization", "precision", args.quant, criteria=criteria)
        manager.set("quantization", "granularity", "per_channel", criteria=criteria)
        manager.apply(hard=True, criteria=criteria)

    config = TensorRTExportConfig(
        output_path=os.path.join(args.out_dir, "model.onnx"),
        backend="tensorrt",
        max_input_len=args.max_input_len,
        convert_options={
            "cache_impl": "io_inplace",
            "max_cache_len": args.max_cache_len,
            "rmsnorm_prescale": False,
            "rmsnorm_upcast": True,
        },
        int4_backend=args.int4_backend,
        build_engine=not args.no_build,
    )
    t0 = time.time()
    result = export_to_backend(model, config=config)
    print(f"exported in {time.time() - t0:.1f}s: {result.onnx_path}")
    print(f"  manifest: {result.manifest_path}")
    print(f"  engine  : {result.engine_path}")
    return result.manifest_path


def main():
    args = parse_args()
    manifest = os.path.join(args.out_dir, "model.manifest.json") if args.skip_export else export(args)
    if args.no_build:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )["input_ids"][0]

    session = TensorRTLLMSession(manifest)
    session.generate(prompt_ids, 4)  # warm-up: engine allocation + CUDA-graph capture
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens = session.generate(prompt_ids, args.max_new_tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"\nprompt tokens : {prompt_ids.numel()}")
    print(f"new tokens    : {len(tokens)}")
    print(f"end-to-end    : {elapsed * 1000:.1f} ms ({len(tokens) / elapsed:.1f} tok/s incl. prefill)")
    print(f"output        : {tokenizer.decode(tokens, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
