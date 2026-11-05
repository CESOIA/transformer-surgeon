"""Orchestrator: plan the E0-E5 experiments and generate (compress + export) every model they need.

    # 1. plan + generate everything for one backend
    python scripts/cost_predictor/generate_models.py --backend xnnpack --out-dir experiments/cost_predictor
    # 2. only some experiments, other backend, resume later (finished artifacts are skipped)
    python scripts/cost_predictor/generate_models.py --backend tensorrt --experiments E0 E1 E2 E3
    # 3. pipeline test without a backend: compress only, tiny random model
    python scripts/cost_predictor/generate_models.py --backend none --tiny --e4-count 6 --e5-count 3

Steps: (1) ``experiment_plan.build_plan`` writes the experiment folders; (2) every configuration that owns its
weights is compressed (hard, per-layer INT8/INT4, FP16 = untouched) and exported once per graph; (3) an
``index_artifacts.csv`` lists the artifact to profile for every (experiment, configuration, P, b_KV).

Graph shape. Static-cache graphs (all backends here) read every cache slot, so decode cost follows
``max_cache_len``, not the live context. To measure the cost at prompt length P the default is therefore one graph
per P with ``max_cache_len = round_up(P + G_max, 128)`` (``--shared-graph`` exports a single graph sized for the
largest P instead). Artifacts: ``<exp>/models/<config>/<backend>/P<P>/``.

KV-cache precision. The library stores the cache in the model dtype and has no KV quantisation, so INT8/INT4 KV
cannot be baked into an artifact. Those configurations (E2, E3 and part of E4/E5) reuse the weights artifact of the
same weights and carry ``b_KV`` in the index (``kv_in_artifact=false``): the profiling runtime must apply the KV
precision (or this has to be added to ``blocks/attention/kv_cache``).
"""

import argparse
import csv
import gc
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch

import experiment_plan as plan_mod
from experiment_plan import LAYER_PATHS, PROJECTIONS

FAMILIES = {   # family -> (causal-LM class, manager class) exported by transformersurgeon.models
    "qwen2": ("Qwen2ForCausalLMCompress", "Qwen2CompressionSchemesManager"),
    "llama": ("LlamaForCausalLMCompress", "LlamaCompressionSchemesManager"),
    "mistral": ("MistralForCausalLMCompress", "MistralCompressionSchemesManager"),
}


@dataclass
class Profile:
    """How a backend is fed: dtype, cache implementation and per-precision quantisation settings (Appendix A:
    symmetric, per-group g=128 where the backend can express it, per-channel otherwise)."""
    dtype: torch.dtype
    kv_ref: str
    quant: dict                       # bits -> quantisation parameters
    cache_impl: str | None = None
    quant_extra: dict = field(default_factory=dict)   # applied to every quantised layer


_GROUP = {"granularity": "per_group", "group_size": plan_mod.GROUP_SIZE}
PROFILES = {
    "xnnpack": Profile(torch.float32, "F32", {8: _GROUP, 4: _GROUP}, "mutable",
                       {"precision_activation": 8, "method_activation": "dynamic_per_token"}),
    # QNN holds block weights only as LPBQ (4-bit multipliers per block); INT8 per-group raises on QNN/TensorRT.
    "qnn": Profile(torch.float16, "F16",
                   {8: {"granularity": "per_channel"}, 4: {**_GROUP, "group_scale_bits": 4}}, "mutable"),
    "tensorrt": Profile(torch.float16, "F16", {8: {"granularity": "per_channel"}, 4: _GROUP}, "io_inplace"),
    "onnx": Profile(torch.float16, "F16", {8: {"granularity": "per_channel"}, 4: _GROUP}, "io_inplace"),
    "none": Profile(torch.float32, "F16", {8: {"granularity": "per_channel"}, 4: _GROUP}),
}


def round_up(n, m=128):
    return -(-n // m) * m


def load_model_classes(family):
    import transformersurgeon.models as models
    model_name, manager_name = FAMILIES[family]
    return getattr(models, model_name), getattr(models, manager_name)


def prepare_base_model(args, out: Path):
    """Return the model path/name to load; the tiny model is created once and saved."""
    if not args.tiny:
        return args.model_name
    path = out / "_tiny_base"
    if not path.exists():
        torch.manual_seed(args.seed)
        model_cls, _ = load_model_classes(args.family)
        model_cls._from_config(plan_mod.tiny_config(), dtype=torch.float32).save_pretrained(path)
    return str(path)


def compress(model, manager_cls, weights, profile: Profile):
    """Hard-quantise the model to the per-layer assignment ``weights[block][proj]`` (16 leaves a layer float)."""
    manager = manager_cls(model)
    counts = {}
    for block, row in enumerate(weights):
        for proj, bits in zip(PROJECTIONS, row):
            if bits == 16:
                continue
            crit = [[LAYER_PATHS[proj], block]]
            manager.set("quantization", "method", "vanilla", criteria=crit)
            manager.set("quantization", "precision", bits, criteria=crit)
            for key, value in {**profile.quant[bits], **profile.quant_extra}.items():
                manager.set("quantization", key, value, criteria=crit)
            counts[bits] = counts.get(bits, 0) + 1
    manager.apply(hard=True)
    return counts


def export_config(backend, out_dir: Path, cache_len, input_len, profile: Profile, args):
    base = {"max_cache_len": cache_len}
    if profile.cache_impl:
        base["cache_impl"] = profile.cache_impl
    if backend == "xnnpack":
        from transformersurgeon.export.backends.xnnpack import XNNPACKExportConfig
        return XNNPACKExportConfig(output_path=str(out_dir / "model.pte"), max_seq_len=cache_len,
                                   convert_options=base, quant_path="torchao")
    if backend == "qnn":
        from transformersurgeon.export.backends.qnn import QNNExportConfig
        return QNNExportConfig(output_path=str(out_dir / "model.pte"), max_seq_len=cache_len,
                               convert_options=base, soc_model=args.soc_model)
    if backend == "tensorrt":
        from transformersurgeon.export.backends.tensorrt import TensorRTExportConfig
        return TensorRTExportConfig(output_path=str(out_dir / "model.onnx"), max_seq_len=cache_len,
                                    max_input_len=input_len,
                                    convert_options={**base, "rmsnorm_prescale": False, "rmsnorm_upcast": True},
                                    build_engine=args.build_engine)
    if backend == "onnx":
        from transformersurgeon.export.backends.onnx import ONNXExportConfig
        return ONNXExportConfig(output_path=str(out_dir / "model.onnx"), max_seq_len=cache_len,
                                max_input_len=input_len,
                                convert_options={**base, "rmsnorm_prescale": False, "rmsnorm_upcast": True})
    raise ValueError(backend)


def graph_shapes(args):
    """[(tag, max_input_len, max_cache_len)] graphs to export per weight configuration."""
    if args.shared_graph:
        p = max(args.prompt_lengths)
        return [(f"P{p}_shared", p, round_up(p + args.g_max))]
    return [(f"P{p}", p, round_up(p + args.g_max)) for p in args.prompt_lengths]


def generate_one(spec, owner_dir: Path, model_source, args, profile, classes):
    """Compress + export one weights configuration for every graph shape; resumable."""
    model_cls, manager_cls = classes
    results = []
    for tag, input_len, cache_len in graph_shapes(args):
        out_dir = owner_dir / args.backend / tag
        status_path = out_dir / "export_status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("status") == "ok" and not args.force:
            results.append((tag, "cached"))
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        status = {"config_id": spec["config_id"], "backend": args.backend, "graph": tag,
                  "max_input_len": input_len, "max_cache_len": cache_len, "dtype": str(profile.dtype)}
        t0 = time.time()
        try:
            model = model_cls.from_pretrained(model_source, dtype=profile.dtype).eval()
            status["quantized_layers"] = {f"INT{b}": n for b, n in compress(model, manager_cls, spec["weights"], profile).items()}
            if args.backend == "none":
                status["status"] = "ok"
                status["note"] = "compression only (no export)"
            else:
                from transformersurgeon.export import export_to_backend
                result = export_to_backend(model, config=export_config(args.backend, out_dir, cache_len, input_len, profile, args))
                status["status"] = "ok"
                status["artifacts"] = {k: str(v) for k, v in vars(result).items() if isinstance(v, (str, Path)) and v}
            del model
        except Exception as exc:  # keep going: one unsupported mixed configuration must not stop the campaign
            status["status"] = "failed"
            status["error"] = "".join(traceback.format_exception(exc))[-4000:]
        status["seconds"] = round(time.time() - t0, 1)
        status_path.write_text(json.dumps(status, indent=2))
        results.append((tag, status["status"]))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def write_index(out: Path, experiments, args):
    rows = []
    for exp in experiments:
        for c in exp.configs:
            owner = out / c.weights_owner.split("/")[0] / "models" / c.weights_owner.split("/")[1]
            for tag, _, _ in graph_shapes(args):
                status_path = owner / args.backend / tag / "export_status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                kv_native = c.kv == PROFILES[args.backend].kv_ref
                rows.append({
                    "experiment": exp.name, "config_id": c.config_id, "backend": args.backend, "graph": tag,
                    "P": "all" if args.shared_graph else tag[1:], "b_KV": c.kv, "kv_in_artifact": kv_native,
                    "weights_owner": c.weights_owner, "status": status.get("status", "missing"),
                    "artifact_dir": str((owner / args.backend / tag).relative_to(out)),
                })
    with open(out / "index_artifacts.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    plan_mod.add_plan_args(parser)
    parser.set_defaults(backends=None)
    parser.add_argument("--backend", default="xnnpack", choices=sorted(PROFILES))
    parser.add_argument("--family", default="qwen2", choices=sorted(FAMILIES))
    parser.add_argument("--experiments", nargs="+", default=["E0", "E1", "E2", "E3", "E4", "E5"])
    parser.add_argument("--shared-graph", action="store_true", help="one graph sized for the largest P")
    parser.add_argument("--soc-model", default="SM8650", help="QNN target SoC")
    parser.add_argument("--build-engine", action="store_true", help="TensorRT: also build the engine here")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many weight configurations")
    parser.add_argument("--force", action="store_true", help="re-export finished artifacts")
    parser.add_argument("--dry-run", action="store_true", help="plan only: write the folders and list the work")
    args = parser.parse_args()
    profile = PROFILES[args.backend]
    args.backends = [args.backend] if args.backend != "none" else ["xnnpack"]
    args.kv_ref = profile.kv_ref      # the backend's default KV precision is the reference

    out = Path(args.out_dir)
    experiments, _ = plan_mod.build_plan(args)
    selected = [e for e in experiments if e.name in args.experiments]

    # One export per distinct weights; the owner may live in an experiment that was not selected.
    needed = {c.weights_key for s in selected for c in s.configs}
    todo = [(exp, c) for exp in experiments for c in exp.configs
            if c.weights_key in needed and c.weights_owner == f"{exp.dirname}/{c.config_id}"]
    if args.limit:
        todo = todo[:args.limit]
    n_graphs = len(graph_shapes(args))
    print(f"{len(todo)} weight configurations x {n_graphs} graph(s) = {len(todo) * n_graphs} {args.backend} artifacts")
    if args.dry_run:
        for exp, c in todo:
            print(f"  {exp.dirname}/{c.config_id}  (KV {c.kv}, {c.description})")
        return

    model_source = prepare_base_model(args, out)
    classes = load_model_classes(args.family)
    report = []
    for i, (exp, c) in enumerate(todo, 1):
        spec = json.loads((out / exp.dirname / "models" / c.config_id / "spec.json").read_text())
        results = generate_one(spec, out / exp.dirname / "models" / c.config_id, model_source, args, profile, classes)
        report.append({"config": f"{exp.dirname}/{c.config_id}", "results": results})
        print(f"[{i}/{len(todo)}] {exp.dirname}/{c.config_id}: " + ", ".join(f"{t}={s}" for t, s in results), flush=True)

    rows = write_index(out, selected, args)
    failed = [r for r in rows if r["status"] not in ("ok",)]
    (out / f"generation_report_{args.backend}.json").write_text(json.dumps(report, indent=2))
    print(f"index: {out / 'index_artifacts.csv'}  ({len(rows)} rows, {len(failed)} not ok)")


if __name__ == "__main__":
    main()
