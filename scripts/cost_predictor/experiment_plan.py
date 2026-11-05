"""Generate the experiment plan of "Hardware Cost Predictor for Compressed LLM Deployment" (E0-E5).

Pure planning, no model is loaded or exported here (``generate_models.py`` does that). For every experiment
the script writes one folder:

    <out-dir>/
        plan.json                         global parameters (model dims, P set, G_max, seed, ...)
        index_configs.csv                 every configuration of every experiment, one row each
        E0_reference/
            experiment.json               purpose, formulas, sweep dimensions, what to measure
            configs.csv                   one row per configuration, with the features f(c)
            results_template.csv          one row per (configuration, P) to fill with measurements
            models/<config_id>/spec.json  the configuration (per-layer weight bits, KV precision, owner of the weights)
        E1_weight_sweep/ ... E5_validation/

Configuration c = per-layer weight precision of the 7 projection types of every block (16 = FP16, 8, 4)
+ one KV-cache precision. Experiments that need the same weights share one artifact (``weights_owner``).

Without ``--kv-alts`` the plan is weights-only (E0, E1, E4, E5, every configuration at the reference KV precision).
``--kv-alts INT8 INT4`` adds E2 (KV sweep) and E3 (weight-KV separability) and spreads the KV precisions over E4/E5.
``generate_models.py`` defaults to the KV precisions the target backend can export (xnnpack, onnx: INT8, INT4).

    python scripts/cost_predictor/experiment_plan.py --out-dir experiments/cost_predictor
    python scripts/cost_predictor/experiment_plan.py --out-dir /tmp/plan --tiny --e4-count 10 --e5-count 4
"""

import argparse
import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LAYER_PATHS = {
    "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj", "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj", "gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}
# Execution order inside one block; precision transitions are counted along it.
EXECUTION_ORDER = PROJECTIONS
PRECISIONS = (16, 8, 4)
PRECISION_NAMES = {16: "FP16", 8: "INT8", 4: "INT4"}
REF_WEIGHT_BITS = 8
KV_LABELS = {"F16": 16, "F32": 32, "INT8": 8, "INT4": 4}
GROUP_SIZE = 128   # Appendix A: per-group, g=128, FP16 scales (used for the byte estimates)

# Estimated kernel family each weight precision runs on, per backend (Appendix A "kernel-family transitions").
KERNEL_FAMILY = {
    "tensorrt": {16: "fp16_gemm", 8: "fp16_gemm", 4: "int4_dequant"},   # INT8 weight-only folds to fp16 on TensorRT
    "onnx": {16: "fp16_gemm", 8: "fp16_gemm", 4: "int4_dequant"},
    "xnnpack": {16: "fp32_gemm", 8: "qd8_qc8w", 4: "qd8_qb4w"},
    "qnn": {16: "a16w16", 8: "a16w8", 4: "a16w4"},
}


@dataclass
class ModelDims:
    num_blocks: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int

    def shape(self, proj):
        """(in_features, out_features) of one projection."""
        q_out = self.heads * self.head_dim
        kv_out = self.kv_heads * self.head_dim
        return {
            "q_proj": (self.hidden, q_out), "k_proj": (self.hidden, kv_out), "v_proj": (self.hidden, kv_out),
            "o_proj": (q_out, self.hidden), "gate_proj": (self.hidden, self.intermediate),
            "up_proj": (self.hidden, self.intermediate), "down_proj": (self.intermediate, self.hidden),
        }[proj]

    def params(self, proj):
        i, o = self.shape(proj)
        return i * o

    def kv_elements_per_token(self):
        return 2 * self.num_blocks * self.kv_heads * self.head_dim


def dims_from_config(config) -> ModelDims:
    heads = config.num_attention_heads
    return ModelDims(
        num_blocks=config.num_hidden_layers, hidden=config.hidden_size, intermediate=config.intermediate_size,
        heads=heads, kv_heads=getattr(config, "num_key_value_heads", heads) or heads,
        head_dim=getattr(config, "head_dim", None) or config.hidden_size // heads,
    )


def tiny_config():
    """Small random-weight Qwen2 configuration for pipeline tests (no download)."""
    from transformers import Qwen2Config
    return Qwen2Config(vocab_size=1024, hidden_size=256, intermediate_size=512, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=16384,
                       tie_word_embeddings=True)


# ----------------------------------------------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class ConfigSpec:
    config_id: str
    experiment: str
    description: str
    weights: list              # [num_blocks][7] weight bits (16 / 8 / 4), column order = PROJECTIONS
    kv: str                    # "F16" | "INT8" | "INT4" (| "F32" for fp32 backends)
    tags: dict = field(default_factory=dict)
    weights_key: str = ""
    weights_owner: str = ""    # "<experiment dir>/<config_id>" holding the exported weights
    features: dict = field(default_factory=dict)


def weights_key(weights):
    return hashlib.sha1(json.dumps(weights).encode()).hexdigest()[:12]


def uniform(dims, bits):
    return [[bits] * len(PROJECTIONS) for _ in range(dims.num_blocks)]


def with_type(dims, proj, bits, base_bits=REF_WEIGHT_BITS):
    w = uniform(dims, base_bits)
    col = PROJECTIONS.index(proj)
    for row in w:
        row[col] = bits
    return w


def features(spec: ConfigSpec, dims: ModelDims, backends) -> dict:
    """f(c): precision counts per projection type, weight bytes, fraction of W4 layers, precision
    transitions, KV precision and estimated kernel-family transitions (Appendix A, group 2)."""
    counts = {p: {PRECISION_NAMES[b]: 0 for b in PRECISIONS} for p in PROJECTIONS}
    bytes_by_prec = {PRECISION_NAMES[b]: 0.0 for b in PRECISIONS}
    n_params = 0
    sequence = []
    for row in spec.weights:
        for proj, bits in zip(PROJECTIONS, row):
            counts[proj][PRECISION_NAMES[bits]] += 1
            params = dims.params(proj)
            n_params += params
            scale_bytes = 0 if bits == 16 else 2 * params / GROUP_SIZE
            bytes_by_prec[PRECISION_NAMES[bits]] += params * bits / 8 + scale_bytes
            sequence.append(bits)
    total_bytes = sum(bytes_by_prec.values())
    directed = {}
    for a, b in zip(sequence, sequence[1:]):
        if a != b:
            key = f"{PRECISION_NAMES[a]}->{PRECISION_NAMES[b]}"
            directed[key] = directed.get(key, 0) + 1
    kv_bits = KV_LABELS[spec.kv]
    kv_bytes_per_token = dims.kv_elements_per_token() * kv_bits / 8
    return {
        "precision_counts_per_type": counts,
        "num_layers": len(sequence),
        "weight_bytes_total": total_bytes,
        "weight_bytes_by_precision": bytes_by_prec,
        "effective_avg_weight_bits": 8 * total_bytes / n_params,
        "fraction_w4_layers": sequence.count(4) / len(sequence),
        "precision_transitions": sum(directed.values()),
        "directed_transitions": directed,
        "kv_precision": spec.kv,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kernel_family_transitions": {
            b: sum(KERNEL_FAMILY[b][x] != KERNEL_FAMILY[b][y] for x, y in zip(sequence, sequence[1:]))
            for b in backends
        },
    }


def sample_mixed_weights(dims: ModelDims, rng: random.Random, strategy: str):
    """One heterogeneous per-layer assignment. The strategies mimic what a compression search produces."""
    L, T = dims.num_blocks, len(PROJECTIONS)
    pick = lambda: rng.choice(PRECISIONS)
    if strategy == "iid":
        probs = [rng.random() + 0.1 for _ in PRECISIONS]
        return [[rng.choices(PRECISIONS, probs)[0] for _ in range(T)] for _ in range(L)]
    if strategy == "per_type":
        row = [pick() for _ in range(T)]
        return [list(row) for _ in range(L)]
    if strategy == "depth_segments":
        cuts = sorted(rng.sample(range(1, L), min(2, L - 1)))
        bounds = [0] + cuts + [L]
        w = []
        for lo, hi in zip(bounds, bounds[1:]):
            attn, mlp = pick(), pick()
            w += [[attn] * 4 + [mlp] * 3 for _ in range(lo, hi)]
        return w
    if strategy == "alternating":
        a, b = pick(), pick()
        return [[a if i % 2 == 0 else b] * T for i in range(L)]
    if strategy == "sparse_deviation":
        w = uniform(dims, REF_WEIGHT_BITS)
        for _ in range(rng.randint(2, max(2, L * T // 4))):
            w[rng.randrange(L)][rng.randrange(T)] = rng.choice([p for p in PRECISIONS if p != REF_WEIGHT_BITS])
        return w
    raise ValueError(strategy)


MIXED_STRATEGIES = ["iid", "per_type", "depth_segments", "alternating", "sparse_deviation"]


def sample_mixed(dims, count, forbidden_keys, kv_options, rng, prefix, experiment):
    """``count`` distinct heterogeneous configurations whose weights are not in ``forbidden_keys``
    (so E4 is disjoint from the controlled sweeps and E5 from E4)."""
    specs, seen, attempts = [], set(forbidden_keys), 0
    while len(specs) < count:
        attempts += 1
        if attempts > 5000 * max(count, 1):
            raise RuntimeError(f"cannot sample {count} distinct configurations (got {len(specs)})")
        # Strategies take turns per attempt, not per accepted configuration, so one whose space runs out
        # (alternating has only 6 assignments) is skipped instead of blocking the others.
        strategy = MIXED_STRATEGIES[(attempts - 1) % len(MIXED_STRATEGIES)]
        weights = sample_mixed_weights(dims, rng, strategy)
        flat = {b for row in weights for b in row}
        key = weights_key(weights)
        if len(flat) < 2 or key in seen:       # uniform configurations belong to E0 / E3
            continue
        seen.add(key)
        i = len(specs) + 1
        kv = kv_options[(i - 1) % len(kv_options)]
        specs.append(ConfigSpec(
            config_id=f"{prefix}{i:03d}", experiment=experiment, weights=weights, kv=kv,
            description=f"heterogeneous mixed configuration ({strategy})", tags={"strategy": strategy},
            weights_key=key,
        ))
    return specs


# ----------------------------------------------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class Experiment:
    dirname: str
    name: str
    purpose: str
    quantities: str
    configs: list = field(default_factory=list)
    reference_config: str | None = None    # config the deviations are measured against ("<dir>/<id>")


def build_experiments(dims, args, backends):
    kv_ref = args.kv_ref
    kv_alts = args.kv_alts
    ref_w = uniform(dims, REF_WEIGHT_BITS)
    ref_key = weights_key(ref_w)

    e0 = Experiment("E0_reference", "E0", "Reference surface: uniform INT8 weights + reference KV precision",
                    "T_ref^pre(P), t_ref^dec(j) for j<=G_max, energy equivalents",
                    [ConfigSpec("E0_ref", "E0", "uniform INT8 weights, reference KV", ref_w, kv_ref, weights_key=ref_key)])

    e1 = Experiment("E1_weight_sweep", "E1",
                    "Weight-structured term: change all layers of one projection type tau to precision b "
                    "(KV = reference, other layers = reference INT8)",
                    "delta_{tau,b}^pre(P), delta_{tau,b}^dec(P, G) for all G<=G_max",
                    reference_config="E0_reference/E0_ref")
    for proj in PROJECTIONS:
        for b in args.weight_alts:
            w = with_type(dims, proj, b)
            e1.configs.append(ConfigSpec(
                f"E1_{proj}_{PRECISION_NAMES[b]}", "E1", f"all {proj} layers at {PRECISION_NAMES[b]}, rest INT8", w, kv_ref,
                tags={"tau": proj, "b": PRECISION_NAMES[b], "N_tau": dims.num_blocks}, weights_key=weights_key(w)))

    # E2/E3 need artifacts with a quantised KV cache: without KV alternatives the plan is weights-only.
    kv_experiments = []
    if kv_alts:
        e2 = Experiment("E2_kv_sweep", "E2",
                        "KV structured term: reference weights, change only the KV-cache precision",
                        "DeltaT_KV^pre(b_KV, P), DeltaT_KV^dec(b_KV, P, G)", reference_config="E0_reference/E0_ref")
        for kv in kv_alts:
            e2.configs.append(ConfigSpec(f"E2_kv{kv}", "E2", f"reference INT8 weights, KV {kv}", ref_w, kv,
                                         tags={"b_KV": kv}, weights_key=ref_key))

        w4 = uniform(dims, 4)
        e3 = Experiment("E3_separability", "E3", "Weight-KV separability: repeat the KV sweep on uniform W4 weights",
                        "S_KV(b_KV, P) = DeltaT_KV(b_KV; W4) - DeltaT_KV(b_KV; W8)  (needs the E2 + E3 runs)",
                        reference_config="E0_reference/E0_ref")
        for kv in [kv_ref] + list(kv_alts):
            e3.configs.append(ConfigSpec(f"E3_w4_kv{kv}", "E3", f"uniform W4 weights, KV {kv}", w4, kv,
                                         tags={"b_KV": kv}, weights_key=weights_key(w4)))
        kv_experiments = [e2, e3]

    controlled = [c.weights_key for e in [e0, e1, *kv_experiments] for c in e.configs]
    rng = random.Random(args.seed)
    kv_options = [kv_ref] + list(kv_alts)
    e4 = Experiment("E4_residual", "E4",
                    "Learn the residual: heterogeneous mixed configurations, disjoint from the controlled sweeps "
                    "(compile + prefill + G_max decode steps; record execution features)",
                    "r^pre, r^dec, execution features e (compiled-graph events) and f(c)")
    e4.configs = sample_mixed(dims, args.e4_count, controlled, kv_options, rng, "C", "E4")
    e5 = Experiment("E5_validation", "E5", "Final validation: held-out heterogeneous configurations",
                    "error of the complete predictor, reported for G in " + str(args.eval_g))
    e5.configs = sample_mixed(dims, args.e5_count, controlled + [c.weights_key for c in e4.configs], kv_options,
                              rng, "H", "E5")

    experiments = [e for e in [e0, e1, *kv_experiments, e4, e5] if e.configs]   # e.g. --e5-count 0 drops E5
    owners = {}
    for e in experiments:
        for c in e.configs:
            owners.setdefault(c.weights_key, f"{e.dirname}/{c.config_id}")
            c.weights_owner = owners[c.weights_key]
            c.features = features(c, dims, backends)
    return experiments


# ----------------------------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------------------------

def feature_columns(f, backends):
    """The numeric features f(c) of one configuration as flat columns (the residual model's input)."""
    row = {"weight_bytes": round(f["weight_bytes_total"]), "avg_weight_bits": round(f["effective_avg_weight_bits"], 3),
           "frac_w4": round(f["fraction_w4_layers"], 4), "precision_transitions": f["precision_transitions"],
           "kv_bytes_per_token": f["kv_bytes_per_token"]}
    for proj in PROJECTIONS:
        for b in PRECISIONS:
            row[f"n_{proj}_{PRECISION_NAMES[b]}"] = f["precision_counts_per_type"][proj][PRECISION_NAMES[b]]
    for a in PRECISIONS:
        for b in PRECISIONS:
            if a != b:
                key = f"{PRECISION_NAMES[a]}->{PRECISION_NAMES[b]}"
                row[f"trans_{PRECISION_NAMES[a]}_to_{PRECISION_NAMES[b]}"] = f["directed_transitions"].get(key, 0)
    for b in backends:
        row[f"kernel_transitions_{b}"] = f["kernel_family_transitions"][b]
    return row


def flat_row(exp: Experiment, c: ConfigSpec, backends):
    return {"experiment": exp.name, "config_id": c.config_id, "b_KV": c.kv, "description": c.description,
            "weights_owner": c.weights_owner, "weights_key": c.weights_key, **feature_columns(c.features, backends)}


RESULT_COLUMNS = ["config_id", "backend", "P", "G_max", "b_KV", "prefill_ms", "prefill_energy_j", "decode_trace_ms",
                  "decode_energy_trace_j", "peak_memory_bytes", "thermal_throttled", "notes"]


def write_experiment(out: Path, exp: Experiment, args, backends):
    d = out / exp.dirname
    (d / "models").mkdir(parents=True, exist_ok=True)
    meta = {"experiment": exp.name, "purpose": exp.purpose, "quantities": exp.quantities,
            "reference_config": exp.reference_config, "prompt_lengths": args.prompt_lengths, "g_max": args.g_max,
            "eval_g": args.eval_g, "num_configs": len(exp.configs),
            "configs": [c.config_id for c in exp.configs]}
    (d / "experiment.json").write_text(json.dumps(meta, indent=2))
    for c in exp.configs:
        cd = d / "models" / c.config_id
        cd.mkdir(exist_ok=True)
        (cd / "spec.json").write_text(json.dumps(asdict(c), indent=2))
    rows = [flat_row(exp, c, backends) for c in exp.configs]
    with open(d / "configs.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(d / "results_template.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for c in exp.configs:
            for backend in backends:
                for p in args.prompt_lengths:
                    writer.writerow({"config_id": c.config_id, "backend": backend, "P": p, "G_max": args.g_max,
                                     "b_KV": c.kv})
    return rows


def add_plan_args(parser):
    parser.add_argument("--out-dir", default="experiments/cost_predictor")
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--tiny", action="store_true", help="tiny random Qwen2 (pipeline test, no download)")
    parser.add_argument("--backends", nargs="+", default=["xnnpack"], choices=sorted(KERNEL_FAMILY),
                        help="targets the kernel-transition features and result templates are written for")
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[128, 512, 2048, 8192], help="the set P")
    parser.add_argument("--g-max", type=int, default=128)
    parser.add_argument("--eval-g", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--weight-alts", nargs="+", type=int, default=[16, 4], choices=[16, 8, 4, 6, 3, 2],
                        help="alternative weight precisions b of E1 (16 = FP16)")
    parser.add_argument("--kv-ref", default="F16", choices=["F16", "F32"], help="backend default KV precision")
    parser.add_argument("--kv-alts", nargs="*", default=[], choices=["INT8", "INT4"],
                        help="alternative KV-cache precisions, adds E2/E3 (default: none, a weights-only plan; "
                             "generate_models.py defaults to what the backend can export)")
    parser.add_argument("--e4-count", type=int, default=40, help="mixed configurations to learn the residual")
    parser.add_argument("--e5-count", type=int, default=15, help="held-out configurations")
    parser.add_argument("--seed", type=int, default=0)


def load_dims(args) -> ModelDims:
    if args.tiny:
        return dims_from_config(tiny_config())
    from transformers import AutoConfig
    return dims_from_config(AutoConfig.from_pretrained(args.model_name))


def build_plan(args):
    """Create all experiments and write the folders; returns ``(experiments, dims)``."""
    dims = load_dims(args)
    backends = list(args.backends)
    experiments = build_experiments(dims, args, backends)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for exp in experiments:
        all_rows += write_experiment(out, exp, args, backends)
    fieldnames = list(dict.fromkeys(k for r in all_rows for k in r))
    with open(out / "index_configs.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    plan = {"model": "tiny-random-qwen2" if args.tiny else args.model_name, "dims": asdict(dims),
            "projection_types": PROJECTIONS, "reference": {"weights": f"INT{REF_WEIGHT_BITS}", "kv": args.kv_ref},
            "kv_alternatives": list(args.kv_alts),
            "prompt_lengths": args.prompt_lengths, "g_max": args.g_max, "eval_g": args.eval_g, "seed": args.seed,
            "group_size": GROUP_SIZE, "backends": backends,
            "experiments": {e.name: {"dir": e.dirname, "configs": len(e.configs)} for e in experiments},
            "unique_weight_artifacts": len({c.weights_key for e in experiments for c in e.configs})}
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    return experiments, dims


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_plan_args(parser)
    args = parser.parse_args()
    experiments, _ = build_plan(args)
    for e in experiments:
        print(f"{e.dirname:24s} {len(e.configs):3d} configurations")
    plan = json.loads((Path(args.out_dir) / "plan.json").read_text())
    print(f"unique weight artifacts: {plan['unique_weight_artifacts']} (x {len(args.prompt_lengths)} prompt lengths "
          f"when graphs are per-P)\nwritten to {args.out_dir}")


if __name__ == "__main__":
    main()
