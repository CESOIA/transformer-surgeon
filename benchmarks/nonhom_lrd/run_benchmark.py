#!/usr/bin/env python3
"""
Non-homogeneous LRD benchmark on Qwen2-1.5B.

Shows that varying the compression distribution across transformer layers
at the same total FLOPs budget yields different perplexity tradeoffs.

Experiment setup
----------------
- Baseline   : float16, no compression
- Homogeneous: SVD-LLM-v2, rank = 25% * min(out, in) for every weight
               (k_proj and v_proj are left uncompressed throughout)
- Non-hom k  : the k "units" with the highest LRD error ||W - W_lrd||_F
               are restored to full rank; remaining compressed layers
               share the same total MACs budget via proportional rank
               scaling.  Error is measured at the base rank before any
               rank rescaling.

Metrics stored per experiment
------------------------------
  ppl           WikiText-2 test-set perplexity
  ops_ratio     total linear-layer MACs / float16 baseline MACs
  params_ratio  total linear-layer params / float16 baseline params

Output
------
  results/calib_state.pt         - calibration statistics (reused across runs)
  results/baseline_float16/      - saved HF model
  results/nonhom_XX_units/       - saved HF model for each experiment
  results/results.json           - all metrics
  results/ppl_vs_budget.png      - summary plot

Usage
-----
  python run_benchmark.py
  python run_benchmark.py --decompress-steps 0 2 4 8 16
  python run_benchmark.py --num-calibration-samples 64 --num-test-samples 100
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress
from transformersurgeon.blocks import LinearCompressed


# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2-1.5B"
LRD_RATIO = 0.4           # rank = 40% of min(out, in)
RANDOM_SEED = 42
SEQ_LEN = 2048
NUM_CALIBRATION_SAMPLES = 32
NUM_TEST_SAMPLES = 50
SKIP_LAYERS = frozenset({"k_proj", "v_proj"})

# Projection layer names belonging to each block type
MHA_PROJECTIONS = ("self_attn.q_proj", "self_attn.o_proj")
MLP_PROJECTIONS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
ALL_PROJECTIONS = MHA_PROJECTIONS + MLP_PROJECTIONS + (
    "self_attn.k_proj", "self_attn.v_proj")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Non-homogeneous LRD benchmark")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--lrd-ratio", type=float, default=LRD_RATIO,
                   help="Fraction of min(out,in) used as base rank")
    p.add_argument("--num-calibration-samples", type=int, default=NUM_CALIBRATION_SAMPLES)
    p.add_argument("--num-test-samples", type=int, default=NUM_TEST_SAMPLES)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--decompress-steps", type=int, nargs="+", default=None,
                   help="Numbers of units to decompress; default: 0 2 4 8 12 16")
    p.add_argument("--skip-calibration", action="store_true",
                   help="Reuse calib_state.pt if it already exists")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip the float16 baseline evaluation")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "results")
    p.add_argument("--device", default=None,
                   help="Force a specific device, e.g. 'cuda:0' or 'cpu'")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────
def build_loaders(tokenizer, num_calib: int, num_test: int, seq_len: int):
    """Return (calibration DataLoader, test DataLoader) from WikiText-2."""
    train_raw = load_dataset(
        "EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="train"
    )
    test_raw = load_dataset(
        "EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", split="test"
    )

    def _encode(ds):
        texts = [t for t in ds["page"] if isinstance(t, str) and t.strip()]
        return tokenizer("\n\n".join(texts), truncation=False,
                         padding=False, return_attention_mask=False)["input_ids"]

    calib_ids = _encode(train_raw)
    test_ids  = _encode(test_raw)

    rng = random.Random(RANDOM_SEED)

    # Random non-overlapping windows for calibration
    calib_examples = []
    for _ in range(min(num_calib, len(calib_ids) // seq_len)):
        start = rng.randint(0, len(calib_ids) - seq_len - 1)
        calib_examples.append({
            "input_ids":      torch.tensor(calib_ids[start:start + seq_len], dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
        })

    # Sequential non-overlapping windows for test (standard wikitext PPL protocol)
    test_examples = []
    for i in range(min(num_test, len(test_ids) // seq_len)):
        s = i * seq_len
        test_examples.append({
            "input_ids":      torch.tensor(test_ids[s:s + seq_len], dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
        })

    return (DataLoader(calib_examples, batch_size=1, shuffle=False),
            DataLoader(test_examples,  batch_size=1, shuffle=False))


# ── Perplexity evaluation ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_perplexity(model, test_loader, device) -> float:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        out = model(input_ids=input_ids, labels=input_ids.clone())
        total_nll   += out.loss.float().item() * input_ids.numel()
        total_tokens += input_ids.numel()
    return math.exp(total_nll / total_tokens)


# ── Layer shape / rank analysis ────────────────────────────────────────────────
def compute_layer_info(model, lrd_ratio: float, skip_layers):
    """
    Walk the model and collect shape info for every compressible linear projection.

    Returns dict keyed by full dotted path:
      {"out": int, "in": int, "layer_name": str, "skip": bool,
       "R_base": int|None, "ops_full": int, "ops_base": int,
       "params_full": int, "params_base": int}
    """
    info = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not any(p in name for p in ALL_PROJECTIONS):
            continue
        if "lm_head" in name or "embed" in name:
            continue

        layer_name = name.split(".")[-1]
        out_f = mod.out_features
        in_f  = mod.in_features
        skip  = layer_name in skip_layers
        R     = max(1, int(min(out_f, in_f) * lrd_ratio)) if not skip else None

        info[name] = {
            "out":          out_f,
            "in":           in_f,
            "layer_name":   layer_name,
            "skip":         skip,
            "R_base":       R,
            "ops_full":     out_f * in_f,
            "ops_base":     R * (out_f + in_f) if not skip else out_f * in_f,
            "params_full":  out_f * in_f,
            "params_base":  R * (out_f + in_f) if not skip else out_f * in_f,
        }
    return info


def compute_model_ops_params(model):
    """
    Compute total MACs and parameter count for all linear layers,
    accounting for LRD if active (two-factor form).
    """
    total_mac = 0
    total_par = 0
    visited   = set()
    for _, mod in model.named_modules():
        if id(mod) in visited:
            continue
        visited.add(id(mod))

        if isinstance(mod, LinearCompressed):
            out_f, in_f = mod.out_features, mod.in_features
            if isinstance(mod.rank, int):
                r = mod.rank
                total_mac += r * (in_f + out_f)
                total_par += r * (in_f + out_f)
            else:
                total_mac += out_f * in_f
                total_par += out_f * in_f
            if mod.bias is not None:
                total_par += out_f

        elif isinstance(mod, torch.nn.Linear):
            out_f, in_f = mod.weight.shape
            total_mac += out_f * in_f
            total_par += out_f * in_f
            if mod.bias is not None:
                total_par += out_f

    return total_mac, total_par


# ── Decompression unit helpers ─────────────────────────────────────────────────
def build_decompression_units(num_layers: int):
    """
    Return the unordered list of all compression units.
    Each entry: {"type": "mha"|"mlp", "layer": int, "path_suffixes": list[str]}
    """
    units = []
    for i in range(num_layers):
        units.append({"type": "mha", "layer": i,
                      "path_suffixes": list(MHA_PROJECTIONS)})
        units.append({"type": "mlp", "layer": i,
                      "path_suffixes": list(MLP_PROJECTIONS)})
    return units


@torch.no_grad()
def compute_lrd_errors(model, layer_info: dict) -> dict:
    """
    Compute the *relative* LRD truncation error for every compressible layer:

        relative_error = ||W - W_R||_F / ||W||_F
                       = sqrt( sum_{i>R} sigma_i^2 / sum_all sigma_i^2 )

    This is the fraction of the weight's Frobenius energy lost to truncation.
    Using relative (not absolute) error makes layers comparable across shapes:
    an (8960x1536) MLP projection and a (1536x1536) MHA projection are judged
    on the same scale regardless of their size.

    SVD values are computed on CPU in float32 for numerical stability.
    Returns {layer_path: float}.  Skipped layers get error 0.0.
    """
    errors = {}
    for name, mod in model.named_modules():
        if name not in layer_info:
            continue
        v = layer_info[name]
        if v["skip"]:
            errors[name] = 0.0
            continue
        R = v["R_base"]
        W = mod.weight.detach().cpu().float()
        S2 = torch.linalg.svdvals(W).pow_(2)  # sigma_i^2, descending
        total_energy = S2.sum().item()
        truncated_energy = S2[R:].sum().item()
        errors[name] = math.sqrt(truncated_energy / total_energy) if total_energy > 0 else 0.0
        del W, S2
    return errors


def sort_units_by_lrd_error(units: list, layer_errors: dict,
                             layer_info: dict) -> list:
    """
    Re-order units by descending mean relative LRD error across their
    non-skipped projections.  Units where a larger fraction of weight energy
    is lost to truncation are restored to full rank first.
    """
    def unit_error(unit):
        errs = [
            layer_errors.get(p, 0.0)
            for p in unit_full_paths(unit)
            if not layer_info.get(p, {}).get("skip", True)
        ]
        return sum(errs) / len(errs) if errs else 0.0
    return sorted(units, key=unit_error, reverse=True)


def unit_full_paths(unit) -> list:
    i = unit["layer"]
    return [f"model.layers.{i}.{s}" for s in unit["path_suffixes"]]


def compute_experiment_ranks(layer_info: dict, units: list, n_decompressed: int):
    """
    Compute per-layer ranks for the experiment that decompresses the first
    n_decompressed units and scales remaining compressed layers proportionally
    so total ops == homogeneous LRD budget.

    Returns (ranks_dict, scale_factor) or None if infeasible (budget exceeded).
    ranks_dict: {layer_path: int | "full"}
    """
    # Total ops budget under uniform compression — skip layers run at full rank
    # so they must not influence the budget allocated to compressible layers.
    total_budget = sum(v["ops_base"] for v in layer_info.values() if not v["skip"])

    decompressed_paths = {
        p
        for unit in units[:n_decompressed]
        for p in unit_full_paths(unit)
    }

    # Ops consumed by full-rank decompressed layers
    ops_from_decompressed = sum(
        v["ops_full"]
        for path, v in layer_info.items()
        if path in decompressed_paths and not v["skip"]
    )

    remaining_budget = total_budget - ops_from_decompressed
    ops_remaining_base = sum(
        v["ops_base"]
        for path, v in layer_info.items()
        if path not in decompressed_paths and not v["skip"]
    )

    if remaining_budget <= 0 or ops_remaining_base <= 0:
        return None  # budget exceeded → infeasible

    scale = remaining_budget / ops_remaining_base

    ranks = {}
    for path, v in layer_info.items():
        if v["skip"] or path in decompressed_paths:
            ranks[path] = "full"
        else:
            ranks[path] = max(1, round(v["R_base"] * scale))

    return ranks, scale


# ── Calibration ────────────────────────────────────────────────────────────────
def run_calibration_and_save(model_name, calib_loader, layer_info,
                              state_path: Path, device):
    """
    Load a fresh model, run one calibration pass to collect SVD-LLM-v2
    covariance statistics for every compressible layer, and save the state.
    Calibration statistics are independent of rank, so they can be reused.
    """
    print("  Loading model for calibration...")
    model = Qwen2ForCausalLMCompress.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()

    manager = Qwen2CompressionSchemesManager(model)
    manager.set("lrd", "method", "svd-llm-v2", criteria="all")

    # Set base ranks to trigger calibration targets; K/V stay at "full"
    for path, v in layer_info.items():
        if v["skip"]:
            manager.set("lrd", "rank", "full", criteria=path)
        else:
            manager.set("lrd", "rank", v["R_base"], criteria=path)

    manager.set_calibration_data(calib_loader)
    print("  Collecting covariance statistics (one forward pass per batch)...")
    manager.run_calibration(
        criteria="all", device=device, offload_to_cpu=True, show_progress=True
    )

    state_path.parent.mkdir(parents=True, exist_ok=True)
    manager.save_state(str(state_path))
    print(f"  Calibration state saved to {state_path}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ── Single experiment ──────────────────────────────────────────────────────────
def run_experiment(model_name, ranks: dict, state_path: Path,
                   test_loader, device, exp_dir: Path):
    """
    Reload a fresh model, apply compression at the specified ranks
    (using cached calibration), evaluate PPL, and save in HF format.

    Returns (ppl, actual_macs, actual_params).
    """
    print("  Loading fresh model...")
    model = Qwen2ForCausalLMCompress.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()

    manager = Qwen2CompressionSchemesManager(model)

    # Set method everywhere; rank per layer from the experiment spec
    manager.set("lrd", "method", "svd-llm-v2", criteria="all")
    for path, rank in ranks.items():
        manager.set("lrd", "rank", rank, criteria=path)

    # Load pre-computed covariance — skips re-running the calibration pass
    manager.load_state(str(state_path))

    print("  Applying compression...")
    manager.apply(hard=False, criteria="all", device=device, offload_to_cpu=True)

    print("  Evaluating perplexity on WikiText-2 test set...")
    ppl = evaluate_perplexity(model, test_loader, device)

    actual_macs, actual_params = compute_model_ops_params(model)

    print(f"  Saving model to {exp_dir} ...")
    exp_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(exp_dir))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ppl, actual_macs, actual_params


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(results: list, out_dir: Path):
    baseline = next((r for r in results if r["name"] == "baseline_float16"), None)
    compressed = [r for r in results if r["name"] != "baseline_float16"]

    if not compressed:
        print("No compressed results to plot.")
        return

    ns   = [r["n_decompressed"] for r in compressed]
    ppls = [r["ppl"]            for r in compressed]
    ops  = [r["ops_ratio"]      for r in compressed]
    pars = [r["params_ratio"]   for r in compressed]

    colors = cm.coolwarm(np.linspace(0, 1, len(ns))) if len(ns) > 1 else ["steelblue"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Non-homogeneous LRD — Qwen2-1.5B | SVD-LLM-v2 | WikiText-2",
        fontsize=13, fontweight="bold"
    )

    # ── Plot 1: PPL vs decompression front ──────────────────────────────────
    ax = axes[0]
    ax.plot(ns, ppls, "o-", color="steelblue", linewidth=2, markersize=7, zorder=3)
    for n, ppl, c in zip(ns, ppls, colors):
        ax.scatter([n], [ppl], color=c, s=80, zorder=4)
        ax.annotate(f"{ppl:.1f}", (n, ppl), textcoords="offset points",
                    xytext=(4, 5), fontsize=8)
    if baseline:
        ax.axhline(baseline["ppl"], color="dimgray", linestyle="--",
                   label=f"float16 baseline ({baseline['ppl']:.1f})")
    hom = next((r for r in compressed if r["n_decompressed"] == 0), None)
    if hom:
        ax.scatter([0], [hom["ppl"]], marker="D", color="darkorange",
                   s=90, zorder=5, label=f"Homogeneous LRD-25 ({hom['ppl']:.1f})")
    ax.set_xlabel("Decompressed units from front\n(0=MHA₀ → 1=MLP₀ → 2=MHA₁ → …)")
    ax.set_ylabel("Perplexity ↓")
    ax.set_title("PPL vs. decompression front\n(same total MACs budget)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: PPL vs ops ratio ─────────────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(ops, ppls, c=ns, cmap="coolwarm", s=90, zorder=4)
    plt.colorbar(sc, ax=ax, label="# decompressed units")
    for op, ppl, n in zip(ops, ppls, ns):
        ax.annotate(f"n={n}", (op, ppl), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
    if baseline:
        ax.scatter([baseline["ops_ratio"]], [baseline["ppl"]],
                   marker="*", color="gold", s=160, zorder=5, label="float16")
        ax.legend(fontsize=8)
    ax.set_xlabel("Ops ratio  (compressed MACs / float16 MACs)")
    ax.set_ylabel("Perplexity ↓")
    ax.set_title("PPL vs. compute budget\n(iso-budget comparison)")
    ax.grid(True, alpha=0.3)

    # ── Plot 3: PPL vs params ratio ──────────────────────────────────────────
    ax = axes[2]
    sc = ax.scatter(pars, ppls, c=ns, cmap="coolwarm", s=90, zorder=4)
    plt.colorbar(sc, ax=ax, label="# decompressed units")
    for pr, ppl, n in zip(pars, ppls, ns):
        ax.annotate(f"n={n}", (pr, ppl), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
    if baseline:
        ax.scatter([baseline["params_ratio"]], [baseline["ppl"]],
                   marker="*", color="gold", s=160, zorder=5, label="float16")
        ax.legend(fontsize=8)
    ax.set_xlabel("Params ratio  (compressed params / float16 params)")
    ax.set_ylabel("Perplexity ↓")
    ax.set_title("PPL vs. memory footprint\n(iso-budget comparison)")
    ax.grid(True, alpha=0.3)

    # Clamp y-axis to compressed-model range so the float16 baseline
    # (which is an outlier in PPL space) doesn't squash the interesting data.
    ppl_lo = min(ppls)
    ppl_hi = max(ppls)
    margin = max((ppl_hi - ppl_lo) * 0.15, 1.0)
    for ax in axes:
        ax.set_ylim(ppl_lo - margin, ppl_hi + margin)

    plt.tight_layout()
    plot_path = out_dir / "ppl_vs_budget.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {plot_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device      : {device}")
    print(f"Model       : {args.model_name}")
    print(f"LRD ratio   : {args.lrd_ratio} (rank = {args.lrd_ratio*100:.0f}% of min dim)")

    out_dir        = args.output_dir
    calib_state_pt = out_dir / "calib_state.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer & data ─────────────────────────────────────────────────────
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(str(out_dir / "tokenizer"))

    print(f"Building WikiText-2 loaders  "
          f"(calib={args.num_calibration_samples}, test={args.num_test_samples}, "
          f"seq={args.seq_len})...")
    calib_loader, test_loader = build_loaders(
        tokenizer, args.num_calibration_samples,
        args.num_test_samples, args.seq_len
    )
    print(f"  Calibration batches: {len(calib_loader)}")
    print(f"  Test batches       : {len(test_loader)}")

    # ── Analyse model structure ───────────────────────────────────────────────
    print(f"\nAnalysing model layer shapes and LRD errors...")
    probe = Qwen2ForCausalLMCompress.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    )
    layer_info  = compute_layer_info(probe, args.lrd_ratio, SKIP_LAYERS)
    baseline_macs, baseline_params = compute_model_ops_params(probe)
    num_layers  = probe.config.num_hidden_layers
    print(f"  Computing SVD truncation errors for all compressible layers "
          f"(this runs on CPU)...")
    layer_errors = compute_lrd_errors(probe, layer_info)
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    compressible = [(p, v) for p, v in layer_info.items() if not v["skip"]]
    # Budget excludes skip layers (k_proj, v_proj stay at full rank always)
    total_budget = sum(v["ops_base"] for _, v in layer_info.items() if not v["skip"])
    # MACs / params from layers NOT in layer_info (lm_head, etc.) — constant across experiments
    macs_not_tracked  = baseline_macs   - sum(v["ops_full"]   for v in layer_info.values())
    params_not_tracked = baseline_params - sum(v["params_full"] for v in layer_info.values())

    print(f"  Hidden layers      : {num_layers}")
    print(f"  Compressible layers: {len(compressible)}")
    print(f"  Baseline MACs      : {baseline_macs/1e9:.3f} G")
    print(f"  Baseline params    : {baseline_params/1e6:.2f} M")
    print(f"  Homogeneous budget : {total_budget/1e9:.3f} G MACs "
          f"({100*total_budget/baseline_macs:.1f}% of projection MACs)")

    # ── Decompression units & experiment schedule ─────────────────────────────
    units = build_decompression_units(num_layers)
    units = sort_units_by_lrd_error(units, layer_errors, layer_info)

    steps = sorted(set(args.decompress_steps)) if args.decompress_steps \
        else [0, 2, 4, 8, 12, 16]

    print(f"\nUnit restoration order (highest LRD error first):")
    max_step = max(steps) if steps else 0
    for i, u in enumerate(units[:max_step + 1]):
        paths = [p for p in unit_full_paths(u) if not layer_info.get(p, {}).get("skip", True)]
        err = sum(layer_errors.get(p, 0.0) for p in paths)
        print(f"  [{i:3d}] {u['type'].upper()} layer={u['layer']}  "
              f"err={err:.4f}")

    print(f"\n{'='*72}")
    print(f"{'Experiment design':^72}")
    print(f"{'='*72}")
    print(f"{'n units':>8}  {'scale':>7}  "
          f"{'rank_remaining':>14}  {'MACs ratio':>10}  {'params ratio':>12}")
    print(f"{'-'*72}")

    experiments = []
    for n in steps:
        if n > len(units):
            print(f"  n={n:3d}  skipped (exceeds total units={len(units)})")
            continue
        result = compute_experiment_ranks(layer_info, units, n)
        if result is None:
            print(f"  n={n:3d}  INFEASIBLE (budget exceeded)")
            continue
        ranks, scale = result

        # Theoretical total ops / params = layer_info contribution + constant remainder
        # (lm_head and any layer not in layer_info are unchanged across experiments)
        theory_macs = (
            sum(
                ranks[p] * (v["out"] + v["in"])
                if isinstance(ranks.get(p), int) else v["ops_full"]
                for p, v in layer_info.items()
            )
            + macs_not_tracked
        )
        theory_params = (
            sum(
                ranks[p] * (v["out"] + v["in"])
                if isinstance(ranks.get(p), int) else v["params_full"]
                for p, v in layer_info.items()
            )
            + params_not_tracked
        )

        # Representative remaining rank (for non-square layers min-dim is 1536)
        sample_rank = next(
            (ranks[p] for p, v in layer_info.items()
             if not v["skip"] and p not in {
                 pp for u in units[:n] for pp in unit_full_paths(u)
             } and isinstance(ranks.get(p), int)),
            "N/A"
        )

        print(f"  n={n:3d}    {scale:7.3f}  {str(sample_rank):>14}  "
              f"{100*theory_macs/baseline_macs:9.1f}%  "
              f"{100*theory_params/baseline_params:11.1f}%")

        experiments.append({
            "n_decompressed":      n,
            "name":                f"nonhom_{n:02d}_units",
            "ranks":               ranks,
            "scale":               scale,
            "theory_macs":         theory_macs,
            "theory_params":       theory_params,
            "theory_ops_ratio":    theory_macs / baseline_macs,
            "theory_params_ratio": theory_params / baseline_params,
        })
    print(f"{'='*72}")

    # ── Calibration (run once, reuse for all experiments) ────────────────────
    if calib_state_pt.exists() and args.skip_calibration:
        print(f"\nReusing existing calibration state: {calib_state_pt}")
    else:
        print(f"\n{'─'*60}")
        print("Running calibration pass (SVD-LLM-v2 covariance collection)...")
        run_calibration_and_save(
            args.model_name, calib_loader, layer_info, calib_state_pt, device
        )

    # ── Results accumulator ───────────────────────────────────────────────────
    results = []

    # ── Baseline: float16 ─────────────────────────────────────────────────────
    if not args.skip_baseline:
        print(f"\n{'─'*60}")
        print("Experiment: baseline_float16")
        base_model = Qwen2ForCausalLMCompress.from_pretrained(
            args.model_name, torch_dtype=torch.float16
        ).to(device).eval()
        base_ppl = evaluate_perplexity(base_model, test_loader, device)
        base_macs, base_par = compute_model_ops_params(base_model)
        print(f"  PPL = {base_ppl:.2f}")
        base_model.save_pretrained(str(out_dir / "baseline_float16"))
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        results.append({
            "name":          "baseline_float16",
            "n_decompressed": None,
            "ppl":           base_ppl,
            "ops_ratio":     base_macs  / baseline_macs,
            "params_ratio":  base_par   / baseline_params,
            "theory_ops_ratio":    1.0,
            "theory_params_ratio": 1.0,
        })

    # ── Compressed experiments ────────────────────────────────────────────────
    for exp in experiments:
        print(f"\n{'─'*60}")
        print(f"Experiment: {exp['name']}  "
              f"(n={exp['n_decompressed']} units decompressed, "
              f"scale={exp['scale']:.3f})")

        ppl, actual_macs, actual_params = run_experiment(
            args.model_name,
            exp["ranks"],
            calib_state_pt,
            test_loader,
            device,
            out_dir / exp["name"],
        )

        # Save tokenizer alongside compressed model
        tokenizer.save_pretrained(str(out_dir / exp["name"]))

        print(f"  PPL         = {ppl:.2f}")
        print(f"  Actual MACs = {actual_macs/1e9:.3f} G  "
              f"({100*actual_macs/baseline_macs:.1f}% of baseline)")
        print(f"  Actual pars = {actual_params/1e6:.2f} M  "
              f"({100*actual_params/baseline_params:.1f}% of baseline)")

        results.append({
            "name":                exp["name"],
            "n_decompressed":      exp["n_decompressed"],
            "scale":               exp["scale"],
            "ppl":                 ppl,
            "ops_ratio":           actual_macs  / baseline_macs,
            "params_ratio":        actual_params / baseline_params,
            "theory_ops_ratio":    exp["theory_ops_ratio"],
            "theory_params_ratio": exp["theory_params_ratio"],
        })

    # ── Persist results ───────────────────────────────────────────────────────
    results_path = out_dir / "results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults → {results_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"{'Summary':^72}")
    print(f"{'='*72}")
    print(f"{'Experiment':<26}  {'PPL':>7}  {'Ops ratio':>10}  {'Params ratio':>12}")
    print(f"{'-'*72}")
    for r in results:
        print(f"  {r['name']:<24}  {r['ppl']:>7.2f}  "
              f"{r['ops_ratio']:>9.3f}   {r['params_ratio']:>11.3f}")
    print(f"{'='*72}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(results, out_dir)


if __name__ == "__main__":
    main()
