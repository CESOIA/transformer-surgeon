#!/usr/bin/env python3
"""
Non-homogeneous quantization benchmark on Qwen2-1.5B.

Shows that varying the quantization distribution across transformer layers
at a fixed FLOPs budget yields different perplexity tradeoffs.

Experiment setup
----------------
- Baseline    : float16, no compression
- Homogeneous : ALL linear projections (incl. k_proj, v_proj) at INT8
                via GPTQ (weights) + per-tensor asymmetric INT8 (activations)
- Non-hom k   : first k "units" in order [MHA0, MLP0, MHA1, MLP1, ...]
                are kept at fp16; remaining units are pushed to INT4 to
                compensate, keeping total effective FLOPs ≈ all-INT8 budget

Budget model
------------
  fp16 layer  → 1.0 FLOP per multiply-accumulate
  INT8 layer  → 0.5 FLOP per multiply-accumulate
  INT4 layer  → 0.25 FLOP per multiply-accumulate

  Homogeneous budget B = 0.5 × sum(projection MACs).
  For n fp16 units (extra cost M_fp16 − 0.5 × M_fp16 = +0.5 × M_fp16):
    remaining INT8 macs x = T − 3 × M_fp16  (exact iso-budget formula)
    remaining units are int8 (first x macs) or int4 (the rest)
  Feasible when M_fp16 ≤ T/3.

Metrics stored per experiment
------------------------------
  ppl           WikiText-2 test-set perplexity
  ops_ratio     effective FLOPs / float16 baseline FLOPs
  params_ratio  effective param-bytes / float16 param-bytes (INT8=0.5×, INT4=0.25×)

Output
------
  results/calib_state.pt         - calibration statistics (reused across runs)
  results/results.json           - all metrics
  results/ppl_vs_budget.png      - summary plot

Usage
-----
  python run_benchmark.py
  python run_benchmark.py --decompress-steps 0 2 4 8 12 16
  python run_benchmark.py --num-calibration-samples 64 --num-test-samples 100
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformersurgeon import Qwen2CompressionSchemesManager, Qwen2ForCausalLMCompress


# ── Defaults ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2-1.5B"
RANDOM_SEED  = 42
SEQ_LEN      = 2048
NUM_CALIBRATION_SAMPLES = 32
NUM_TEST_SAMPLES        = 50

# All projection layer suffixes (full dotted suffixes relative to model.layers.i)
ALL_PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)
# Substring criteria that target projection layers in the manager (OR logic)
PROJ_CRITERIA = ["self_attn", "mlp"]

# FLOPs / memory budget scale factors
FLOPS_FP16 = 1.0
FLOPS_INT8 = 0.5
FLOPS_INT4 = 0.25
PREC_FLOPS = {"fp16": FLOPS_FP16, 8: FLOPS_INT8, 4: FLOPS_INT4}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Non-homogeneous INT8/INT4 quantization benchmark")
    p.add_argument("--model-name", default=MODEL_NAME)
    p.add_argument("--num-calibration-samples", type=int, default=NUM_CALIBRATION_SAMPLES)
    p.add_argument("--num-test-samples",        type=int, default=NUM_TEST_SAMPLES)
    p.add_argument("--seq-len",                 type=int, default=SEQ_LEN)
    p.add_argument("--decompress-steps", type=int, nargs="+", default=None,
                   help="Numbers of fp16 units from front; default: 0 2 4 8 12 16")
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

    calib_examples = []
    for _ in range(min(num_calib, len(calib_ids) // seq_len)):
        start = rng.randint(0, len(calib_ids) - seq_len - 1)
        calib_examples.append({
            "input_ids":      torch.tensor(calib_ids[start:start + seq_len], dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
        })

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


# ── Layer shape analysis ───────────────────────────────────────────────────────
def compute_layer_info(model):
    """
    Walk the model and collect shape info for every tracked projection.

    Returns dict keyed by full dotted path:
      {"out": int, "in": int, "ops_full": int}
    """
    info = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(name.endswith(s) for s in ALL_PROJECTIONS):
            continue
        if "lm_head" in name or "embed" in name:
            continue
        info[name] = {
            "out":      mod.out_features,
            "in":       mod.in_features,
            "ops_full": mod.out_features * mod.in_features,
        }
    return info


def compute_baseline_flops_params(model):
    """Total MACs and params (fp16 baseline — all layers contribute factor 1.0)."""
    total_mac = 0
    total_par = 0
    visited   = set()
    for _, mod in model.named_modules():
        if id(mod) in visited:
            continue
        visited.add(id(mod))
        if not isinstance(mod, nn.Linear):
            continue
        out_f, in_f = mod.weight.shape
        total_mac += out_f * in_f
        total_par += out_f * in_f
        if mod.bias is not None:
            total_par += out_f
    return total_mac, total_par


def compute_effective_flops_params(model):
    """
    Effective FLOPs and params after applying quantization.

    INT4/INT8 layers (detected via _soft_quant_precision) contribute 0.25/0.5
    for both FLOPs (speed) and params (memory footprint).
    """
    prec_to_factor = {4: FLOPS_INT4, 8: FLOPS_INT8}
    total_flops = 0.0
    total_par   = 0
    visited     = set()
    for _, mod in model.named_modules():
        if id(mod) in visited:
            continue
        visited.add(id(mod))
        if not isinstance(mod, nn.Linear):
            continue
        out_f, in_f = mod.weight.shape
        factor = prec_to_factor.get(getattr(mod, "_soft_quant_precision", None), FLOPS_FP16)
        total_flops += out_f * in_f * factor
        total_par   += out_f * in_f
        if mod.bias is not None:
            total_par += out_f
    return total_flops, total_par


# ── Decompression unit helpers ─────────────────────────────────────────────────
def build_decompression_units(num_layers: int):
    """
    Ordered list of (de)compression units:
      [MHA_0, MLP_0, MHA_1, MLP_1, ..., MHA_{N-1}, MLP_{N-1}]

    Each unit: {"type": "mha"|"mlp", "layer": int, "criteria": list}
    where criteria is the AND list passed to manager.iter_filtered.
    """
    units = []
    for i in range(num_layers):
        units.append({"type": "mha", "layer": i, "criteria": [i, "self_attn"]})
        units.append({"type": "mlp", "layer": i, "criteria": [i, "mlp"]})
    return units


def unit_full_paths(unit, layer_info: dict) -> list:
    """Return the layer_info keys that belong to *unit*."""
    i      = unit["layer"]
    prefix = "self_attn" if unit["type"] == "mha" else "mlp"
    return [p for p in layer_info if f"layers.{i}.{prefix}" in p]


# ── Iso-budget precision assignment ───────────────────────────────────────────
def compute_experiment_precisions(layer_info: dict, units: list, n_decompressed: int):
    """
    Assign per-unit precision to maintain the all-INT8 budget.

    Budget B = 0.5 × T  (all projection layers at INT8)

    For n units at fp16 (total macs M_fp16):
      INT8 macs for remaining = x = T − 3 × M_fp16   (exact iso-budget)
      Units from position n onward are greedily set to INT8 until x is consumed,
      the rest go to INT4.

    Returns list of precisions (one per unit: "fp16" | 8 | 4),
    or None if M_fp16 > B (infeasible: fp16 cost alone exceeds budget).
    """
    T = sum(v["ops_full"] for v in layer_info.values())
    B = 0.5 * T

    unit_macs = [
        sum(layer_info[p]["ops_full"] for p in unit_full_paths(u, layer_info) if p in layer_info)
        for u in units
    ]

    M_fp16 = sum(unit_macs[i] for i in range(n_decompressed))
    # Infeasible when fp16 cost exceeds T/3: even all-INT4 for remaining
    # can't bring total down to B = 0.5×T.
    if 3 * M_fp16 > T:
        return None

    M_rest = sum(unit_macs[i] for i in range(n_decompressed, len(units)))

    # Solve: 0.5x + 0.25(M_rest − x) = B − M_fp16  →  x = T − 3×M_fp16
    x = float(T - 3 * M_fp16)
    x = max(0.0, min(x, float(M_rest)))

    precisions = ["fp16"] * n_decompressed
    accumulated_int8 = 0.0
    for i in range(n_decompressed, len(units)):
        macs = unit_macs[i]
        if accumulated_int8 + macs <= x + 1e-6:
            precisions.append(8)
            accumulated_int8 += macs
        else:
            precisions.append(4)

    return precisions


def compute_theory_flops(layer_info: dict, units: list, unit_precisions: list,
                          macs_not_tracked: int):
    """Theoretical effective FLOPs given per-unit precision assignments."""
    path_prec = {}
    for u, prec in zip(units, unit_precisions):
        for p in unit_full_paths(u, layer_info):
            path_prec[p] = prec
    flops = macs_not_tracked * FLOPS_FP16
    for path, v in layer_info.items():
        flops += v["ops_full"] * PREC_FLOPS[path_prec.get(path, "fp16")]
    return flops


# ── Quantization configuration helpers ────────────────────────────────────────
def _configure_precision(manager, precision, criteria):
    """
    Configure *criteria*-selected layers to the given precision.

    precision: "fp16" | 8 | 4
    """
    if precision == "fp16":
        manager.set("quantization", "precision",            "full",   criteria=criteria)
        # manager.set("quantization", "precision_activation", "full",   criteria=criteria)
    else:
        manager.set("quantization", "precision",            precision, criteria=criteria)
        manager.set("quantization", "method",               "gptq",   criteria=criteria)
        manager.set("quantization", "granularity",          "per_channel", criteria=criteria)
        manager.set("quantization", "sparsity",             0.0,      criteria=criteria)
        manager.set("quantization", "sparse_method",        "magnitude", criteria=criteria)
        manager.set("quantization", "eps",                  1e-6,     criteria=criteria)
        # manager.set("quantization", "precision_activation", precision, criteria=criteria)
        manager.set("quantization", "method_activation",    "maxmin", criteria=criteria)
        manager.set("quantization", "scheme_activation",    "asymmetric", criteria=criteria)


# ── Calibration ────────────────────────────────────────────────────────────────
def run_calibration_and_save(model_name, calib_loader, state_path: Path, device):
    """
    Load a fresh model, configure all projections at INT8, run one calibration
    pass to collect GPTQ covariance + activation-range statistics, and save.

    INT8 calibration data is also valid for INT4 (same XᵀX covariance; activation
    min/max ranges are precision-independent).
    """
    print("  Loading model for calibration...")
    model = Qwen2ForCausalLMCompress.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()

    manager = Qwen2CompressionSchemesManager(model)
    _configure_precision(manager, 8, PROJ_CRITERIA)
    manager.set_calibration_data(calib_loader)

    print("  Collecting covariance + activation-range statistics "
          "(one forward pass per batch)...")
    manager.run_calibration(
        criteria=PROJ_CRITERIA, device=device, offload_to_cpu=True, show_progress=True
    )

    state_path.parent.mkdir(parents=True, exist_ok=True)
    manager.save_state(str(state_path))
    print(f"  Calibration state saved to {state_path}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ── Single experiment ──────────────────────────────────────────────────────────
def run_experiment(model_name, units, unit_precisions, state_path: Path,
                   test_loader, device):
    """
    Reload a fresh model, apply the per-unit precision assignment,
    evaluate PPL, and return metrics.

    Returns (ppl, effective_flops, actual_params).
    """
    print("  Loading fresh model...")
    model = Qwen2ForCausalLMCompress.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device).eval()

    manager = Qwen2CompressionSchemesManager(model)

    # Apply per-unit precision (fp16 / int8 / int4)
    for u, prec in zip(units, unit_precisions):
        _configure_precision(manager, prec, [u["criteria"]])

    # Load pre-computed calibration stats — valid for both int8 and int4
    manager.load_state(str(state_path))

    # Only apply on quantized layers (fp16 units are no-ops)
    print("  Applying quantization...")
    manager.apply(hard=False, criteria=PROJ_CRITERIA, device=device, offload_to_cpu=True)

    print("  Evaluating perplexity on WikiText-2 test set...")
    ppl = evaluate_perplexity(model, test_loader, device)

    effective_flops, actual_params = compute_effective_flops_params(model)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ppl, effective_flops, actual_params


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(results: list, out_dir: Path):
    baseline   = next((r for r in results if r["name"] == "baseline_float16"), None)
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
        "Non-homogeneous INT8/INT4 quantization — Qwen2-1.5B | GPTQ | WikiText-2\n"
        "iso-budget: fp16 front units compensated by INT4 back units",
        fontsize=12, fontweight="bold"
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
                   s=90, zorder=5, label=f"All-INT8 homogeneous ({hom['ppl']:.1f})")
    ax.set_xlabel("fp16 units from front\n(remaining: INT8 → INT4 to maintain budget)")
    ax.set_ylabel("Perplexity ↓")
    ax.set_title("PPL vs. decompression front\n(same total FLOPs budget)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: PPL vs ops ratio ─────────────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(ops, ppls, c=ns, cmap="coolwarm", s=90, zorder=4)
    plt.colorbar(sc, ax=ax, label="# fp16 units from front")
    for op, ppl, n in zip(ops, ppls, ns):
        ax.annotate(f"n={n}", (op, ppl), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
    if baseline:
        ax.scatter([baseline["ops_ratio"]], [baseline["ppl"]],
                   marker="*", color="gold", s=160, zorder=5, label="float16")
        ax.legend(fontsize=8)
    ax.set_xlabel("Ops ratio  (effective FLOPs / float16 FLOPs)")
    ax.set_ylabel("Perplexity ↓")
    ax.set_title("PPL vs. compute budget\n(INT4=0.25×, INT8=0.5×, fp16=1.0×)")
    ax.grid(True, alpha=0.3)

    # ── Plot 3: PPL vs params ratio ──────────────────────────────────────────
    ax = axes[2]
    sc = ax.scatter(pars, ppls, c=ns, cmap="coolwarm", s=90, zorder=4)
    plt.colorbar(sc, ax=ax, label="# fp16 units from front")
    for pr, ppl, n in zip(pars, ppls, ns):
        ax.annotate(f"n={n}", (pr, ppl), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
    if baseline:
        ax.scatter([baseline["params_ratio"]], [baseline["ppl"]],
                   marker="*", color="gold", s=160, zorder=5, label="float16")
        ax.legend(fontsize=8)
    ax.set_xlabel("Params ratio  (effective param-bytes / float16 param-bytes)")
    ax.set_ylabel("Perplexity ↓")
    ax.set_title("PPL vs. memory footprint\n(INT4=0.25×, INT8=0.5×, fp16=1.0×)")
    ax.grid(True, alpha=0.3)

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
    print(f"Budget      : fp16=1.0, INT8=0.5, INT4=0.25 FLOP per MAC")
    print(f"             Fixed at B = 0.5 × projection MACs (all-INT8)")

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
    print(f"\nAnalysing model layer shapes...")
    probe = Qwen2ForCausalLMCompress.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    )
    layer_info              = compute_layer_info(probe)
    baseline_macs, baseline_params = compute_baseline_flops_params(probe)
    num_layers              = probe.config.num_hidden_layers
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    T = sum(v["ops_full"] for v in layer_info.values())  # projection macs only
    macs_not_tracked = baseline_macs - T
    budget_B         = 0.5 * T  # all-INT8 projection budget

    print(f"  Hidden layers      : {num_layers}")
    print(f"  Tracked projections: {len(layer_info)}")
    print(f"  Baseline MACs      : {baseline_macs/1e9:.3f} G")
    print(f"  Projection MACs T  : {T/1e9:.3f} G")
    print(f"  Budget B (all-INT8): {budget_B/1e9:.3f} G  "
          f"(ops_ratio = {(budget_B + macs_not_tracked)/baseline_macs:.3f})")
    print(f"  Max feasible units : ≤ T/3 ≈ "
          f"{T / 3 / (T / (2*num_layers)):.1f} units  "
          f"(beyond this, fp16 cost > B)")

    # ── Decompression units & experiment schedule ─────────────────────────────
    units = build_decompression_units(num_layers)

    steps = sorted(set(args.decompress_steps)) if args.decompress_steps \
        else [0, 2, 4, 8, 12, 16]

    print(f"\n{'='*76}")
    print(f"{'Experiment design':^76}")
    print(f"{'='*76}")
    print(f"{'n fp16':>7}  {'n int8':>7}  {'n int4':>7}  "
          f"{'ops ratio':>10}  {'last fp16 unit':>16}")
    print(f"{'-'*76}")

    experiments = []
    for n in steps:
        if n > len(units):
            print(f"  n={n:3d}  skipped (exceeds total units={len(units)})")
            continue

        unit_precisions = compute_experiment_precisions(layer_info, units, n)
        if unit_precisions is None:
            print(f"  n={n:3d}  INFEASIBLE (fp16 MACs > T/3; INT4 can't compensate)")
            continue

        n_fp16 = unit_precisions.count("fp16")
        n_int8 = unit_precisions.count(8)
        n_int4 = unit_precisions.count(4)

        theory_flops     = compute_theory_flops(layer_info, units, unit_precisions, macs_not_tracked)
        theory_ops_ratio = theory_flops / baseline_macs

        last_unit = units[n - 1] if n > 0 else None
        last_label = (
            f"MHA{last_unit['layer']}" if last_unit and last_unit["type"] == "mha"
            else (f"MLP{last_unit['layer']}" if last_unit else "—")
        )

        print(f"  n={n:3d}    {n_fp16:6d}   {n_int8:6d}   {n_int4:6d}  "
              f"{theory_ops_ratio:9.3f}    {last_label:>16}")

        experiments.append({
            "n_decompressed":    n,
            "name":              f"nonhom_{n:02d}_units",
            "unit_precisions":   unit_precisions,
            "n_fp16":            n_fp16,
            "n_int8":            n_int8,
            "n_int4":            n_int4,
            "theory_flops":      theory_flops,
            "theory_ops_ratio":  theory_ops_ratio,
        })
    print(f"{'='*76}")

    # ── Calibration (run once, reused for int8 and int4) ─────────────────────
    if calib_state_pt.exists() and args.skip_calibration:
        print(f"\nReusing existing calibration state: {calib_state_pt}")
    else:
        print(f"\n{'─'*60}")
        print("Running calibration pass (GPTQ covariance + activation-range)...")
        run_calibration_and_save(
            args.model_name, calib_loader, calib_state_pt, device
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
        base_macs, base_par = compute_baseline_flops_params(base_model)
        print(f"  PPL = {base_ppl:.2f}")
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        results.append({
            "name":             "baseline_float16",
            "n_decompressed":    None,
            "ppl":               base_ppl,
            "ops_ratio":         base_macs  / baseline_macs,
            "params_ratio":      base_par   / baseline_params,
            "theory_ops_ratio":  1.0,
        })

    # ── Compressed experiments ────────────────────────────────────────────────
    for exp in experiments:
        print(f"\n{'─'*60}")
        print(f"Experiment: {exp['name']}  "
              f"(fp16={exp['n_fp16']}, int8={exp['n_int8']}, int4={exp['n_int4']}, "
              f"theory ops={exp['theory_ops_ratio']:.3f})")

        ppl, eff_flops, actual_params = run_experiment(
            args.model_name,
            units,
            exp["unit_precisions"],
            calib_state_pt,
            test_loader,
            device,
        )

        actual_ops_ratio    = eff_flops     / baseline_macs
        actual_params_ratio = actual_params / baseline_params

        print(f"  PPL              = {ppl:.2f}")
        print(f"  Effective FLOPs  = {eff_flops/1e9:.3f} G  "
              f"({100*actual_ops_ratio:.1f}% of fp16 baseline)")
        print(f"  Actual params    = {actual_params/1e6:.2f} M  "
              f"({100*actual_params_ratio:.1f}% of baseline)")

        results.append({
            "name":             exp["name"],
            "n_decompressed":   exp["n_decompressed"],
            "n_fp16":           exp["n_fp16"],
            "n_int8":           exp["n_int8"],
            "n_int4":           exp["n_int4"],
            "ppl":              ppl,
            "ops_ratio":        actual_ops_ratio,
            "params_ratio":     actual_params_ratio,
            "theory_ops_ratio": exp["theory_ops_ratio"],
        })

    # ── Persist results ───────────────────────────────────────────────────────
    results_path = out_dir / "results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults → {results_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Summary':^80}")
    print(f"{'='*80}")
    print(f"{'Experiment':<24}  {'PPL':>7}  {'fp16':>5}  {'int8':>5}  {'int4':>5}  "
          f"{'Ops ratio':>10}  {'Params ratio':>12}")
    print(f"{'-'*80}")
    for r in results:
        fp16_s = str(r.get("n_fp16", "—")).rjust(5)
        int8_s = str(r.get("n_int8", "—")).rjust(5)
        int4_s = str(r.get("n_int4", "—")).rjust(5)
        print(f"  {r['name']:<22}  {r['ppl']:>7.2f}  {fp16_s}  {int8_s}  {int4_s}  "
              f"{r['ops_ratio']:>9.3f}   {r['params_ratio']:>11.3f}")
    print(f"{'='*80}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(results, out_dir)


if __name__ == "__main__":
    main()
