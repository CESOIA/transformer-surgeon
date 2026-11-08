"""Fit the hardware cost predictor from the measured experiments (E0-E4) and validate it on E5.

    python scripts/cost_predictor/fit_predictor.py --out-dir experiments/cp_xnnpack
    python scripts/cost_predictor/fit_predictor.py --out-dir experiments/cp_xnnpack --residual-model ridge

Input: the folder generate_models.py wrote for one backend, with the measurements of each experiment in
``<experiment dir>/results.csv``. Copy ``results_template.csv`` (same columns) and fill it, one row per run;
repeated runs of a configuration are combined by the median, per decode step:

    prefill_ms              prefill time of the P-token prompt
    decode_trace_ms         per-step decode times t(j), j = 0 .. G_max-1: inline ("12.1;12.0;...") or the path,
                            relative to results.csv, of a .json list or a text file with one value per line
    prefill_energy_j        optional: prefill energy
    decode_energy_trace_j   optional: per-step decode energy (same formats as decode_trace_ms)
    thermal_throttled       rows marked 1/true/yes are dropped (--keep-throttled keeps them)

Steps (the estimation of delta and DeltaT_KV, then the overall experimental plan):

1. E0: reference surface T_ref^pre(P) and T_ref^dec(P, G) = sum of the first G per-step times.
2. E1: delta_{tau,b}(P[, G]) = (T(c_F) - T(c_ref)) / N_tau for each projection type tau and precision b.
3. E2: DeltaT_KV(b_KV, P[, G]) = T(reference weights, b_KV) - T(c_ref).
4. E3: S_KV = DeltaT_KV(b_KV; W4) - DeltaT_KV(b_KV; W8), the weight-KV separability check (reported only).
5. E4: residuals r_pre = T_meas^pre - T_struct^pre and r_dec = (T_meas^dec(G_max) - T_struct^dec(G_max)) / G_max,
   learned per prompt length P from the configuration features f(c); delta and DeltaT_KV are frozen first.
6. E5: errors of T_ref ("reference"), T_ref + DeltaT_comp ("structured") and the full predictor ("full"), per P
   and G, for the prefill, the first G decode steps and the whole request (prefill + G decode steps).

Energy goes through the same steps when its columns are filled. Everything is written to <out-dir>/predictor/;
``predictor_<metric>.json`` + ``residual_<metric>.pkl`` load with ``CostPredictor.load`` for a compression search.
Predictions exist only at the profiled P and for G up to the measured window: there is no interpolation.
"""

import argparse
import csv
import json
import pickle
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import experiment_plan as plan_mod
from experiment_plan import PRECISION_NAMES, PROJECTIONS, REF_WEIGHT_BITS

METRICS = {"time": ("prefill_ms", "decode_trace_ms"), "energy": ("prefill_energy_j", "decode_energy_trace_j")}
LEVELS = ("reference", "structured", "full")
REF_W = PRECISION_NAMES[REF_WEIGHT_BITS]
MIN_SAMPLES = 5   # fewer E4 samples at a P: the residual falls back to their mean
_TRUE = {"1", "true", "yes", "y"}


# ----------------------------------------------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------------------------------------------

def _numbers(text):
    return [float(v) for v in re.split(r"[;,\s]+", text.strip()) if v]


def parse_trace(cell, base: Path):
    """Per-step values from an inline list or from a file next to the results (``None`` when empty)."""
    cell = (cell or "").strip()
    if not cell:
        return None
    try:
        return np.array(_numbers(cell))
    except ValueError:
        path = base / cell
        text = path.read_text()
        return np.asarray(json.loads(text) if path.suffix == ".json" else _numbers(text), dtype=float)


@dataclass
class Measurement:
    """One configuration at one prompt length, combined over runs."""
    pre: float | None           # median prefill
    steps: np.ndarray | None    # median per-step decode

    @property
    def cum(self):
        """``cum[G]`` = sum of the first G decode steps (``cum[0] = 0``)."""
        return None if self.steps is None else np.concatenate([[0.0], np.cumsum(self.steps)])


def load_results(path: Path, metric, backend, keep_throttled):
    """``{(config_id, P): Measurement}`` from one results.csv ({} when the file is missing)."""
    if not path.is_file():
        return {}
    pre_col, dec_col = METRICS[metric]
    runs = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if backend and (row.get("backend") or backend) != backend:
                continue
            if not keep_throttled and (row.get("thermal_throttled") or "").strip().lower() in _TRUE:
                continue
            pre = float(row[pre_col]) if (row.get(pre_col) or "").strip() else None
            steps = parse_trace(row.get(dec_col), path.parent)
            if pre is not None or steps is not None:
                runs.setdefault((row["config_id"], int(row["P"])), []).append((pre, steps))
    out = {}
    for key, measured in runs.items():
        pres = [pre for pre, _ in measured if pre is not None]
        traces = [steps for _, steps in measured if steps is not None and len(steps)]
        steps = None
        if traces:
            n = min(len(t) for t in traces)
            steps = np.median(np.stack([t[:n] for t in traces]), axis=0)
        out[key] = Measurement(statistics.median(pres) if pres else None, steps)
    return out


@dataclass
class ExperimentData:
    specs: dict    # config_id -> spec.json
    meas: dict     # (config_id, P) -> Measurement


def load_experiments(out: Path, plan, metric, backend, keep_throttled, results_name):
    data = {}
    for name, info in plan["experiments"].items():
        exp_dir = out / info["dir"]
        configs = json.loads((exp_dir / "experiment.json").read_text())["configs"]
        specs = {cid: json.loads((exp_dir / "models" / cid / "spec.json").read_text()) for cid in configs}
        data[name] = ExperimentData(specs, load_results(exp_dir / results_name, metric, backend, keep_throttled))
    return data


# ----------------------------------------------------------------------------------------------------------------
# Structured deviation model
# ----------------------------------------------------------------------------------------------------------------

def _deviations(counts):
    """``("tau|b", N_{tau,b})`` for every projection type and precision other than the reference."""
    for tau in PROJECTIONS:
        for b, n in counts[tau].items():
            if b != REF_W and n:
                yield f"{tau}|{b}", n


def _need(table, key, P, what):
    try:
        return table[key][P]
    except KeyError:
        raise KeyError(f"no {what} measurement for {key} at P={P}") from None


@dataclass
class Structured:
    """T_struct = T_ref + sum_{tau, b != ref} N_{tau,b}(c) delta_{tau,b} + DeltaT_KV(b_KV), per phase and P.
    Decode terms are cumulative over the first G steps (index G)."""
    kv_ref: str
    ref_pre: dict      # P -> T_ref^pre
    ref_cum: dict      # P -> [T_ref^dec(G)]
    delta_pre: dict    # "tau|b" -> {P: delta^pre}
    delta_cum: dict    # "tau|b" -> {P: [delta^dec(G)]}
    kv_pre: dict       # b_KV -> {P: DeltaT_KV^pre}
    kv_cum: dict       # b_KV -> {P: [DeltaT_KV^dec(G)]}

    def g_max(self, P):
        """Largest G every decode term covers at P."""
        tables = [self.delta_cum, self.kv_cum]
        return min([len(self.ref_cum[P])] + [len(t[P]) for table in tables for t in table.values() if P in t]) - 1

    def prefill(self, counts, kv, P):
        value = self.ref_pre[P]
        for key, n in _deviations(counts):
            value += n * _need(self.delta_pre, key, P, "E1 prefill")
        if kv != self.kv_ref:
            value += _need(self.kv_pre, kv, P, "E2 prefill")
        return value

    def decode(self, counts, kv, P, G):
        value = self.ref_cum[P][G]
        for key, n in _deviations(counts):
            value += n * _need(self.delta_cum, key, P, "E1 decode")[G]
        if kv != self.kv_ref:
            value += _need(self.kv_cum, kv, P, "E2 decode")[G]
        return value


def _difference(m, base, scale=1.0):
    """(prefill difference, cumulative decode difference) of two measurements, divided by ``scale``."""
    pre = (m.pre - base.pre) / scale if m.pre is not None and base.pre is not None else None
    cum = None
    if m.cum is not None and base.cum is not None:
        n = min(len(m.cum), len(base.cum))
        cum = ((m.cum[:n] - base.cum[:n]) / scale).tolist()
    return pre, cum


def fit_structured(data, kv_ref, warn):
    """Reference surface (E0), per-layer deviations (E1) and KV deviations (E2)."""
    ref = {P: m for (cid, P), m in data["E0"].meas.items() if cid == "E0_ref"}
    if not ref:
        raise SystemExit("E0 has no measurement: the reference surface is required.")
    s = Structured(kv_ref, {}, {}, {}, {}, {}, {})
    for P, m in ref.items():
        if m.pre is not None:
            s.ref_pre[P] = m.pre
        if m.cum is not None:
            s.ref_cum[P] = m.cum.tolist()
    sweeps = [("E1", lambda spec: (f"{spec['tags']['tau']}|{spec['tags']['b']}", spec["tags"]["N_tau"]),
               s.delta_pre, s.delta_cum),
              ("E2", lambda spec: (spec["kv"], 1.0), s.kv_pre, s.kv_cum)]
    for name, key_of, pre_table, cum_table in sweeps:
        if name not in data:
            continue
        for cid, spec in data[name].specs.items():
            key, scale = key_of(spec)
            for P, base in ref.items():
                m = data[name].meas.get((cid, P))
                if m is None:
                    warn(f"{name} {cid} has no measurement at P={P}")
                    continue
                pre, cum = _difference(m, base, scale)
                if pre is not None:
                    pre_table.setdefault(key, {})[P] = pre
                if cum is not None:
                    cum_table.setdefault(key, {})[P] = cum
    return s


def separability(data, s, g_list):
    """E3: S_KV = DeltaT_KV(b_KV; W4) - DeltaT_KV(b_KV; W8) per phase, P and G."""
    rows = []
    if "E3" not in data:
        return rows
    e3 = data["E3"]
    base_id = next(cid for cid, spec in e3.specs.items() if spec["kv"] == s.kv_ref)
    for cid, spec in e3.specs.items():
        kv = spec["kv"]
        if cid == base_id:
            continue
        for P in sorted({P for c, P in e3.meas if c == cid}):
            base = e3.meas.get((base_id, P))
            if base is None:
                continue
            pre_w4, cum_w4 = _difference(e3.meas[(cid, P)], base)
            if pre_w4 is not None and P in s.kv_pre.get(kv, {}):
                rows.append(_s_row(kv, P, "prefill", "", s.kv_pre[kv][P], pre_w4))
            cum_w8 = s.kv_cum.get(kv, {}).get(P)
            if cum_w4 is not None and cum_w8 is not None:
                for G in g_list:
                    if G < min(len(cum_w4), len(cum_w8)):
                        rows.append(_s_row(kv, P, "decode", G, cum_w8[G], cum_w4[G]))
    return rows


def _s_row(kv, P, phase, G, w8, w4):
    return {"b_KV": kv, "P": P, "phase": phase, "G": G, "dT_KV_W8": w8, "dT_KV_W4": w4, "S_KV": w4 - w8,
            "S_KV_relative": (w4 - w8) / abs(w8) if w8 else float("nan")}


# ----------------------------------------------------------------------------------------------------------------
# Predictor (structured part + learned residual)
# ----------------------------------------------------------------------------------------------------------------

def make_regressor(kind):
    if kind == "gbdt":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                                         random_state=0)
    if kind == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    raise ValueError(kind)


def _int_keys(obj):
    """JSON turns the integer P keys into strings: turn them back."""
    if isinstance(obj, dict):
        return {int(k) if isinstance(k, str) and k.isdigit() else k: _int_keys(v) for k, v in obj.items()}
    return obj


class CostPredictor:
    """T_hat(c, P, G) = T_ref(P, G) + DeltaT_comp(c, P, G) + R_pre(f(c)) + G * R_dec(f(c)) for one backend and
    metric.

    ``predict(weights, kv, P, G)`` takes the per-layer weight bits ``weights[block][projection]`` (16, 8 or 4, in
    the order of ``experiment_plan.PROJECTIONS``) and a KV label (the reference, ``"INT8"``, ``"INT4"`` or ``"FP8"``).
    """

    def __init__(self, meta, structured, residuals):
        self.meta = meta              # metric, backend, dims, backends, feature_names, residual_model
        self.structured = structured
        self.residuals = residuals    # P -> {"pre": regressor or constant, "dec": regressor or constant}
        self.dims = plan_mod.ModelDims(**meta["dims"])

    def features(self, weights, kv):
        """(precision counts per projection type, f(c) vector in ``feature_names`` order)."""
        spec = plan_mod.ConfigSpec("candidate", "search", "", weights, kv)
        f = plan_mod.features(spec, self.dims, self.meta["backends"])
        columns = plan_mod.feature_columns(f, self.meta["backends"])
        return f["precision_counts_per_type"], np.array([float(columns[k]) for k in self.meta["feature_names"]])

    def residual(self, P, phase, x):
        model = self.residuals.get(P, {}).get(phase, 0.0)
        return float(model.predict(x[None])[0]) if hasattr(model, "predict") else float(model)

    def predict(self, weights, kv, P, G, level="full"):
        """``{"prefill", "decode", "total"}`` at prompt length P after G generated tokens (``None`` for a phase
        that was not profiled). ``level``: ``"reference"``, ``"structured"`` or ``"full"``."""
        s = self.structured
        if P not in s.ref_pre and P not in s.ref_cum:
            raise ValueError(f"P={P} was not profiled (profiled: {sorted(set(s.ref_pre) | set(s.ref_cum))}).")
        counts, x = self.features(weights, kv)
        pre = dec = None
        if P in s.ref_pre:
            pre = s.ref_pre[P] if level == "reference" else s.prefill(counts, kv, P)
            if level == "full":
                pre += self.residual(P, "pre", x)
        if P in s.ref_cum:
            if G > s.g_max(P):
                raise ValueError(f"G={G} is beyond the measured decode window at P={P} ({s.g_max(P)}).")
            dec = s.ref_cum[P][G] if level == "reference" else s.decode(counts, kv, P, G)
            if level == "full":
                dec += G * self.residual(P, "dec", x)
        total = pre + dec if pre is not None and dec is not None else None
        return {"prefill": pre, "decode": dec, "total": total}

    def save(self, directory: Path):
        metric = self.meta["metric"]
        payload = {**self.meta, "structured": asdict(self.structured)}
        (directory / f"predictor_{metric}.json").write_text(json.dumps(payload, indent=1))
        with open(directory / f"residual_{metric}.pkl", "wb") as fh:
            pickle.dump(self.residuals, fh)

    @classmethod
    def load(cls, directory, metric="time"):
        directory = Path(directory)
        payload = json.loads((directory / f"predictor_{metric}.json").read_text())
        structured = Structured(**_int_keys(payload.pop("structured")))
        with open(directory / f"residual_{metric}.pkl", "rb") as fh:
            residuals = pickle.load(fh)
        return cls(payload, structured, residuals)


def fit_residuals(predictor, data, kind, warn):
    """One regressor per (P, phase) on the E4 residuals; returns (models, residual rows)."""
    s = predictor.structured
    samples, rows = {}, []
    if "E4" not in data or not data["E4"].meas:
        warn("E4 has no measurement: the predictor has no residual term")
        return {}, rows
    for (cid, P), m in sorted(data["E4"].meas.items()):
        spec = data["E4"].specs[cid]
        counts, x = predictor.features(spec["weights"], spec["kv"])
        try:
            r_pre = m.pre - s.prefill(counts, spec["kv"], P) if m.pre is not None and P in s.ref_pre else None
            r_dec = None
            G = min(len(m.cum) - 1, s.g_max(P)) if m.cum is not None and P in s.ref_cum else 0
            if G > 0:
                r_dec = (m.cum[G] - s.decode(counts, spec["kv"], P, G)) / G
        except KeyError as exc:
            warn(f"E4 {cid}: {exc}")
            continue
        samples.setdefault(P, []).append((x, r_pre, r_dec))
        rows.append({"config_id": cid, "P": P, "b_KV": spec["kv"], "r_pre": r_pre, "r_dec_per_token": r_dec})
    models = {}
    for P, points in samples.items():
        models[P] = {}
        for phase, index in (("pre", 1), ("dec", 2)):
            usable = [(p[0], p[index]) for p in points if p[index] is not None]
            if kind == "none" or not usable:
                models[P][phase] = 0.0
                continue
            X, y = np.stack([u[0] for u in usable]), np.array([u[1] for u in usable])
            if kind == "mean" or len(y) < MIN_SAMPLES:
                if kind != "mean":
                    warn(f"E4 at P={P} ({phase}): {len(y)} samples, residual = their mean")
                models[P][phase] = float(y.mean())
            else:
                models[P][phase] = make_regressor(kind).fit(X, y)
    return models, rows


# ----------------------------------------------------------------------------------------------------------------
# Validation and output
# ----------------------------------------------------------------------------------------------------------------

def evaluate(predictor, data, name, label, g_list, warn):
    """Absolute percentage error of every level, phase, P and G for the measured configurations of ``name``."""
    rows = []
    if name not in data:
        return rows
    s = predictor.structured
    for (cid, P), m in sorted(data[name].meas.items()):
        spec = data[name].specs[cid]
        g_values = [G for G in g_list if m.cum is not None and P in s.ref_cum and G <= min(len(m.cum) - 1, s.g_max(P))]
        for level in LEVELS:
            try:
                points = [(G, predictor.predict(spec["weights"], spec["kv"], P, G, level)) for G in g_values or [0]]
            except KeyError as exc:
                warn(f"{name} {cid}: {exc}")
                break
            if m.pre is not None and points[0][1]["prefill"] is not None:
                rows.append(_error_row(label, cid, P, "prefill", "", level, m.pre, points[0][1]["prefill"]))
            for G, predicted in (points if g_values else []):
                rows.append(_error_row(label, cid, P, "decode", G, level, m.cum[G], predicted["decode"]))
                if m.pre is not None and predicted["total"] is not None:
                    rows.append(_error_row(label, cid, P, "total", G, level, m.pre + m.cum[G], predicted["total"]))
    return rows


def _error_row(label, cid, P, phase, G, level, measured, predicted):
    return {"set": label, "config_id": cid, "P": P, "phase": phase, "G": G, "level": level, "measured": measured,
            "predicted": predicted, "ape": abs(predicted - measured) / abs(measured) if measured else float("nan")}


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["set"], r["phase"], r["P"], r["G"], r["level"]), []).append(r["ape"])
    return [{"set": k[0], "phase": k[1], "P": k[2], "G": k[3], "level": k[4], "n": len(v),
             "mape": float(np.mean(v)), "median_ape": float(np.median(v)), "max_ape": float(np.max(v))}
            for k, v in groups.items()]


def write_csv(path: Path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def structured_tables(s, g_list):
    """Long-format tables of the reference surface, the weight deviations and the KV deviations."""
    reference = [{"P": P, "G": G, "T_ref_pre": s.ref_pre.get(P), "T_ref_dec": value}
                 for P, cum in sorted(s.ref_cum.items()) for G, value in enumerate(cum)]
    reference += [{"P": P, "G": "", "T_ref_pre": value, "T_ref_dec": None}
                  for P, value in sorted(s.ref_pre.items()) if P not in s.ref_cum]

    def deviations(pre_table, cum_table, names):
        rows = []
        for key in sorted(set(pre_table) | set(cum_table)):
            ident = dict(zip(names, key.split("|")))
            for P, value in sorted(pre_table.get(key, {}).items()):
                rows.append({**ident, "P": P, "phase": "prefill", "G": "", "value": value})
            for P, cum in sorted(cum_table.get(key, {}).items()):
                rows += [{**ident, "P": P, "phase": "decode", "G": G, "value": cum[G]} for G in g_list if G < len(cum)]
        return rows

    return (reference, deviations(s.delta_pre, s.delta_cum, ("tau", "b")),
            deviations(s.kv_pre, s.kv_cum, ("b_KV",)))


def print_summary(summary, label):
    rows = [r for r in summary if r["set"] == label]
    if not rows:
        print(f"  {label}: no measurement")
        return
    print(f"  {label}: mean absolute percentage error  (reference | structured | full)")
    table = {}
    for r in rows:
        table.setdefault((r["phase"], r["P"], r["G"]), {})[r["level"]] = r["mape"]
    order = {"prefill": 0, "decode": 1, "total": 2}
    for (phase, P, G), levels in sorted(table.items(), key=lambda item: (order[item[0][0]], item[0][1],
                                                                          -1 if item[0][2] == "" else item[0][2])):
        cells = " | ".join(f"{100 * levels[l]:6.2f}%" if l in levels else "     - " for l in LEVELS)
        print(f"    {phase:8s} P={P:<6} G={str(G):<5} {cells}")


def fit(args, plan, metric):
    out = Path(args.out_dir)
    warnings = []
    warn = warnings.append
    backend = args.backend or plan["backends"][0]
    data = load_experiments(out, plan, metric, backend, args.keep_throttled, args.results_name)
    if not any(e.meas for e in data.values()):
        return None
    kv_ref = data["E0"].specs["E0_ref"]["kv"]
    g_list = args.g_list or sorted(set(plan["eval_g"]) | {plan["g_max"]})

    structured = fit_structured(data, kv_ref, warn)
    names = list(plan_mod.feature_columns(data["E0"].specs["E0_ref"]["features"], plan["backends"]))
    meta = {"metric": metric, "backend": backend, "dims": plan["dims"], "backends": plan["backends"],
            "feature_names": names, "residual_model": args.residual_model, "model": plan["model"]}
    predictor = CostPredictor(meta, structured, {})
    predictor.residuals, residual_rows = fit_residuals(predictor, data, args.residual_model, warn)

    errors = (evaluate(predictor, data, "E5", "E5 held out", g_list, warn)
              + evaluate(predictor, data, "E4", "E4 training", g_list, warn))
    summary = summarize(errors)

    target = Path(args.predictor_dir or out / "predictor")
    target.mkdir(parents=True, exist_ok=True)
    reference, deltas, kv = structured_tables(structured, g_list)
    for name, rows in (("reference_surface", reference), ("delta_weights", deltas), ("delta_kv", kv),
                       ("separability", separability(data, structured, g_list)), ("residuals", residual_rows),
                       ("validation", errors), ("validation_summary", summary)):
        write_csv(target / f"{name}_{metric}.csv", rows)
    predictor.save(target)
    unique = list(dict.fromkeys(warnings))
    (target / f"warnings_{metric}.txt").write_text("\n".join(unique) + ("\n" if unique else ""))

    print(f"[{metric}] {backend}: E1 terms {len(structured.delta_cum) or len(structured.delta_pre)}, "
          f"KV terms {len(structured.kv_cum) or len(structured.kv_pre)}, E4 residual samples {len(residual_rows)}")
    print_summary(summary, "E5 held out")
    if unique:
        print(f"  {len(unique)} warning(s), see {target / f'warnings_{metric}.txt'}")
    return predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="the folder generate_models.py wrote for one backend")
    parser.add_argument("--backend", default=None, help="use only rows of this backend (default: the plan's)")
    parser.add_argument("--metric", choices=["time", "energy", "both"], default="both")
    parser.add_argument("--residual-model", choices=["gbdt", "ridge", "mean", "none"], default="gbdt",
                        help="learned correction per P: gradient-boosted trees, ridge regression, the mean "
                             "residual, or nothing (structured model only)")
    parser.add_argument("--results-name", default="results.csv", help="results file in every experiment folder")
    parser.add_argument("--g-list", nargs="+", type=int, default=None, help="G values to report "
                        "(default: the plan's eval_g and G_max)")
    parser.add_argument("--keep-throttled", action="store_true", help="keep runs marked thermal_throttled")
    parser.add_argument("--predictor-dir", default=None, help="output folder (default: <out-dir>/predictor)")
    args = parser.parse_args()

    plan = json.loads((Path(args.out_dir) / "plan.json").read_text())
    metrics = ["time", "energy"] if args.metric == "both" else [args.metric]
    fitted = [m for m in metrics if fit(args, plan, m) is not None]
    for metric in sorted(set(metrics) - set(fitted)):
        print(f"[{metric}] no measurement found in any {args.results_name}")
    if fitted:
        print(f"written to {Path(args.predictor_dir or Path(args.out_dir) / 'predictor')}")


if __name__ == "__main__":
    main()
