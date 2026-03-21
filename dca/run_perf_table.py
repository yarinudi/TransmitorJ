from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from run_dca import CalibrationAnalyzer, CalibrationConfig

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Use CSV or Parquet.")


def _parse_model_specs(specs: Sequence[str]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    for spec in specs:
        if ":" in spec:
            label, col = spec.split(":", 1)
            label, col = label.strip(), col.strip()
        else:
            col = spec.strip()
            label = col
        if not label or not col:
            raise ValueError(f"Invalid model spec: '{spec}'. Use 'Label:column' or 'column'.")
        models[label] = col
    if not models:
        raise ValueError("At least one --model must be provided.")
    return models


def _extract_risks(
    df: pd.DataFrame, models: Dict[str, str], clip: bool,
) -> Dict[str, np.ndarray]:
    risks: Dict[str, np.ndarray] = {}
    for label, col in models.items():
        r = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isnan(r).any():
            raise ValueError(f"Non-numeric values in risk column '{col}'.")
        if clip:
            r = np.clip(r, 0.0, 1.0)
        elif r.min() < 0.0 or r.max() > 1.0:
            raise ValueError(
                f"Risk column '{col}' outside [0,1]. Use --clip-risk."
            )
        risks[label] = r
    return risks


# ---------------------------------------------------------------------------
# KM censoring helpers (mirrored from run_dca to avoid tight coupling)
# ---------------------------------------------------------------------------

def _km_censoring_survival(
    times: np.ndarray, event: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """KM survival for the censoring distribution G(t) = P(C > t)."""
    censor_indicator = (event == 0).astype(int)
    order = np.argsort(times)
    t_sorted = times[order]
    d_sorted = censor_indicator[order]
    unique_t = np.unique(t_sorted)
    n = len(t_sorted)
    surv = 1.0
    surv_values = []
    for tt in unique_t:
        at_risk = n - np.searchsorted(t_sorted, tt, side="left")
        d_cens = np.sum((t_sorted == tt) & (d_sorted == 1))
        if at_risk > 0:
            surv *= 1.0 - d_cens / at_risk
        surv_values.append(surv)
    return unique_t, np.asarray(surv_values, dtype=float)


def _step_value(
    x: np.ndarray, y: np.ndarray, query: np.ndarray,
) -> np.ndarray:
    """Right-continuous step function; returns 1.0 before first x."""
    idx = np.searchsorted(x, query, side="right") - 1
    out = np.ones_like(query, dtype=float)
    valid = idx >= 0
    out[valid] = y[idx[valid]]
    return out


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_survival(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    event_value: str,
    horizon: float,
    models: Dict[str, str],
    clip_risk: bool,
) -> Tuple[
    np.ndarray,                 # time_all  (all subjects after NA drop)
    np.ndarray,                 # event_all
    Dict[str, np.ndarray],      # risks_all
    np.ndarray,                 # y_ipcw    (IPCW-usable subset)
    np.ndarray,                 # w_ipcw
    Dict[str, np.ndarray],      # risks_ipcw
]:
    """Return full arrays for C-td AND IPCW-transformed arrays for
    calibration / classification metrics."""
    needed = [time_col, event_col] + list(models.values())
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    work = df[needed].copy().dropna()
    if work.empty:
        raise ValueError("No rows remain after dropping NA values.")

    time = pd.to_numeric(work[time_col], errors="coerce").to_numpy(dtype=float)
    event = (work[event_col].astype(str) == str(event_value)).astype(int).to_numpy()
    if np.isnan(time).any():
        raise ValueError(f"Non-numeric values in time column '{time_col}'.")
    if np.any(time < 0):
        raise ValueError("Time values must be non-negative.")

    risks_all = _extract_risks(work, models, clip_risk)

    # IPCW at horizon
    km_t, km_g = _km_censoring_survival(time, event)
    g_ti = np.clip(_step_value(km_t, km_g, time), 1e-8, 1.0)
    g_h = float(np.clip(
        _step_value(km_t, km_g, np.array([horizon]))[0], 1e-8, 1.0,
    ))

    is_case = (event == 1) & (time <= horizon)
    is_control = time > horizon
    usable = is_case | is_control
    if usable.sum() == 0:
        raise ValueError("No usable rows after IPCW filtering.")

    y_ipcw = np.where(is_case, 1.0, 0.0)[usable]
    w_ipcw = np.where(
        is_case, 1.0 / g_ti, np.where(is_control, 1.0 / g_h, 0.0),
    )[usable]

    if np.sum(y_ipcw) == 0:
        raise ValueError("No events observed by horizon after IPCW setup.")

    risks_ipcw = {k: v[usable] for k, v in risks_all.items()}
    return time, event, risks_all, y_ipcw, w_ipcw, risks_ipcw


def prepare_binary(
    df: pd.DataFrame,
    outcome_col: str,
    event_value: str,
    models: Dict[str, str],
    clip_risk: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Return y_true, unit weights, risks (all subjects)."""
    needed = [outcome_col] + list(models.values())
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    work = df[needed].copy().dropna()
    if work.empty:
        raise ValueError("No rows remain after dropping NA values.")

    y = (work[outcome_col].astype(str) == str(event_value)).astype(int).to_numpy(dtype=float)
    if y.sum() == 0:
        raise ValueError("No positive events found. Check --event-value.")
    if y.sum() == len(y):
        raise ValueError("All rows are events; metrics are not meaningful.")

    risks = _extract_risks(work, models, clip_risk)
    return y, np.ones_like(y, dtype=float), risks


# ---------------------------------------------------------------------------
# Metric 1: Uno's C-td
# ---------------------------------------------------------------------------

def unos_c_td(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    horizon: float,
) -> float:
    """Uno's concordance statistic C_td(tau).

    For each case i with event at T_i <= tau and each subject j at risk
    (T_j > T_i), the pair is concordant when risk_i > risk_j.
    Each pair is weighted by 1 / G(T_i)^2 to correct for censoring.
    """
    n = len(time)
    if n < 2:
        return float("nan")

    km_t, km_g = _km_censoring_survival(time, event)
    g_ti = np.clip(_step_value(km_t, km_g, time), 1e-8, 1.0)

    case_idx = np.where((event == 1) & (time <= horizon))[0]
    if len(case_idx) == 0:
        return float("nan")

    numerator = 0.0
    denominator = 0.0
    for i in case_idx:
        at_risk = time > time[i]
        if not at_risk.any():
            continue
        w = 1.0 / (g_ti[i] ** 2)
        r_j = risk[at_risk]
        concordant = float(np.sum(risk[i] > r_j))
        tied = float(np.sum(risk[i] == r_j))
        n_pairs = float(at_risk.sum())
        numerator += w * (concordant + 0.5 * tied)
        denominator += w * n_pairs

    return float(numerator / denominator) if denominator > 0 else float("nan")


def binary_concordance(y_true: np.ndarray, risk: np.ndarray) -> float:
    """Standard concordance index for binary outcomes (no censoring)."""
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    numerator = 0.0
    total = 0.0
    for i in pos:
        r_neg = risk[neg]
        numerator += float(np.sum(risk[i] > r_neg)) + 0.5 * float(np.sum(risk[i] == r_neg))
        total += float(len(r_neg))
    return float(numerator / total) if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Metric 2: IPCW-weighted classification at threshold
# ---------------------------------------------------------------------------

def classification_at_threshold(
    y_true: np.ndarray,
    risk: np.ndarray,
    sample_weight: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """PPV, NPV, sensitivity, specificity using IPCW weights."""
    pred_pos = risk >= threshold
    pred_neg = ~pred_pos
    w = sample_weight.astype(float)

    tp = float(np.sum(w * pred_pos * (y_true == 1)))
    fp = float(np.sum(w * pred_pos * (y_true == 0)))
    tn = float(np.sum(w * pred_neg * (y_true == 0)))
    fn = float(np.sum(w * pred_neg * (y_true == 1)))

    return {
        "threshold": threshold,
        "PPV": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else float("nan"),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Metric 3: NRI
# ---------------------------------------------------------------------------

def compute_nri(
    y_true: np.ndarray,
    risk_new: np.ndarray,
    risk_old: np.ndarray,
    cutoffs: np.ndarray,
) -> Dict[str, object]:
    """NRI via reclassification package (unweighted)."""
    try:
        from reclassification.reclassification import calculate_nri
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "reclassification is required for NRI. "
            "Install: pip install reclassification"
        ) from exc
    y = y_true.astype(int).tolist()
    rn = risk_new.astype(float).tolist()
    ro = risk_old.astype(float).tolist()
    results: Dict[str, float] = {}
    for c in cutoffs:
        results[f"{float(c):.6f}"] = float(calculate_nri(y, rn, ro, float(c)))
    return {"cutoff_nri": results}


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_metrics(
    time_all: Optional[np.ndarray],
    event_all: Optional[np.ndarray],
    risks_all: Optional[Dict[str, np.ndarray]],
    y_ipcw: np.ndarray,
    w_ipcw: np.ndarray,
    risks_ipcw: Dict[str, np.ndarray],
    horizon: Optional[float],
    is_survival: bool,
    ppv_npv_thresholds: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Bootstrap 95 % CIs for C-td, Brier, PPV, and NPV.

    Returns ``{model_label: {metric_key: (ci_low, ci_high)}}``.
    """
    rng = np.random.default_rng(seed)
    metric_keys: List[str] = ["C_td", "Brier"]
    for t in ppv_npv_thresholds:
        metric_keys += [f"PPV@{t:.4f}", f"NPV@{t:.4f}"]

    collectors: Dict[str, Dict[str, List[float]]] = {
        label: {k: [] for k in metric_keys} for label in risks_ipcw
    }

    n_ipcw = len(y_ipcw)
    n_all = len(time_all) if time_all is not None else n_ipcw

    for _ in range(n_boot):
        # IPCW-subset resample for calibration / classification
        idx_ipcw = rng.integers(0, n_ipcw, size=n_ipcw)
        y_b = y_ipcw[idx_ipcw]
        w_b = w_ipcw[idx_ipcw]

        # Full-data resample for C-td (survival only)
        idx_all = rng.integers(0, n_all, size=n_all) if is_survival else None

        for label in risks_ipcw:
            r_ipcw_b = risks_ipcw[label][idx_ipcw]

            # C-td
            if is_survival and time_all is not None and event_all is not None and idx_all is not None:
                c_val = unos_c_td(
                    time_all[idx_all], event_all[idx_all],
                    risks_all[label][idx_all], horizon,
                )
            else:
                c_val = binary_concordance(y_b, r_ipcw_b)
            collectors[label]["C_td"].append(c_val)

            # Brier
            brier = float(np.sum(w_b * (r_ipcw_b - y_b) ** 2) / np.sum(w_b))
            collectors[label]["Brier"].append(brier)

            # PPV / NPV
            for t in ppv_npv_thresholds:
                cls = classification_at_threshold(y_b, r_ipcw_b, w_b, t)
                collectors[label][f"PPV@{t:.4f}"].append(cls["PPV"])
                collectors[label][f"NPV@{t:.4f}"].append(cls["NPV"])

    ci: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for label, metrics in collectors.items():
        ci[label] = {}
        for key, values in metrics.items():
            arr = np.array(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                ci[label][key] = (
                    float(np.percentile(arr, 2.5)),
                    float(np.percentile(arr, 97.5)),
                )
            else:
                ci[label][key] = (float("nan"), float("nan"))
    return ci


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _fmt(val, decimals: int = 3) -> str:
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "--"
    return f"{float(val):.{decimals}f}"


def _fmt_ci(
    val, ci: Optional[Tuple[float, float]], decimals: int = 3,
) -> str:
    base = _fmt(val, decimals)
    if base == "--" or ci is None:
        return base
    lo, hi = ci
    if not np.isfinite(lo) or not np.isfinite(hi):
        return base
    return f"{base} ({_fmt(lo, decimals)}\u2013{_fmt(hi, decimals)})"


def build_wide_table(
    model_labels: List[str],
    c_td: Dict[str, float],
    calibration: Dict[str, Dict[str, Optional[float]]],
    ppv_npv: Dict[str, Dict[float, Dict[str, float]]],
    nri_vals: Dict[str, Optional[Dict]],
    nri_reference: Optional[str],
    ci: Optional[Dict[str, Dict[str, Tuple[float, float]]]],
    ppv_npv_thresholds: np.ndarray,
) -> pd.DataFrame:
    """One row per model, all metrics as columns."""
    rows = []
    for label in model_labels:
        row: Dict[str, str] = {"Model": label}

        c_ci = ci[label].get("C_td") if ci else None
        row["C_td"] = _fmt_ci(c_td[label], c_ci)

        cal = calibration[label]
        row["Cal_Intercept"] = _fmt(cal.get("calibration_intercept"))
        row["Cal_Slope"] = _fmt(cal.get("calibration_slope"))

        brier_ci = ci[label].get("Brier") if ci else None
        row["Brier"] = _fmt_ci(cal.get("brier_score", float("nan")), brier_ci)

        for t in ppv_npv_thresholds:
            cls = ppv_npv[label][t]
            ppv_ci = ci[label].get(f"PPV@{t:.4f}") if ci else None
            npv_ci = ci[label].get(f"NPV@{t:.4f}") if ci else None
            pct = f"{t:.0%}"
            row[f"PPV@{pct}"] = _fmt_ci(cls["PPV"], ppv_ci)
            row[f"NPV@{pct}"] = _fmt_ci(cls["NPV"], npv_ci)

        if nri_reference is not None:
            if label == nri_reference:
                row["NRI"] = "ref"
            elif nri_vals.get(label) is not None:
                cutoff_nri = nri_vals[label]["cutoff_nri"]
                first_key = next(iter(cutoff_nri))
                row["NRI"] = _fmt(cutoff_nri[first_key])
            else:
                row["NRI"] = "--"

        rows.append(row)
    return pd.DataFrame(rows)


def build_classification_table(
    model_labels: List[str],
    ppv_npv: Dict[str, Dict[float, Dict[str, float]]],
    ppv_npv_thresholds: np.ndarray,
) -> pd.DataFrame:
    """Long format: one row per (model, threshold)."""
    rows = []
    for label in model_labels:
        for t in ppv_npv_thresholds:
            cls = ppv_npv[label][t]
            rows.append({
                "Model": label,
                "Threshold": t,
                "PPV": _fmt(cls["PPV"]),
                "NPV": _fmt(cls["NPV"]),
                "Sensitivity": _fmt(cls["sensitivity"]),
                "Specificity": _fmt(cls["specificity"]),
            })
    return pd.DataFrame(rows)


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def wide_table_to_latex(
    df: pd.DataFrame,
    caption: str = "Model performance comparison",
    label: str = "tab:performance",
) -> str:
    cols = list(df.columns)
    col_spec = "l" + "c" * (len(cols) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(_latex_escape(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        vals = " & ".join(_latex_escape(str(row[c])) for c in cols) + r" \\"
        lines.append(vals)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Performance comparison table for survival / binary risk models.  "
            "Computes C-td (Uno), calibration (intercept, slope, Brier), "
            "PPV/NPV at thresholds, and optional NRI."
        ),
    )

    p.add_argument(
        "--mode", choices=["survival", "binary"], default="survival",
        help="Analysis mode (default: survival).",
    )
    p.add_argument("--data-path", required=True, help="Path to CSV or Parquet input.")
    p.add_argument("--outcome-col", default="", help="Binary outcome column (binary mode).")
    p.add_argument("--time-col", default="", help="Time-to-event column (survival mode).")
    p.add_argument("--event-col", default="", help="Event indicator column (survival mode).")
    p.add_argument(
        "--horizon", type=float, default=None,
        help="Prediction horizon in same units as time column (survival mode).",
    )
    p.add_argument("--event-value", default="1", help="Value treated as event (default: '1').")
    p.add_argument(
        "--model", action="append", required=True,
        help="Model risk spec (repeatable): 'Label:column' or 'column'.",
    )
    p.add_argument("--clip-risk", action="store_true", help="Clip risk values to [0,1].")
    p.add_argument(
        "--ppv-npv-thresholds", default="0.05,0.10,0.15",
        help="Comma-separated thresholds for PPV/NPV (default: 0.05,0.10,0.15).",
    )
    p.add_argument(
        "--calibration-bins", type=int, default=10,
        help="Number of bins for calibration (default: 10).",
    )
    p.add_argument(
        "--calibration-strategy", choices=["equal_width", "quantile"],
        default="equal_width", help="Binning strategy (default: equal_width).",
    )

    p.add_argument("--nri", action="store_true", help="Compute NRI versus a reference model.")
    p.add_argument(
        "--nri-reference", default="",
        help="Reference model label for NRI (default: first --model).",
    )
    p.add_argument(
        "--nri-cutoffs", default="0",
        help="Comma-separated cutoffs for NRI (0 = continuous; default: 0).",
    )

    p.add_argument(
        "--bootstrap", type=int, default=0,
        help="Bootstrap iterations for 95%% CIs (default: 0 = off).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument(
        "--latex", action="store_true",
        help="Emit a LaTeX table (.tex) alongside CSV.",
    )
    p.add_argument(
        "--latex-caption", default="Model performance comparison",
        help="Caption for LaTeX table.",
    )
    p.add_argument(
        "--latex-label", default="tab:performance",
        help="Label for LaTeX table.",
    )
    p.add_argument(
        "--outdir", default="analysis/dca/output_perf",
        help="Output directory (default: analysis/dca/output_perf).",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "binary":
        if not args.outcome_col:
            raise SystemExit("Error: --outcome-col is required in binary mode.")
    else:
        if not args.time_col or not args.event_col:
            raise SystemExit("Error: --time-col and --event-col are required in survival mode.")
        if args.horizon is None:
            raise SystemExit("Error: --horizon is required in survival mode.")
        if args.horizon <= 0:
            raise SystemExit("Error: --horizon must be positive.")

    models = _parse_model_specs(args.model)
    model_labels = list(models.keys())

    ppv_npv_thresholds = np.array(
        sorted({float(x.strip()) for x in args.ppv_npv_thresholds.split(",") if x.strip()})
    )
    if len(ppv_npv_thresholds) == 0:
        raise SystemExit("Error: --ppv-npv-thresholds must have at least one value.")
    if np.any(ppv_npv_thresholds <= 0) or np.any(ppv_npv_thresholds >= 1):
        raise SystemExit("Error: PPV/NPV thresholds must be strictly between 0 and 1.")

    nri_cutoffs = np.array([], dtype=float)
    nri_reference: Optional[str] = None
    if args.nri:
        nri_cutoffs = np.array(
            sorted({float(x.strip()) for x in args.nri_cutoffs.split(",") if x.strip()}),
            dtype=float,
        )
        if len(nri_cutoffs) == 0:
            raise SystemExit("Error: --nri-cutoffs must have at least one value when --nri is set.")
        nri_reference = args.nri_reference.strip() or model_labels[0]
        if nri_reference not in models:
            raise SystemExit(
                f"Error: --nri-reference '{nri_reference}' is not among model labels: {model_labels}"
            )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load & prepare ────────────────────────────────────────────────
    df = _load_table(Path(args.data_path))
    is_survival = args.mode == "survival"

    time_all: Optional[np.ndarray] = None
    event_all: Optional[np.ndarray] = None
    risks_all: Optional[Dict[str, np.ndarray]] = None

    if is_survival:
        time_all, event_all, risks_all, y_ipcw, w_ipcw, risks_ipcw = (
            prepare_survival(
                df, args.time_col, args.event_col, args.event_value,
                args.horizon, models, args.clip_risk,
            )
        )
    else:
        y_ipcw, w_ipcw, risks_ipcw = prepare_binary(
            df, args.outcome_col, args.event_value, models, args.clip_risk,
        )

    # ── Compute point estimates ───────────────────────────────────────
    c_td_vals: Dict[str, float] = {}
    cal_vals: Dict[str, Dict[str, Optional[float]]] = {}
    ppv_npv_vals: Dict[str, Dict[float, Dict[str, float]]] = {}
    nri_vals: Dict[str, Optional[Dict]] = {}

    cal_analyzer = CalibrationAnalyzer(CalibrationConfig(
        bins=args.calibration_bins,
        strategy=args.calibration_strategy,
        make_plot=False,
        plot_name="",
        plot_title="",
    ))

    for label in model_labels:
        risk_ipcw = risks_ipcw[label]

        # C-td (Uno) -- uses ALL subjects in survival mode
        if is_survival:
            c_td_vals[label] = unos_c_td(
                time_all, event_all, risks_all[label], args.horizon,
            )
        else:
            c_td_vals[label] = binary_concordance(y_ipcw, risk_ipcw)

        # Calibration -- uses IPCW subset
        cal_vals[label] = cal_analyzer.calibration_metrics(
            y_ipcw, risk_ipcw, w_ipcw,
        )

        # PPV / NPV at each threshold -- uses IPCW subset
        ppv_npv_vals[label] = {}
        for t in ppv_npv_thresholds:
            ppv_npv_vals[label][t] = classification_at_threshold(
                y_ipcw, risk_ipcw, w_ipcw, t,
            )

        # NRI
        if args.nri:
            if label == nri_reference:
                nri_vals[label] = None
            else:
                nri_vals[label] = compute_nri(
                    y_ipcw, risk_ipcw, risks_ipcw[nri_reference], nri_cutoffs,
                )

    # ── Bootstrap CIs ─────────────────────────────────────────────────
    ci = None
    if args.bootstrap > 0:
        print(f"Running {args.bootstrap} bootstrap iterations …")
        ci = _bootstrap_metrics(
            time_all=time_all,
            event_all=event_all,
            risks_all=risks_all,
            y_ipcw=y_ipcw,
            w_ipcw=w_ipcw,
            risks_ipcw=risks_ipcw,
            horizon=args.horizon,
            is_survival=is_survival,
            ppv_npv_thresholds=ppv_npv_thresholds,
            n_boot=args.bootstrap,
            seed=args.seed,
        )

    # ── Build & write tables ──────────────────────────────────────────
    perf_df = build_wide_table(
        model_labels, c_td_vals, cal_vals, ppv_npv_vals,
        nri_vals, nri_reference, ci, ppv_npv_thresholds,
    )
    cls_df = build_classification_table(
        model_labels, ppv_npv_vals, ppv_npv_thresholds,
    )

    perf_path = outdir / "performance_table.csv"
    cls_path = outdir / "classification_by_threshold.csv"
    perf_df.to_csv(perf_path, index=False)
    cls_df.to_csv(cls_path, index=False)

    print(f"\nPerformance table  -> {perf_path}")
    print(f"Classification     -> {cls_path}")
    print()
    print(perf_df.to_string(index=False))

    if args.latex:
        tex = wide_table_to_latex(
            perf_df, caption=args.latex_caption, label=args.latex_label,
        )
        tex_path = outdir / "performance_table.tex"
        tex_path.write_text(tex, encoding="utf-8")
        print(f"\nLaTeX table        -> {tex_path}")

    # ── JSON summary ──────────────────────────────────────────────────
    summary: Dict[str, object] = {
        "mode": args.mode,
        "n_rows_ipcw": int(len(y_ipcw)),
        "weighted_event_rate": float(np.sum(w_ipcw * y_ipcw) / np.sum(w_ipcw)),
        "ppv_npv_thresholds": ppv_npv_thresholds.tolist(),
        "models": {},
    }
    if is_survival:
        summary["n_rows_total"] = int(len(time_all))
        summary["horizon"] = args.horizon

    for label in model_labels:
        entry: Dict[str, object] = {
            "risk_column": models[label],
            "C_td": c_td_vals[label],
            "calibration": cal_vals[label],
            "classification": {
                f"{t:.4f}": ppv_npv_vals[label][t] for t in ppv_npv_thresholds
            },
        }
        if args.nri:
            if label == nri_reference:
                entry["nri"] = {"is_reference": True}
            elif nri_vals.get(label) is not None:
                entry["nri"] = nri_vals[label]
        if ci:
            entry["bootstrap_ci"] = {k: list(v) for k, v in ci[label].items()}
        summary["models"][label] = entry

    json_path = outdir / "performance_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"JSON summary       -> {json_path}")


if __name__ == "__main__":
    main()
