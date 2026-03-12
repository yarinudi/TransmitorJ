from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

plt = None
nri_calculate = None


def _get_plt():
    global plt
    if plt is None:
        try:
            plt = importlib.import_module("matplotlib.pyplot")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required for plotting. Install matplotlib or run with --no-plot."
            ) from exc
    return plt


def _get_nri_calculate() -> Callable[[Sequence[int], Sequence[float], Sequence[float], float], float]:
    global nri_calculate
    if nri_calculate is None:
        try:
            from reclassification.reclassification import calculate_nri as _calculate_nri
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "reclassification is required for NRI. Install it with: pip install reclassification"
            ) from exc
        nri_calculate = _calculate_nri
    return nri_calculate


@dataclass
class DCAConfig:
    data_path: Path
    mode: str
    outcome_col: Optional[str]
    time_col: Optional[str]
    event_col: Optional[str]
    horizon: Optional[float]
    model_specs: List[str]
    thresholds: np.ndarray
    outdir: Path
    event_value: str
    bootstrap: int
    seed: int
    clip_risk: bool
    make_plot: bool
    figure_name: str
    figure_title: str
    run_calibration: bool
    calibration_bins: int
    calibration_strategy: str
    calibration_plot_name: str
    calibration_plot_title: str
    run_nri: bool
    nri_reference: Optional[str]
    nri_cutoffs: np.ndarray


@dataclass
class CalibrationConfig:
    bins: int
    strategy: str
    make_plot: bool
    plot_name: str
    plot_title: str


@dataclass
class PreparedData:
    y_true: np.ndarray
    sample_weight: np.ndarray
    risks: Dict[str, np.ndarray]
    n_rows_used: int
    weighted_event_rate: float


class CalibrationAnalyzer:
    """Reusable calibration diagnostics for binary risk models."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        if self.config.bins < 2:
            raise ValueError("--calibration-bins must be >= 2.")
        if self.config.strategy not in {"equal_width", "quantile"}:
            raise ValueError("--calibration-strategy must be one of: equal_width, quantile.")

    @staticmethod
    def _brier_score(y_true: np.ndarray, risk: np.ndarray, sample_weight: np.ndarray) -> float:
        sw = np.asarray(sample_weight, dtype=float)
        return float(np.sum(sw * (risk - y_true) ** 2) / np.sum(sw))

    @staticmethod
    def _safe_logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _fit_logistic_irls(
        y: np.ndarray,
        x: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> Optional[np.ndarray]:
        """
        Fit logistic regression (intercept + one predictor) with IRLS.
        Returns [intercept, slope] or None if fitting fails.
        """
        n = len(y)
        X = np.column_stack([np.ones(n), x])
        beta = np.zeros(2, dtype=float)
        sw = np.ones(n, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)

        for _ in range(max_iter):
            eta = X @ beta
            mu = 1.0 / (1.0 + np.exp(-eta))
            mu = np.clip(mu, 1e-8, 1.0 - 1e-8)
            base_w = mu * (1.0 - mu)
            w = sw * base_w
            z = eta + (y - mu) / base_w
            XtWX = X.T @ (w[:, None] * X)
            XtWz = X.T @ (w * z)

            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                return None

            if np.max(np.abs(beta_new - beta)) < tol:
                return beta_new
            beta = beta_new

        return beta

    @staticmethod
    def _calibration_intercept_in_the_large(
        y_true: np.ndarray,
        risk: np.ndarray,
        sample_weight: np.ndarray,
    ) -> float:
        # Intercept with slope fixed at 1; robust fallback if slope fit is unstable.
        eps = 1e-12
        sw = np.asarray(sample_weight, dtype=float)
        event_rate = np.clip(float(np.sum(sw * y_true) / np.sum(sw)), eps, 1.0 - eps)
        mean_pred = np.clip(float(np.sum(sw * risk) / np.sum(sw)), eps, 1.0 - eps)
        return float(np.log(event_rate / (1.0 - event_rate)) - np.log(mean_pred / (1.0 - mean_pred)))

    def calibration_metrics(
        self,
        y_true: np.ndarray,
        risk: np.ndarray,
        sample_weight: np.ndarray,
    ) -> Dict[str, Optional[float]]:
        risk_logit = self._safe_logit(risk)
        can_fit_slope = bool(np.std(risk_logit) > 0)
        slope: Optional[float] = None
        intercept: Optional[float] = None
        fit_status = "fallback_intercept_only"

        if can_fit_slope:
            beta = self._fit_logistic_irls(y_true.astype(float), risk_logit, sample_weight=sample_weight)
            if beta is not None:
                # Guard against unstable fits (e.g., near-complete separation in tiny samples).
                if np.all(np.isfinite(beta)) and np.max(np.abs(beta)) <= 20:
                    intercept = float(beta[0])
                    slope = float(beta[1])
                    fit_status = "full_intercept_and_slope"
                else:
                    fit_status = "unstable_full_fit_fallback_intercept_only"

        if intercept is None:
            intercept = self._calibration_intercept_in_the_large(y_true, risk, sample_weight)

        return {
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "brier_score": self._brier_score(y_true, risk, sample_weight),
            "fit_status": fit_status,
        }

    def calibration_curve(
        self,
        y_true: np.ndarray,
        risk: np.ndarray,
        sample_weight: np.ndarray,
    ) -> pd.DataFrame:
        tmp = pd.DataFrame(
            {
                "y": y_true.astype(float),
                "risk": risk.astype(float),
                "weight": np.asarray(sample_weight, dtype=float),
            }
        )

        if self.config.strategy == "quantile":
            tmp["bin"] = pd.qcut(tmp["risk"], q=self.config.bins, duplicates="drop")
        else:
            # equal_width
            edges = np.linspace(0.0, 1.0, self.config.bins + 1)
            tmp["bin"] = pd.cut(tmp["risk"], bins=edges, include_lowest=True)

        tmp["wy"] = tmp["weight"] * tmp["y"]
        tmp["wr"] = tmp["weight"] * tmp["risk"]
        grouped = tmp.groupby("bin", observed=True)
        curve = grouped.agg(
            n=("weight", "sum"),
            wy=("wy", "sum"),
            wr=("wr", "sum"),
            min_risk=("risk", "min"),
            max_risk=("risk", "max"),
        ).reset_index(drop=True)
        curve["observed_event_rate"] = curve["wy"] / curve["n"]
        curve["mean_predicted_risk"] = curve["wr"] / curve["n"]
        curve = curve.drop(columns=["wy", "wr"])

        return curve.sort_values("mean_predicted_risk").reset_index(drop=True)

    def plot(self, curves_by_model: Dict[str, pd.DataFrame], outdir: Path) -> Path:
        plt_mod = _get_plt()
        plt_mod.figure(figsize=(8.5, 5.5))
        plt_mod.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

        for label, curve in curves_by_model.items():
            plt_mod.plot(
                curve["mean_predicted_risk"],
                curve["observed_event_rate"],
                marker="o",
                linewidth=1.8,
                label=label,
            )

        plt_mod.xlabel("Mean predicted risk")
        plt_mod.ylabel("Observed event rate")
        plt_mod.title(self.config.plot_title)
        plt_mod.legend()
        plt_mod.tight_layout()

        plot_path = outdir / self.config.plot_name
        plt_mod.savefig(plot_path, dpi=300)
        plt_mod.close()
        return plot_path


class DecisionCurveAnalyzer:
    """Reusable Decision Curve Analysis runner for binary outcomes."""

    def __init__(self, config: DCAConfig) -> None:
        self.config = config
        self.models: Dict[str, str] = self._parse_model_specs(config.model_specs)

    @staticmethod
    def _parse_model_specs(model_specs: Sequence[str]) -> Dict[str, str]:
        models: Dict[str, str] = {}
        for spec in model_specs:
            if ":" in spec:
                label, col = spec.split(":", 1)
                label = label.strip()
                col = col.strip()
            else:
                col = spec.strip()
                label = col

            if not label or not col:
                raise ValueError(f"Invalid model spec: '{spec}'. Use 'Label:column' or 'column'.")
            models[label] = col

        if not models:
            raise ValueError("At least one --model must be provided.")
        return models

    @staticmethod
    def _load_table(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file type '{suffix}'. Use CSV or Parquet.")

    def _prepare_data(self, df: pd.DataFrame) -> PreparedData:
        if self.config.mode == "binary":
            return self._prepare_binary_data(df)
        return self._prepare_survival_data(df)

    def _extract_and_validate_risks(self, work: pd.DataFrame) -> Dict[str, np.ndarray]:
        risks: Dict[str, np.ndarray] = {}
        for label, col in self.models.items():
            r = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
            if np.isnan(r).any():
                raise ValueError(f"Non-numeric values found in risk column '{col}'.")
            if self.config.clip_risk:
                r = np.clip(r, 0.0, 1.0)
            elif (r.min() < 0.0) or (r.max() > 1.0):
                raise ValueError(
                    f"Risk column '{col}' has values outside [0, 1]. "
                    "Use --clip-risk to force clipping."
                )
            risks[label] = r
        return risks

    def _prepare_binary_data(self, df: pd.DataFrame) -> PreparedData:
        if not self.config.outcome_col:
            raise ValueError("--outcome-col is required in binary mode.")
        needed_cols = [self.config.outcome_col] + list(self.models.values())
        missing_cols = [c for c in needed_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        work = df[needed_cols].copy().dropna()
        if work.empty:
            raise ValueError("No rows remain after dropping NA values in required columns.")

        y_raw = work[self.config.outcome_col]
        y = (y_raw.astype(str) == str(self.config.event_value)).astype(int).to_numpy()
        if y.sum() == 0:
            raise ValueError("No positive events found after outcome mapping. Check --event-value.")
        if y.sum() == len(y):
            raise ValueError("All rows are events after outcome mapping; DCA is not informative.")

        risks = self._extract_and_validate_risks(work)
        sample_weight = np.ones_like(y, dtype=float)
        return PreparedData(
            y_true=y.astype(float),
            sample_weight=sample_weight,
            risks=risks,
            n_rows_used=int(len(y)),
            weighted_event_rate=float(y.mean()),
        )

    @staticmethod
    def _km_survival_for_censoring(times: np.ndarray, censor_event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kaplan-Meier survival of censoring distribution G(t)=P(C>t).
        censor_event: 1 if censored at time, 0 otherwise.
        """
        order = np.argsort(times)
        t = times[order]
        d = censor_event[order]
        unique_t = np.unique(t)
        surv_values = []
        surv = 1.0
        n = len(t)

        for tt in unique_t:
            at_risk = n - np.searchsorted(t, tt, side="left")
            d_cens = np.sum((t == tt) & (d == 1))
            if at_risk > 0:
                surv *= (1.0 - d_cens / at_risk)
            surv_values.append(surv)

        return unique_t, np.asarray(surv_values, dtype=float)

    @staticmethod
    def _step_value(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
        # Right-continuous step function values: y(x_i) for max x_i <= q; 1.0 before first x.
        idx = np.searchsorted(x, query, side="right") - 1
        out = np.ones_like(query, dtype=float)
        valid = idx >= 0
        out[valid] = y[idx[valid]]
        return out

    def _prepare_survival_data(self, df: pd.DataFrame) -> PreparedData:
        if self.config.horizon is None:
            raise ValueError("--horizon is required in survival mode.")
        if not self.config.time_col or not self.config.event_col:
            raise ValueError("--time-col and --event-col are required in survival mode.")

        needed_cols = [self.config.time_col, self.config.event_col] + list(self.models.values())
        missing_cols = [c for c in needed_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        work = df[needed_cols].copy().dropna()
        if work.empty:
            raise ValueError("No rows remain after dropping NA values in required columns.")

        time = pd.to_numeric(work[self.config.time_col], errors="coerce").to_numpy(dtype=float)
        event = (work[self.config.event_col].astype(str) == str(self.config.event_value)).astype(int).to_numpy()
        if np.isnan(time).any():
            raise ValueError(f"Non-numeric values found in time column '{self.config.time_col}'.")
        if np.any(time < 0):
            raise ValueError("Time values must be non-negative.")

        # IPCW targets at horizon t.
        horizon = float(self.config.horizon)
        censor_event = (event == 0).astype(int)
        km_t, km_g = self._km_survival_for_censoring(time, censor_event)
        g_ti = np.clip(self._step_value(km_t, km_g, time), 1e-8, 1.0)
        g_h = float(np.clip(self._step_value(km_t, km_g, np.array([horizon]))[0], 1e-8, 1.0))

        is_case_by_h = (event == 1) & (time <= horizon)
        is_control_by_h = time > horizon
        usable = is_case_by_h | is_control_by_h
        if usable.sum() == 0:
            raise ValueError("No usable rows after horizon/IPCW filtering. Check horizon and data coverage.")

        y_ipcw = np.where(is_case_by_h, 1.0, 0.0)
        w_ipcw = np.where(is_case_by_h, 1.0 / g_ti, np.where(is_control_by_h, 1.0 / g_h, 0.0))

        y_ipcw = y_ipcw[usable]
        w_ipcw = w_ipcw[usable]
        if np.sum(y_ipcw) == 0:
            raise ValueError("No events observed by horizon after IPCW setup.")

        risks_all = self._extract_and_validate_risks(work)
        risks = {k: v[usable] for k, v in risks_all.items()}
        weighted_event_rate = float(np.sum(w_ipcw * y_ipcw) / np.sum(w_ipcw))
        return PreparedData(
            y_true=y_ipcw,
            sample_weight=w_ipcw,
            risks=risks,
            n_rows_used=int(np.sum(usable)),
            weighted_event_rate=weighted_event_rate,
        )

    @staticmethod
    def dca_curve(
        y_true: np.ndarray,
        risk: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
    ) -> pd.DataFrame:
        sw = np.asarray(sample_weight, dtype=float)
        n_eff = float(np.sum(sw))
        prevalence = float(np.sum(sw * y_true) / n_eff)
        out = []

        for pt in thresholds:
            pred_pos = risk >= pt
            tp = float(np.sum(sw * pred_pos * (y_true == 1)))
            fp = float(np.sum(sw * pred_pos * (y_true == 0)))
            weight = pt / (1.0 - pt)

            out.append(
                {
                    "threshold": pt,
                    "nb_model": (tp / n_eff) - (fp / n_eff) * weight,
                    "nb_all": prevalence - (1.0 - prevalence) * weight,
                    "nb_none": 0.0,
                }
            )

        return pd.DataFrame(out)

    @staticmethod
    def _bootstrap_ci(
        y_true: np.ndarray,
        risk: np.ndarray,
        thresholds: np.ndarray,
        sample_weight: np.ndarray,
        n_boot: int,
        seed: int,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = len(y_true)
        curves = []

        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_curve = DecisionCurveAnalyzer.dca_curve(y_true[idx], risk[idx], thresholds, sample_weight[idx])
            curves.append(boot_curve["nb_model"].to_numpy())

        mat = np.vstack(curves)
        return pd.DataFrame(
            {
                "threshold": thresholds,
                "nb_model_mean": mat.mean(axis=0),
                "nb_model_ci_low": np.percentile(mat, 2.5, axis=0),
                "nb_model_ci_high": np.percentile(mat, 97.5, axis=0),
            }
        )

    @staticmethod
    def _intervals_from_mask(thresholds: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
        intervals: List[Tuple[float, float]] = []
        start_idx = None
        for i, is_true in enumerate(mask):
            if is_true and start_idx is None:
                start_idx = i
            if not is_true and start_idx is not None:
                intervals.append((float(thresholds[start_idx]), float(thresholds[i - 1])))
                start_idx = None
        if start_idx is not None:
            intervals.append((float(thresholds[start_idx]), float(thresholds[-1])))
        return intervals

    def _resolve_nri_reference_label(self, explicit_reference: Optional[str]) -> str:
        if explicit_reference:
            ref = explicit_reference
            if ref not in self.models:
                raise ValueError(
                    f"--nri-reference '{ref}' is not one of the provided model labels: {list(self.models.keys())}"
                )
            return ref
        # Default to first model in CLI order.
        return next(iter(self.models.keys()))

    def _nri_for_model_pair(
        self,
        y_true: np.ndarray,
        risk_new: np.ndarray,
        risk_old: np.ndarray,
        cutoffs: np.ndarray,
    ) -> Dict[str, object]:
        calculate_nri = _get_nri_calculate()
        y = y_true.astype(int).tolist()
        risk_new_list = risk_new.astype(float).tolist()
        risk_old_list = risk_old.astype(float).tolist()
        by_cutoff: Dict[str, float] = {}
        for cutoff in cutoffs:
            value = float(
                calculate_nri(
                    outcome=y,
                    prob_new=risk_new_list,
                    prob_old=risk_old_list,
                    cutoff=float(cutoff),
                )
            )
            by_cutoff[f"{float(cutoff):.6f}"] = value
        return {
            "method": "reclassification.calculate_nri",
            "weighting": "unweighted",
            "cutoff_nri": by_cutoff,
        }

    def run(self) -> Dict[str, object]:
        df = self._load_table(self.config.data_path)
        prepared = self._prepare_data(df)
        y_true = prepared.y_true
        sample_weight = prepared.sample_weight
        risks = prepared.risks
        self.config.outdir.mkdir(parents=True, exist_ok=True)

        all_rows = []
        summary: Dict[str, object] = {
            "mode": self.config.mode,
            "n_rows_used": prepared.n_rows_used,
            "weighted_event_rate": prepared.weighted_event_rate,
            "models": {},
        }
        if self.config.mode == "binary":
            summary["outcome_column"] = self.config.outcome_col
            summary["event_value"] = self.config.event_value
        else:
            summary["time_column"] = self.config.time_col
            summary["event_column"] = self.config.event_col
            summary["event_value"] = self.config.event_value
            summary["horizon"] = self.config.horizon
        nri_reference_label: Optional[str] = None
        if self.config.run_nri:
            nri_reference_label = self._resolve_nri_reference_label(self.config.nri_reference)
            summary["nri"] = {
                "reference_model": nri_reference_label,
                "cutoffs": [float(x) for x in self.config.nri_cutoffs.tolist()],
                "method": "reclassification.calculate_nri",
                "weighting": "unweighted",
                "note": (
                    "NRI is computed on binary outcomes without IPCW/sample-weight adjustment. "
                    "Interpret survival-mode NRI as exploratory."
                ),
            }
        calibration_curves_by_model: Dict[str, pd.DataFrame] = {}
        calibration_metrics_by_model: Dict[str, Dict[str, Optional[float]]] = {}
        calibration_plot_path: Optional[str] = None
        calibration_runner: Optional[CalibrationAnalyzer] = None

        if self.config.run_calibration:
            calibration_runner = CalibrationAnalyzer(
                CalibrationConfig(
                    bins=self.config.calibration_bins,
                    strategy=self.config.calibration_strategy,
                    make_plot=self.config.make_plot,
                    plot_name=self.config.calibration_plot_name,
                    plot_title=self.config.calibration_plot_title,
                )
            )

        for label, risk in risks.items():
            curve = self.dca_curve(y_true, risk, self.config.thresholds, sample_weight)
            curve["model"] = label
            all_rows.append(curve.assign(strategy=label, net_benefit=curve["nb_model"])[["threshold", "strategy", "net_benefit", "model"]])
            all_rows.append(curve.assign(strategy="Treat all", net_benefit=curve["nb_all"], model=label)[["threshold", "strategy", "net_benefit", "model"]])
            all_rows.append(curve.assign(strategy="Treat none", net_benefit=curve["nb_none"], model=label)[["threshold", "strategy", "net_benefit", "model"]])

            better_mask = (curve["nb_model"] > curve["nb_all"]) & (curve["nb_model"] > curve["nb_none"])
            intervals = self._intervals_from_mask(curve["threshold"].to_numpy(), better_mask.to_numpy())

            model_summary: Dict[str, object] = {
                "risk_column": self.models[label],
                "better_than_treat_all_and_none_threshold_intervals": intervals,
                "max_net_benefit": float(curve["nb_model"].max()),
            }

            if self.config.run_nri:
                assert nri_reference_label is not None
                if label == nri_reference_label:
                    model_summary["nri_vs_reference"] = {
                        "reference_model": nri_reference_label,
                        "is_reference_model": True,
                    }
                else:
                    model_summary["nri_vs_reference"] = {
                        "reference_model": nri_reference_label,
                        **self._nri_for_model_pair(
                            y_true=y_true,
                            risk_new=risk,
                            risk_old=risks[nri_reference_label],
                            cutoffs=self.config.nri_cutoffs,
                        ),
                    }

            if self.config.bootstrap > 0:
                ci_curve = self._bootstrap_ci(
                    y_true=y_true,
                    risk=risk,
                    thresholds=self.config.thresholds,
                    sample_weight=sample_weight,
                    n_boot=self.config.bootstrap,
                    seed=self.config.seed,
                )
                ci_path = self.config.outdir / f"dca_ci_{label.replace(' ', '_')}.csv"
                ci_curve.to_csv(ci_path, index=False)
                model_summary["bootstrap_ci_file"] = str(ci_path)

            if calibration_runner is not None:
                model_cal_metrics = calibration_runner.calibration_metrics(y_true, risk, sample_weight)
                model_summary["calibration"] = model_cal_metrics
                calibration_metrics_by_model[label] = model_cal_metrics

                model_curve = calibration_runner.calibration_curve(y_true, risk, sample_weight)
                model_curve_path = self.config.outdir / f"calibration_curve_{label.replace(' ', '_')}.csv"
                model_curve.to_csv(model_curve_path, index=False)
                model_summary["calibration_curve_file"] = str(model_curve_path)
                calibration_curves_by_model[label] = model_curve

            summary["models"][label] = model_summary

        curves_df = pd.concat(all_rows, ignore_index=True)
        curves_path = self.config.outdir / "dca_curves_long.csv"
        curves_df.to_csv(curves_path, index=False)

        summary_path = self.config.outdir / "dca_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if calibration_runner is not None:
            metrics_path = self.config.outdir / "calibration_metrics.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(calibration_metrics_by_model, f, indent=2)
            summary["calibration_metrics_file"] = str(metrics_path)

            if calibration_runner.config.make_plot:
                calibration_plot_path = str(calibration_runner.plot(calibration_curves_by_model, self.config.outdir))
                summary["calibration_plot_file"] = calibration_plot_path

            # Persist final summary again with calibration metadata included.
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        if self.config.make_plot:
            self._plot(curves_df)

        return summary

    def _plot(self, curves_df: pd.DataFrame) -> None:
        plt_mod = _get_plt()
        plt_mod.figure(figsize=(9, 5.5))
        baselines = curves_df[curves_df["strategy"].isin(["Treat all", "Treat none"])]
        first_model = next(iter(self.models.keys()))
        base_first_model = baselines[baselines["model"] == first_model]
        plt_mod.plot(
            base_first_model[base_first_model["strategy"] == "Treat all"]["threshold"],
            base_first_model[base_first_model["strategy"] == "Treat all"]["net_benefit"],
            linestyle="--",
            color="gray",
            label="Treat all",
        )
        plt_mod.plot(
            base_first_model[base_first_model["strategy"] == "Treat none"]["threshold"],
            base_first_model[base_first_model["strategy"] == "Treat none"]["net_benefit"],
            linestyle=":",
            color="black",
            label="Treat none",
        )

        for label in self.models.keys():
            m = curves_df[(curves_df["model"] == label) & (curves_df["strategy"] == label)]
            plt_mod.plot(m["threshold"], m["net_benefit"], linewidth=2, label=label)

        plt_mod.xlabel("Threshold probability")
        plt_mod.ylabel("Net benefit")
        plt_mod.title(self.config.figure_title)
        plt_mod.legend()
        plt_mod.tight_layout()

        fig_path = self.config.outdir / self.config.figure_name
        plt_mod.savefig(fig_path, dpi=300)
        plt_mod.close()


def _parse_thresholds(args: argparse.Namespace) -> np.ndarray:
    if args.thresholds:
        values = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
        thresholds = np.array(sorted(set(values)))
    else:
        thresholds = np.arange(args.threshold_min, args.threshold_max + (args.threshold_step / 2), args.threshold_step)

    if len(thresholds) == 0:
        raise ValueError("No thresholds were generated.")
    if np.any(thresholds <= 0) or np.any(thresholds >= 1):
        raise ValueError("Thresholds must be strictly between 0 and 1.")
    return np.round(thresholds, 6)


def _parse_nri_cutoffs(args: argparse.Namespace) -> np.ndarray:
    if not args.nri:
        return np.array([], dtype=float)
    if not args.nri_cutoffs.strip():
        return np.array([0.0], dtype=float)
    values = [float(x.strip()) for x in args.nri_cutoffs.split(",") if x.strip()]
    if len(values) == 0:
        raise ValueError("When --nri is enabled, --nri-cutoffs must contain at least one numeric value.")
    cutoffs = np.array(sorted(set(values)), dtype=float)
    if np.any(cutoffs < 0):
        raise ValueError("NRI cutoffs must be >= 0.")
    return np.round(cutoffs, 6)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic Decision Curve Analysis runner (survival mode default; binary mode optional)."
    )
    parser.add_argument(
        "--mode",
        choices=["survival", "binary"],
        default="survival",
        help="Analysis mode. Default is survival (IPCW at horizon).",
    )
    parser.add_argument("--data-path", required=True, help="Path to input table (CSV or Parquet).")
    parser.add_argument("--outcome-col", default="", help="Binary outcome column name (binary mode only).")
    parser.add_argument("--time-col", default="", help="Time-to-event column (survival mode).")
    parser.add_argument("--event-col", default="", help="Event indicator column (survival mode).")
    parser.add_argument("--horizon", type=float, default=None, help="Prediction horizon in same units as time column (survival mode).")
    parser.add_argument(
        "--event-value",
        default="1",
        help="Value in outcome column considered as event (default: '1').",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model risk spec, repeatable: 'Label:column' or 'column'.",
    )
    parser.add_argument("--thresholds", default="", help="Optional comma-separated thresholds (e.g., 0.02,0.05,0.1).")
    parser.add_argument("--threshold-min", type=float, default=0.02, help="Min threshold if --thresholds not provided.")
    parser.add_argument("--threshold-max", type=float, default=0.20, help="Max threshold if --thresholds not provided.")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Step for threshold grid.")
    parser.add_argument("--outdir", default="analysis/dca/output", help="Output directory.")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap iterations for model NB CIs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--clip-risk", action="store_true", help="Clip risk values to [0, 1] if needed.")
    parser.add_argument("--no-plot", action="store_true", help="Disable PNG figure output.")
    parser.add_argument("--figure-name", default="dca_curves.png", help="Output figure file name.")
    parser.add_argument("--figure-title", default="Decision Curve Analysis", help="Figure title.")
    parser.add_argument("--no-calibration", action="store_true", help="Disable calibration diagnostics.")
    parser.add_argument("--calibration-bins", type=int, default=10, help="Number of bins for calibration curve.")
    parser.add_argument(
        "--calibration-strategy",
        choices=["equal_width", "quantile"],
        default="equal_width",
        help="Binning strategy for calibration curve.",
    )
    parser.add_argument(
        "--calibration-plot-name",
        default="calibration_plot.png",
        help="Output calibration figure file name.",
    )
    parser.add_argument(
        "--calibration-plot-title",
        default="Calibration Plot",
        help="Calibration plot title.",
    )
    parser.add_argument(
        "--nri",
        action="store_true",
        help="Compute Net Reclassification Improvement (NRI) for each model versus a reference model.",
    )
    parser.add_argument(
        "--nri-reference",
        default="",
        help="Reference model label for NRI (must match a --model label). Default: first --model.",
    )
    parser.add_argument(
        "--nri-cutoffs",
        default="0",
        help=(
            "Comma-separated cutoffs for reclassification.calculate_nri. "
            "Use 0 for continuous NRI (default)."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode == "binary":
        if not args.outcome_col:
            raise ValueError("--outcome-col is required in binary mode.")
    else:
        if not args.time_col or not args.event_col:
            raise ValueError("--time-col and --event-col are required in survival mode.")
        if args.horizon is None:
            raise ValueError("--horizon is required in survival mode.")
        if args.horizon <= 0:
            raise ValueError("--horizon must be positive.")
    if args.nri:
        # Parse early to fail fast on malformed cutoff inputs.
        _parse_nri_cutoffs(args)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _validate_args(args)

    config = DCAConfig(
        data_path=Path(args.data_path),
        mode=args.mode,
        outcome_col=args.outcome_col or None,
        time_col=args.time_col or None,
        event_col=args.event_col or None,
        horizon=args.horizon,
        model_specs=args.model,
        thresholds=_parse_thresholds(args),
        outdir=Path(args.outdir),
        event_value=args.event_value,
        bootstrap=args.bootstrap,
        seed=args.seed,
        clip_risk=args.clip_risk,
        make_plot=not args.no_plot,
        figure_name=args.figure_name,
        figure_title=args.figure_title,
        run_calibration=not args.no_calibration,
        calibration_bins=args.calibration_bins,
        calibration_strategy=args.calibration_strategy,
        calibration_plot_name=args.calibration_plot_name,
        calibration_plot_title=args.calibration_plot_title,
        run_nri=args.nri,
        nri_reference=args.nri_reference or None,
        nri_cutoffs=_parse_nri_cutoffs(args),
    )

    runner = DecisionCurveAnalyzer(config)
    summary = runner.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
