# Decision Curve Analysis (DCA) Runner
The implementation is based on the following resources:
- <a href="https://mskcc-epi-bio.github.io/decisioncurveanalysis/">Decision Curve Analysis</a>


This folder provides a reusable Python runner for decision curve analysis with **survival mode as default**:

- Script: `analysis/dca/run_dca.py`
- Main object: `DecisionCurveAnalyzer`
- Entry point: `main()`

The script is designed to be reused across projects/papers by passing column names and file paths via CLI flags.

## What It Does

- Computes net benefit curves for one or more prediction models
- Compares each model against:
  - treat-all strategy
  - treat-none strategy
- In survival mode, uses IPCW-style weighting at a chosen horizon to account for censoring
- Computes calibration diagnostics for each model:
  - calibration intercept (ideal near `0`)
  - calibration slope (ideal near `1`)
  - Brier score (lower is better)
  - calibration curve + calibration plot
- Optionally computes bootstrap confidence intervals for model net benefit
- Writes machine-readable outputs (`.csv`, `.json`) and a publication-ready plot (`.png`)

## Input Requirements

Input file can be CSV or Parquet.

### Default (`--mode survival`)

Must include:

- `time` column (time to event/censoring)
- `event` column (event indicator)
- one or more predicted risk columns in `[0, 1]`
- a chosen `--horizon` in the same time units as `time`

Example columns:

- `time_years`
- `event_pd`
- `risk_q_5y`
- `risk_qs_5y`

### Optional (`--mode binary`)

Must include:

- one binary outcome column (e.g., event by fixed horizon)
- one or more model risk columns with probabilities in `[0, 1]`

Example columns:

- `y_true_horizon`
- `risk_q`
- `risk_qs`
- `risk_qsa`

## CLI Arguments

Required:

- `--data-path` path to CSV/Parquet
- `--model` model risk spec (repeatable)
  - format: `"Label:column"` or just `"column"`

Common optional:

- `--mode` `survival` (default) or `binary`
- `--time-col`, `--event-col`, `--horizon` (required in survival mode)
- `--outcome-col` (required in binary mode)
- `--event-value` value considered event in outcome column (default: `1`)
- `--threshold-min`, `--threshold-max`, `--threshold-step`
- `--thresholds` explicit comma-separated thresholds
- `--bootstrap` number of bootstrap iterations for CIs
- `--outdir` output directory
- `--clip-risk` clip risk values to `[0,1]`
- `--no-plot` disable PNG output
- `--figure-name`, `--figure-title`
- `--no-calibration` disable calibration diagnostics
- `--calibration-bins` number of bins for calibration curve (default: `10`)
- `--calibration-strategy` binning strategy: `equal_width` or `quantile`
- `--calibration-plot-name`, `--calibration-plot-title`
- `--nri` enable Net Reclassification Improvement (NRI) calculation
- `--nri-reference` model label used as NRI reference (default: first `--model`)
- `--nri-cutoffs` comma-separated cutoffs for NRI (`0` = continuous NRI; default: `0`)

For NRI support, install:

```bash
pip install reclassification

From github:
https://github.com/morleytj/reclassification
```

## Why Calibration Is Necessary Before/Alongside DCA

DCA uses **absolute predicted risk** and compares decisions across threshold probabilities (for example, 5%, 10%, 20%).  
If risk values are miscalibrated (too high or too low), net benefit can be over- or under-estimated even when discrimination is acceptable.

Use calibration diagnostics to verify that:

- predicted probabilities align with observed event rates
- threshold-based decisions are interpretable clinically
- DCA conclusions are based on trustworthy risk scales

## Command Presets

### 1) Quick Run (survival mode, default)

```bash
python analysis/dca/run_dca.py \
  --data-path data/dca_input.csv \
  --time-col time_years \
  --event-col event_pd \
  --horizon 5 \
  --event-value 1 \
  --model "Q:risk_q_5y" \
  --model "Q+S:risk_qs_5y" \
  --threshold-min 0.02 \
  --threshold-max 0.20 \
  --threshold-step 0.01 \
  --outdir analysis/dca/output_quick
```

### 2) Publication Run (survival mode + bootstrap CIs)

```bash
python analysis/dca/run_dca.py \
  --data-path data/dca_input.csv \
  --time-col time_years \
  --event-col event_pd \
  --horizon 5 \
  --event-value 1 \
  --model "Q:risk_q_5y" \
  --model "Q+S:risk_qs_5y" \
  --model "Q+S+A:risk_qsa_5y" \
  --threshold-min 0.02 \
  --threshold-max 0.20 \
  --threshold-step 0.01 \
  --bootstrap 1000 \
  --seed 42 \
  --calibration-bins 10 \
  --calibration-strategy equal_width \
  --outdir analysis/dca/output_pub \
  --figure-name dca_curves_pub.png \
  --figure-title "Decision Curve Analysis (5-year PD risk)" \
  --calibration-plot-name calibration_pub.png \
  --calibration-plot-title "Calibration (5-year PD risk)"
```

### 3) Binary Mode Fallback (no censoring-aware weighting)

```bash
python analysis/dca/run_dca.py \
  --mode binary \
  --data-path data/dca_input.csv \
  --outcome-col y_true_horizon \
  --event-value 1 \
  --model "risk_q" \
  --model "risk_qs" \
  --thresholds 0.02,0.05,0.10,0.15,0.20 \
  --no-plot \
  --outdir analysis/dca/output_batch
```

### 4) Multi-Horizon Runs (3y, 5y, 10y)

Run the same analysis separately per horizon (recommended for survival models).

```bash
# 3-year
python analysis/dca/run_dca.py \
  --data-path data/dca_input.csv \
  --time-col time_years \
  --event-col event_pd \
  --horizon 3 \
  --event-value 1 \
  --model "Q:risk_q_3y" \
  --model "Q+S:risk_qs_3y" \
  --model "Q+S+A:risk_qsa_3y" \
  --threshold-min 0.02 \
  --threshold-max 0.20 \
  --threshold-step 0.01 \
  --outdir analysis/dca/output_3y \
  --figure-title "Decision Curve Analysis (3-year PD risk)" \
  --calibration-plot-title "Calibration (3-year PD risk)"

# 5-year
python analysis/dca/run_dca.py \
  --data-path data/dca_input.csv \
  --time-col time_years \
  --event-col event_pd \
  --horizon 5 \
  --event-value 1 \
  --model "Q:risk_q_5y" \
  --model "Q+S:risk_qs_5y" \
  --model "Q+S+A:risk_qsa_5y" \
  --threshold-min 0.02 \
  --threshold-max 0.20 \
  --threshold-step 0.01 \
  --outdir analysis/dca/output_5y \
  --figure-title "Decision Curve Analysis (5-year PD risk)" \
  --calibration-plot-title "Calibration (5-year PD risk)"

# 10-year
python analysis/dca/run_dca.py \
  --data-path data/dca_input.csv \
  --time-col time_years \
  --event-col event_pd \
  --horizon 10 \
  --event-value 1 \
  --model "Q:risk_q_10y" \
  --model "Q+S:risk_qs_10y" \
  --model "Q+S+A:risk_qsa_10y" \
  --threshold-min 0.02 \
  --threshold-max 0.20 \
  --threshold-step 0.01 \
  --outdir analysis/dca/output_10y \
  --figure-title "Decision Curve Analysis (10-year PD risk)" \
  --calibration-plot-title "Calibration (10-year PD risk)"
```

### 5) Add NRI (relative to base model)

```bash
python analysis/dca/run_dca.py \
  --data-path data/dca_input.csv \
  --time-col time_years \
  --event-col event_pd \
  --horizon 3 \
  --event-value 1 \
  --model "Q:risk_q_3y" \
  --model "Q+S:risk_qs_3y" \
  --model "Q+S+A:risk_qsa_3y" \
  --nri \
  --nri-reference "Q" \
  --nri-cutoffs 0,0.02 \
  --outdir analysis/dca/output_3y_nri \
  --no-plot
```

## Outputs

The output directory contains:

- `dca_curves_long.csv`  
  Long-format net benefit values by threshold, model, and strategy.
- `dca_summary.json`  
  Includes:
  - mode metadata (survival or binary)
  - `n_rows_used`
  - weighted event rate
  - per-model max net benefit
  - threshold intervals where model net benefit is above both treat-all and treat-none
- `dca_curves.png` (unless `--no-plot`)
- `dca_ci_<model>.csv` (if `--bootstrap > 0`)
- `calibration_metrics.json` (unless `--no-calibration`)
- `calibration_curve_<model>.csv` (unless `--no-calibration`)
- `calibration_plot.png` (unless `--no-calibration` or `--no-plot`)
- NRI results in `dca_summary.json` under top-level `nri` and per-model `nri_vs_reference` (if `--nri`)

## Calibration Data Requirements

For the calibration metrics and plot implemented here, **no extra data is required** beyond:

- binary target at horizon (internally derived in survival mode via IPCW, or direct in binary mode)
- predicted risks in `[0, 1]`

Optional extra data (not required in this implementation, but useful in advanced settings):

- sample weights
- subgroup columns for subgroup calibration reporting
- time-to-event and censoring columns for survival-aware calibration

## Notes

- DCA assumes clinically meaningful threshold probabilities; choose thresholds intentionally.
- DCA evaluates *clinical utility* of risk-based decisions, not only discrimination.
- Survival mode is the default and should be preferred when censoring is present.
- `matplotlib` is only required when plotting is enabled; use `--no-plot` for metrics-only runs.
- NRI currently uses `reclassification.calculate_nri` and is unweighted (sample/IPCW weights are not applied).

---

# Performance Comparison Table

Companion script for producing a consolidated model-comparison table alongside the DCA pipeline.

- Script: `analysis/dca/run_perf_table.py`
- Entry point: `main()`

Uses the same input data and CLI conventions as `run_dca.py`.

## What It Does

Computes four families of metrics for each prediction model and writes them as a single comparison table:

| Metric | Description |
|---|---|
| **C-td (Uno)** | IPCW-weighted concordance at the prediction horizon; all subjects contribute |
| **Calibration** | Intercept (ideal 0), slope (ideal 1), and Brier score via logistic recalibration |
| **PPV / NPV** | IPCW-weighted positive and negative predictive values at user-specified thresholds |
| **NRI** | Net Reclassification Improvement versus a reference model (optional) |

Calibration reuses `CalibrationAnalyzer` from `run_dca.py`. Uno's C-td and PPV/NPV are implemented locally with the same KM censoring estimator.

## CLI Arguments

Required (same as `run_dca.py`):

- `--data-path` path to CSV/Parquet
- `--model` model risk spec (repeatable): `"Label:column"` or `"column"`

Survival mode (default):

- `--time-col`, `--event-col`, `--horizon`
- `--event-value` (default: `1`)

Binary mode:

- `--mode binary`
- `--outcome-col`

Performance-table specific:

- `--ppv-npv-thresholds` comma-separated thresholds for PPV/NPV (default: `0.05,0.10,0.15`)
- `--calibration-bins` number of bins (default: `10`)
- `--calibration-strategy` `equal_width` or `quantile`
- `--nri` enable NRI
- `--nri-reference` reference model label (default: first `--model`)
- `--nri-cutoffs` comma-separated cutoffs (default: `0` = continuous)
- `--bootstrap` number of bootstrap iterations for 95% CIs (default: `0` = off)
- `--seed` random seed (default: `42`)
- `--latex` emit a LaTeX table alongside CSV
- `--latex-caption`, `--latex-label` customise the LaTeX table
- `--outdir` output directory (default: `analysis/dca/output_perf`)
- `--clip-risk` clip risk values to `[0, 1]`

## Command Presets

### 1) Quick comparison (survival, 3-year horizon)

```bash
python analysis/dca/run_perf_table.py \
  --data-path data/dca_input.csv \
  --time-col time_years \
  --event-col event_pd \
  --horizon 3 \
  --event-value 1 \
  --model "Q:risk_q_3y" \
  --model "Q+S:risk_qs_3y" \
  --model "Q+S+A:risk_qsa_3y" \
  --ppv-npv-thresholds 0.05,0.10,0.15 \
  --outdir analysis/dca/output_perf_3y
```

### 2) Publication table (with NRI, bootstrap CIs, LaTeX)

```bash
python analysis/dca/run_perf_table.py \
  --data-path data/dca_input.csv \
  --time-col time_years \
  --event-col event_pd \
  --horizon 5 \
  --event-value 1 \
  --model "Q:risk_q_5y" \
  --model "Q+S:risk_qs_5y" \
  --model "Q+S+A:risk_qsa_5y" \
  --ppv-npv-thresholds 0.05,0.10,0.15 \
  --nri \
  --nri-reference "Q" \
  --nri-cutoffs 0 \
  --bootstrap 1000 \
  --seed 42 \
  --latex \
  --latex-caption "Model discrimination, calibration, and reclassification (5-year)" \
  --latex-label "tab:perf_5y" \
  --outdir analysis/dca/output_perf_5y
```

### 3) Binary mode

```bash
python analysis/dca/run_perf_table.py \
  --mode binary \
  --data-path data/dca_input.csv \
  --outcome-col y_true_horizon \
  --event-value 1 \
  --model "risk_q" \
  --model "risk_qs" \
  --ppv-npv-thresholds 0.05,0.10,0.15,0.20 \
  --outdir analysis/dca/output_perf_binary
```

## Outputs

- `performance_table.csv` — wide format, one row per model
- `classification_by_threshold.csv` — long format with PPV, NPV, sensitivity, specificity per (model, threshold)
- `performance_summary.json` — full results including per-model metrics and optional bootstrap CIs
- `performance_table.tex` — LaTeX `tabular` (only when `--latex` is set)

## Relationship to `run_dca.py`

| Concern | `run_dca.py` | `run_perf_table.py` |
|---|---|---|
| Clinical utility curves | Yes (DCA net benefit) | No |
| Calibration diagnostics | Yes (metrics + plot) | Yes (metrics only, reuses `CalibrationAnalyzer`) |
| Discrimination (C-td) | No | Yes (Uno's C-statistic) |
| PPV / NPV | No | Yes (IPCW-weighted, at multiple thresholds) |
| NRI | Yes (in DCA summary) | Yes (in performance table) |
| Output format | Curves + JSON | Flat comparison table (CSV / LaTeX) |

Both scripts accept the same input data and `--model` spec format, so they can be run back-to-back on the same dataset.
