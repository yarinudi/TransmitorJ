"""
Generate Table 1 (descriptive statistics) for two manuscripts:
  1. npj Digital Medicine  — overall cohort, 7-outcome event panel
  2. npj Parkinson's Disease — stratified by incident PD status

Requires: tableone, pandas, pathlib
Install:  pip install tableone

CLI usage:
  python generate_table1.py --data <path_to_cohort_csv_or_parquet>
  python generate_table1.py --data cohort.csv --sway sway_features.csv

Programmatic usage (notebook / script):
  from analysis.table1.generate_table1 import generate_table1

  # Returns dict of {"dm": TableOne, "pd": TableOne}
  results = generate_table1(df, paper="both")

  # Or generate one at a time, optionally saving to disk
  from analysis.table1.generate_table1 import table1_npj_dm, table1_npj_pd
  t1 = table1_npj_dm(df)                          # no files saved
  t1 = table1_npj_pd(df, out_dir=Path("output"))  # saves csv/tex
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tableone import TableOne

# ── Output directories ────────────────────────────────────────────────────────
OUT_DIR_DM = Path(__file__).resolve().parents[1] / "submissions" / "npj-digital-medicine"
OUT_DIR_PD = Path(__file__).resolve().parents[1] / "submissions" / "npj-parkinsons-disease"

# ── Column name maps (from WHS SAS data dictionary) ──────────────────────────
# Adjust these if your working CSV uses different names.

DEMO_CONTINUOUS = [
    "ageaccel",     # Age at accelerometer study
    "bmi",          # BMI at accelerometer study
    "weight",       # Weight (lbs)
    "heightbs",     # Height (imputed)
]

DEMO_CATEGORICAL = [
    "RACE",         # 1-6 + None
    "EDUC",         # 1-6 + None
    "smoke",        # 1=never, 2=past, 3=current
    "alcuse",       # 1=rarely, 2=monthly, 3=weekly, 4=daily
]

SF12_ITEMS = [
    "SF12_modact",        # Moderate activities limited
    "SF12_climbsev",      # Climbing several flights of stairs
    "SF12_climbone",      # Climbing one flight of stairs
    "SF12_lifting",       # Lifting or carrying groceries
    "SF12_bending",       # Bending, kneeling or stooping
    "SF12_walkmile",      # Walking more than one mile
    "SF12_walkblocks",    # Walking several blocks
    "SF12_walkblock",     # Walking one block
    "SF12_bath",          # Bathing or dressing yourself
    "SF12_vigact",        # Vigorous activities
    "SF12_cut_time",      # Cut down time on work/activities
    "SF12_phys_accless",  # Accomplished less (physical)
    "SF12_limitedwork",   # Limited in kind of work
    "SF12_diff_work",     # Difficulty performing work
    "SF12_emot_accless",  # Accomplished less (emotional)
    "SF12_not_careful",   # Did not do work as carefully
    "SF12_pain",          # Pain interference
    "SF12_energy",        # Lot of energy
    "SF12_felt_calm",     # Felt calm and peaceful
    "SF12_felt_down",     # Felt downhearted and blue
    "SF12_soc_act",       # Social activity interference
]

HEALTH_HISTORY = [
    "genhealth",    # General health 1-4
    "histhtn",      # History of hypertension
    "depression",   # Depression 1/0
]

PHYSICAL_ACTIVITY = [
    "tmethrst",     # MET-hr/wk from physical activities
    "walkpace",     # Usual walking pace
    "walking",      # Time spent walking per week
    "jogging",      # Time spent jogging per week
    "running",      # Time spent running per week
    "biking",       # Time spent biking per week
    "aerobic",      # Time spent aerobic per week
    "swim",         # Time spent swimming per week
    "stairclimb",   # Stairs climbed per week
]

SITTING = [
    "sit_work",     # Sitting at work/away/driving
    "sit_hometv",   # Sitting at home watching TV
    "sithome",      # Sitting at home
]

# ── Outcome definitions (7 outcomes matching both manuscripts) ────────────────
# Raw columns: event indicator → time-to-event column
OUTCOME_RAW = {
    "death":           ("death",           "randyears"),
    "jointreplace":    ("jointreplace",    "jointyrs"),
    "Osteoarthritis":  ("Osteoarthritis",  "Osteoarthritisyrs"),
    "parkinson":       ("parkinson",       "parkyrs"),
    "stkconf":         ("stkconf",         "strokeyears"),
    "tiaunr":          ("tiaunr",          "tiayrs"),
    "first_fall_post": ("first_fall_post_label", "first_fall_post"),
}

# Derived incident column names (created by derive_incident_events)
INCIDENT_EVENT = {
    "death":           "incident_death",
    "jointreplace":    "incident_jointreplace",
    "Osteoarthritis":  "incident_Osteoarthritis",
    "parkinson":       "incident_parkinson",
    "stkconf":         "incident_stkconf",
    "tiaunr":          "incident_tiaunr",
    "first_fall_post": "incident_first_fall_post",
}

INCIDENT_TIME = {
    "death":           "time_death",
    "jointreplace":    "time_jointreplace",
    "Osteoarthritis":  "time_Osteoarthritis",
    "parkinson":       "time_parkinson",
    "stkconf":         "time_stkconf",
    "tiaunr":          "time_tiaunr",
    "first_fall_post": "time_first_fall_post",
}

OUTCOME_LABELS = {
    "death":           "Mortality",
    "jointreplace":    "Joint replacement",
    "Osteoarthritis":  "Osteoarthritis",
    "parkinson":       "Parkinson's disease",
    "stkconf":         "Stroke (confirmed)",
    "tiaunr":          "TIA (unrefuted)",
    "first_fall_post": "Hip fracture",
}

# Sway-derived features to include in npj PD Table 1
# (top GAM predictors from the Q+S model — adjust if your CSV names differ)
SWAY_TOP_FEATURES = [
    # "Area_stds",
    # "RMS_stds",
    # "RangeA_stds",
    # "CentroidFreq_p5",
    # "PWR_stds",
    # "MedianDist_p99",
]

# Step-rate features (classified as Q in the manuscripts)
STEP_RATE = [
    # "mean_spm",       # Mean steps-per-minute
    # "mean_std_spm",   # SD of steps-per-minute
]

# ── Readable labels ───────────────────────────────────────────────────────────
LABELS = {
    "ageaccel":           "Age, years",
    "bmi":                "BMI, kg/m²",
    "weight":             "Weight, lbs",
    "heightbs":           "Height, inches",
    "RACE":               "Race/ethnicity",
    "EDUC":               "Education level",
    "smoke":              "Smoking status",
    "alcuse":             "Alcohol intake",
    "genhealth":          "General health",
    "histhtn":            "History of hypertension",
    "depression":         "Depression",
    "tmethrst":           "Physical activity, MET-hr/wk",
    "walkpace":           "Usual walking pace",
    "walking":            "Walking, time/wk",
    "sit_work":           "Sitting at work/driving",
    "sit_hometv":         "Sitting at home (TV)",
    "sithome":            "Sitting at home (other)",
    "SF12_modact":        "SF-12: Moderate activities",
    "SF12_climbsev":      "SF-12: Climb several flights",
    "SF12_climbone":      "SF-12: Climb one flight",
    "SF12_lifting":       "SF-12: Lifting/carrying",
    "SF12_bending":       "SF-12: Bending/kneeling",
    "SF12_walkmile":      "SF-12: Walk >1 mile",
    "SF12_walkblocks":    "SF-12: Walk several blocks",
    "SF12_walkblock":     "SF-12: Walk one block",
    "SF12_bath":          "SF-12: Bathing/dressing",
    "SF12_vigact":        "SF-12: Vigorous activities",
    "SF12_cut_time":      "SF-12: Cut time on work",
    "SF12_phys_accless":  "SF-12: Accomplished less (phys.)",
    "SF12_limitedwork":   "SF-12: Limited kind of work",
    "SF12_diff_work":     "SF-12: Difficulty at work",
    "SF12_emot_accless":  "SF-12: Accomplished less (emot.)",
    "SF12_not_careful":   "SF-12: Not careful at work",
    "SF12_pain":          "SF-12: Pain interference",
    "SF12_energy":        "SF-12: Energy level",
    "SF12_felt_calm":     "SF-12: Felt calm/peaceful",
    "SF12_felt_down":     "SF-12: Felt downhearted",
    "SF12_soc_act":       "SF-12: Social activity interference",
    "mean_spm":           "Steps/min (mean)",
    "mean_std_spm":       "Steps/min (SD)",
    "Area_stds":          "Sway area (SD across bouts)",
    "RMS_stds":           "RMS acceleration (SD across bouts)",
    "RangeA_stds":        "Primary-axis range (SD across bouts)",
    "CentroidFreq_p5":    "Centroid frequency (5th pctl.)",
    "PWR_stds":           "Spectral power (SD across bouts)",
    "MedianDist_p99":     "Median distance (99th pctl.)",
    "randyears":          "Follow-up, years",
    "follow_up_years":    "Follow-up, years",
}

RACE_MAP = {1: "White", 2: "Black", 3: "Hispanic", 4: "Asian/Pacific Islander",
            5: "American Indian", 6: "Other"}
EDUC_MAP = {1: "Less than high school", 2: "High school graduate",
            3: "Some college", 4: "College graduate",
            5: "Master's degree", 6: "Doctoral/professional degree"}
SMOKE_MAP = {1: "Never", 2: "Past", 3: "Current"}
ALCUSE_MAP = {1: "Rarely/never", 2: "Monthly", 3: "Weekly", 4: "Daily"}
GENHEALTH_MAP = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair/poor"}


def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".sas7bdat"}:
        return pd.read_sas(path, format="sas7bdat")
    raise ValueError(f"Unsupported format: {suffix}")


def derive_incident_events(df: pd.DataFrame) -> pd.DataFrame:
    """Derive incident event indicators and proper follow-up times.

    Mirrors the ``load_target_events`` / ``format_events`` logic from the
    analysis pipeline:

    1. Derive ``first_fall_post`` (time to first osteoporotic fracture after
       accel wear) and ``first_fall_post_label`` from the ``fractpyr*`` columns.
    2. For each of the 7 outcomes, create:
       - ``incident_<outcome>`` : 1 if the raw event occurred *after* accel
         deployment (time > 0), else 0.
       - ``time_<outcome>`` : time-to-event for incident cases; censoring
         time (``randyears``) for non-events / prevalent cases.
    3. ``follow_up_years`` : overall follow-up (``randyears``), safe to use
       in both Table 1 variants.
    """
    df = df.copy()

    # ── 1. Derive first_fall_post from fractpyr columns ──────────────────
    cols_fractpyr = sorted(c for c in df.columns if c.startswith("fractpyr"))
    if cols_fractpyr:
        positive_fract = df[cols_fractpyr].where(df[cols_fractpyr] > 0)
        df["first_fall_post"] = positive_fract.min(axis=1)
        df["first_fall_post_label"] = (~df["first_fall_post"].isna()).astype(int)
        df["first_fall_post"] = df["first_fall_post"].fillna(df["randyears"])
    else:
        df["first_fall_post"] = df.get("randyears", np.nan)
        df["first_fall_post_label"] = 0

    # ── 2. Create incident indicators and proper time columns ────────────
    censor_time = df["randyears"]

    for key, (event_col, time_col) in OUTCOME_RAW.items():
        if event_col not in df.columns or time_col not in df.columns:
            continue

        raw_label = df[event_col]
        raw_time = df[time_col]

        incident = ((raw_label > 0) & (raw_time > 0)).astype(int)
        time = np.where(incident == 1, raw_time, censor_time)

        df[INCIDENT_EVENT[key]] = incident
        df[INCIDENT_TIME[key]] = time

    # ── 3. Overall follow-up ─────────────────────────────────────────────
    df["follow_up_years"] = censor_time

    return df


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map coded values to readable category labels."""
    col_maps = {
        "RACE": RACE_MAP,
        "EDUC": EDUC_MAP,
        "smoke": SMOKE_MAP,
        "alcuse": ALCUSE_MAP,
        "genhealth": GENHEALTH_MAP,
    }
    for col, mapping in col_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna("Unknown")
    return df


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return only columns that exist in df."""
    return [c for c in cols if c in df.columns]


def build_event_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary of incident outcome event counts."""
    rows = []
    for key, label in OUTCOME_LABELS.items():
        inc_col = INCIDENT_EVENT[key]
        time_col = INCIDENT_TIME[key]
        if inc_col not in df.columns:
            continue
        n_events = int(df[inc_col].sum())
        pct = n_events / len(df) * 100
        median_fu = df[time_col].median() if time_col in df.columns else None
        rows.append({
            "Outcome": label,
            "Events, n (%)": f"{n_events:,} ({pct:.1f}%)",
            "Median follow-up, years": f"{median_fu:.1f}" if median_fu is not None else "",
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 1 — npj Digital Medicine (overall cohort, no stratification)
# ═════════════════════════════════════════════════════════════════════════════
def table1_npj_dm(
    df: pd.DataFrame,
    out_dir: Path | str | None = None,
    save: bool = True,
) -> tuple[TableOne, pd.DataFrame]:
    """Generate Table 1 for npj Digital Medicine (overall cohort).

    Parameters
    ----------
    df : pd.DataFrame
        Cohort dataframe (already label-mapped or raw — both work).
    out_dir : Path or str, optional
        Directory for output files.  Defaults to the npj-digital-medicine
        submission folder when *save* is True and *out_dir* is None.
    save : bool
        If True, write csv / tex files.  Set False when calling
        from a notebook and you only want the returned objects.

    Returns
    -------
    (TableOne, pd.DataFrame)
        The descriptive-statistics table and the outcome event panel.
    """
    columns = available(df, (
        DEMO_CONTINUOUS + DEMO_CATEGORICAL + HEALTH_HISTORY
        + ["SF12_modact", "SF12_pain", "SF12_energy", "SF12_felt_down",
           "SF12_soc_act", "SF12_cut_time"]
        + STEP_RATE
        + ["tmethrst", "walkpace", "follow_up_years"]
    ))

    categorical = available(df, DEMO_CATEGORICAL + HEALTH_HISTORY + [
        "SF12_modact", "SF12_pain", "SF12_energy", "SF12_felt_down",
        "SF12_soc_act", "SF12_cut_time",
    ])

    nonnormal = available(df, ["tmethrst", "follow_up_years"])

    rename = {k: v for k, v in LABELS.items() if k in columns}

    t1 = TableOne(
        df,
        columns=columns,
        categorical=categorical,
        nonnormal=nonnormal,
        rename=rename,
        missing=True,
        overall=True,
        label_suffix=False,
    )

    event_panel = build_event_panel(df)

    if save:
        dest = Path(out_dir) if out_dir is not None else OUT_DIR_DM
        dest.mkdir(parents=True, exist_ok=True)
        t1.to_csv(dest / "table1_cohort.csv")
        t1.to_latex(dest / "table1_cohort.tex")
        event_panel.to_csv(dest / "table1_events.csv", index=False)
        event_panel.to_latex(dest / "table1_events.tex", index=False)
        print(f"Saved to {dest}")

    print(f"\n{'='*72}")
    print("npj Digital Medicine — Table 1 (overall cohort)")
    print(f"{'='*72}")
    print(t1.tabulate(tablefmt="grid"))
    print("\nOutcome event counts:")
    print(event_panel.to_string(index=False))

    return t1, event_panel


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 1 — npj Parkinson's Disease (stratified by incident PD)
# ═════════════════════════════════════════════════════════════════════════════
def table1_npj_pd(
    df: pd.DataFrame,
    out_dir: Path | str | None = None,
    save: bool = True,
) -> TableOne:
    """Generate Table 1 for npj Parkinson's Disease (stratified by PD).

    Parameters
    ----------
    df : pd.DataFrame
        Cohort dataframe.  Must contain an ``incident_parkinson`` column
        (created by ``derive_incident_events``).
    out_dir : Path or str, optional
        Directory for output files.  Defaults to the npj-parkinsons-disease
        submission folder when *save* is True and *out_dir* is None.
    save : bool
        If True, write csv / tex files.  Set False when calling
        from a notebook and you only want the returned object.

    Returns
    -------
    TableOne
    """
    strat_col = "incident_parkinson"
    if strat_col not in df.columns:
        raise KeyError(
            f"Column '{strat_col}' not found — run derive_incident_events() first. "
            f"Available: {list(df.columns[:20])}..."
        )

    df = df.copy()
    df["PD status"] = df[strat_col].map({0: "No PD", 1: "Incident PD"})

    sf12_pd = [
        "SF12_modact", "SF12_cut_time", "SF12_felt_down", "SF12_energy",
        "SF12_soc_act", "SF12_phys_accless", "SF12_emot_accless",
        "SF12_pain", "SF12_vigact",
    ]

    columns = available(df, (
        DEMO_CONTINUOUS + DEMO_CATEGORICAL + HEALTH_HISTORY
        + sf12_pd + STEP_RATE + SWAY_TOP_FEATURES
        + ["tmethrst", "walkpace", "follow_up_years"]
    ))

    categorical = available(df, DEMO_CATEGORICAL + HEALTH_HISTORY + sf12_pd)

    nonnormal = available(df, ["tmethrst", "follow_up_years"] + SWAY_TOP_FEATURES)

    rename = {k: v for k, v in LABELS.items() if k in columns}

    t1 = TableOne(
        df,
        columns=columns,
        categorical=categorical,
        nonnormal=nonnormal,
        groupby="PD status",
        rename=rename,
        pval=True,
        smd=True,
        missing=True,
        overall=True,
        label_suffix=False,
    )

    if save:
        dest = Path(out_dir) if out_dir is not None else OUT_DIR_PD
        dest.mkdir(parents=True, exist_ok=True)
        t1.to_csv(dest / "table1_pd_stratified.csv")
        t1.to_latex(dest / "table1_pd_stratified.tex")
        print(f"Saved to {dest}")

    print(f"\n{'='*72}")
    print("npj Parkinson's Disease — Table 1 (stratified by PD status)")
    print(f"{'='*72}")
    print(t1.tabulate(tablefmt="grid"))

    return t1


# ═════════════════════════════════════════════════════════════════════════════
# Convenience wrapper for programmatic use
# ═════════════════════════════════════════════════════════════════════════════
def generate_table1(
    df: pd.DataFrame,
    paper: Literal["dm", "pd", "both"] = "both",
    *,
    out_dir: Path | str | None = None,
    save: bool = False,
    remap_labels: bool = True,
) -> dict[str, TableOne | tuple[TableOne, pd.DataFrame]]:
    """One-call entry point for generating Table 1 from a loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The cohort dataframe (WHS columns, optionally with sway features
        already merged in).  If incident event columns are not yet present,
        ``derive_incident_events`` is called automatically.
    paper : {"dm", "pd", "both"}
        Which table(s) to generate.
    out_dir : Path or str, optional
        Override the output directory for saved files.
    save : bool
        If True, write csv / tex outputs.
    remap_labels : bool
        If True, map coded integers (RACE, EDUC, smoke, alcuse, genhealth)
        to human-readable labels before building the table.

    Returns
    -------
    dict
        Keys are ``"dm"`` and/or ``"pd"``.
        - ``"dm"`` value is ``(TableOne, event_panel_DataFrame)``
        - ``"pd"`` value is ``TableOne``

    Examples
    --------
    >>> import pandas as pd
    >>> from analysis.table1.generate_table1 import generate_table1
    >>> df = pd.read_csv("my_cohort.csv")
    >>> results = generate_table1(df, paper="both")
    >>> results["dm"][0]   # TableOne for npj Digital Medicine
    >>> results["pd"]      # TableOne for npj Parkinson's Disease

    >>> # In a notebook — just print, don't save files
    >>> t1_pd = generate_table1(df, paper="pd", save=False)["pd"]
    >>> print(t1_pd.tabulate(tablefmt="fancy_grid"))
    """
    df = df.copy()

    if "incident_parkinson" not in df.columns:
        df = derive_incident_events(df)

    if remap_labels:
        df = apply_labels(df)

    results: dict[str, TableOne | tuple[TableOne, pd.DataFrame]] = {}

    if paper in ("dm", "both"):
        results["dm"] = table1_npj_dm(df, out_dir=out_dir, save=save)

    if paper in ("pd", "both"):
        results["pd"] = table1_npj_pd(df, out_dir=out_dir, save=save)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate Table 1 for npj Digital Medicine and npj Parkinson's Disease")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to main cohort file (CSV, Parquet, or SAS7BDAT). "
                             "Must contain the WHS demographic + questionnaire columns.")
    parser.add_argument("--sway", type=Path, default=None,
                        help="Path to sway feature file (CSV/Parquet). "
                             "Merged on 'newid'. Only needed for npj PD table.")
    parser.add_argument("--paper", choices=["dm", "pd", "both"], default="both",
                        help="Which table to generate: dm, pd, or both (default: both)")
    args = parser.parse_args()

    print(f"Loading cohort data from {args.data} ...")
    df = load_data(args.data)
    print(f"  → {len(df):,} rows, {len(df.columns)} columns")

    if args.sway is not None:
        print(f"Loading sway features from {args.sway} ...")
        sway = load_data(args.sway)
        merge_key = "newid" if "newid" in sway.columns else sway.columns[0]
        df = df.merge(sway, on=merge_key, how="left")
        print(f"  → After merge: {len(df):,} rows, {len(df.columns)} columns")

    print("Deriving incident event indicators ...")
    df = derive_incident_events(df)
    for key, label in OUTCOME_LABELS.items():
        inc_col = INCIDENT_EVENT[key]
        if inc_col in df.columns:
            print(f"  {label}: {int(df[inc_col].sum()):,} incident events")

    generate_table1(df, paper=args.paper, save=True)


if __name__ == "__main__":
    main()
