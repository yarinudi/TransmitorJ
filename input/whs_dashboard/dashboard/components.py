"""Reusable Streamlit UI blocks: sidebar, KPI cards, cohort table, config editor."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import streamlit as st

from whs_preprocessor import PreprocessConfig

from . import data_loader as dl
from . import plots
from .pipeline_runner import config_field_specs, run_pipeline


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------

def kpi_row(items: list[tuple[str, str]]) -> None:
    """Render a row of label/value chips. ``items`` is [(label, value_str), ...]."""
    if not items:
        return
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


# ---------------------------------------------------------------------------
# Config editor
# ---------------------------------------------------------------------------

def config_editor(default_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render a collapsible editor for every PreprocessConfig field.

    Returns a plain dict suitable for passing to ``run_pipeline``. Editing any
    field re-runs the script and invalidates the pipeline cache (because the
    cache key includes ``config_dict``).
    """
    specs = config_field_specs()
    if default_cfg is None:
        default_cfg = {name: default for name, _, default in specs}

    edited: dict[str, Any] = {}
    with st.sidebar.expander("Preprocessing config", expanded=False):
        for name, type_, _ in specs:
            current = default_cfg.get(name)
            label = name.replace("_", " ")
            if type_ is bool:
                edited[name] = bool(st.checkbox(label, value=bool(current), key=f"cfg_{name}"))
            elif type_ is int:
                edited[name] = int(st.number_input(label, value=int(current),
                                                   step=1, key=f"cfg_{name}"))
            elif type_ is float:
                edited[name] = float(st.number_input(label, value=float(current),
                                                     step=0.001, format="%.4f",
                                                     key=f"cfg_{name}"))
            else:
                edited[name] = st.text_input(label, value=str(current), key=f"cfg_{name}")

        if st.button("Reset to defaults", key="cfg_reset"):
            for name, _, default in specs:
                st.session_state[f"cfg_{name}"] = default
            st.rerun()

    return edited


# ---------------------------------------------------------------------------
# Sidebar — patient + recording start + config
# ---------------------------------------------------------------------------

def sidebar(default_data_dir: str = "./example_data/") -> dict[str, Any]:
    """Render the sidebar. Returns a dict with:

      * ``patient``   — the loaded ``(data, start_datetime, subject_id)`` tuple
      * ``config``    — edited PreprocessConfig as a plain dict
      * ``patients``  — list of discovered ``PatientRecord``s (plus synthetic)
    """
    st.sidebar.header("Patient")
    data_dir = st.sidebar.text_input("Raw data directory", value=default_data_dir,
                                     help="Directory scanned for .npy / .csv / .csv.gz files.")
    records = dl.discover_patients(data_dir)

    options: list[tuple[str, Any]] = []
    options.extend((r.subject_id, r) for r in records)
    options.append(("SYNTH001 (synthetic demo)", "__synth__"))

    labels = [o[0] for o in options]
    idx = st.sidebar.selectbox("Subject", range(len(labels)), format_func=lambda i: labels[i])
    choice = options[idx][1]

    if choice == "__synth__":
        data, default_start, subject_id = dl.get_synthetic_patient()
    else:
        data, default_start, subject_id = dl.load_patient(str(choice.path), choice.mtime)

    st.sidebar.caption(
        f"{subject_id} — {data.shape[0]:,} samples ({data.shape[0] / (30 * 3600):.2f} h @ 30 Hz)"
    )

    st.sidebar.markdown("**Recording start**")
    start_date = st.sidebar.date_input("Date", value=default_start.date(),
                                       key="rec_start_date")
    start_time = st.sidebar.time_input("Time", value=default_start.time(),
                                       key="rec_start_time")
    start_datetime = datetime.combine(start_date, start_time)

    config = config_editor()

    return {
        "patient": (data, start_datetime, subject_id),
        "config": config,
        "patients": records,
    }


# ---------------------------------------------------------------------------
# Cohort table
# ---------------------------------------------------------------------------

def cohort_table(records: list[dl.PatientRecord], config: dict[str, Any],
                 synthetic_available: bool = True) -> None:
    """Render the cohort overview: one row per discovered patient + a wear-hours bar."""
    rows: list[dict[str, Any]] = []

    iter_records: list[tuple[str, Any, float]] = [
        (r.subject_id, str(r.path), r.mtime) for r in records
    ]
    if synthetic_available and not records:
        iter_records.append(("SYNTH001", "__synth__", 0.0))

    for subject_id, path_or_synth, mtime in iter_records:
        if path_or_synth == "__synth__":
            data, start_dt, _ = dl.get_synthetic_patient()
        else:
            data, start_dt, _ = dl.load_patient(path_or_synth, mtime)
        result = run_pipeline(subject_id, data, start_dt, config)
        wh = result.get("wear_hours_by_day", {})
        rows.append({
            "subject_id": subject_id,
            "samples": data.shape[0],
            "duration_h": round(data.shape[0] / (30 * 3600), 2),
            "valid": "PASS" if result["valid"] else "FAIL",
            "reason": result.get("reason", ""),
            "n_days_logged": len(wh),
            "max_wear_h": round(max(wh.values()), 2) if wh else 0.0,
            "_wear_hours": wh,
        })

    if not rows:
        st.info("No patients found and synthetic generation is disabled.")
        return

    table_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    st.dataframe(table_rows, width='stretch', hide_index=True)

    st.markdown("**Wear hours per day**")
    cols = st.columns(min(4, len(rows)))
    cfg = PreprocessConfig(**config)
    for i, row in enumerate(rows):
        with cols[i % len(cols)]:
            st.caption(row["subject_id"])
            st.plotly_chart(
                plots.wear_hours_bar(row["_wear_hours"],
                                     min_daily_wear_hours=cfg.min_daily_wear_hours,
                                     height=160),
                width='stretch',
                key=f"wear_bar_{row['subject_id']}",
            )


# ---------------------------------------------------------------------------
# Invalid-subject panel
# ---------------------------------------------------------------------------

def invalid_subject_panel(result: dict[str, Any], config: dict[str, Any],
                          key_suffix: str = "") -> None:
    """Render the failure-reason banner + per-day wear bar.

    ``key_suffix`` disambiguates the embedded ``plotly_chart`` when the panel
    is rendered on more than one tab in the same run (Streamlit's auto-id
    deduper otherwise rejects the second copy).
    """
    cfg = PreprocessConfig(**config)
    st.error(f"Subject failed validation: {result.get('reason', 'unknown')}")
    st.caption(
        f"Requirement: >= {cfg.min_valid_days} days with >= "
        f"{cfg.min_daily_wear_hours:g} h daytime wear "
        f"({cfg.daytime_start_hour:02d}:00–{cfg.daytime_end_hour:02d}:00)."
    )
    st.plotly_chart(
        plots.wear_hours_bar(result.get("wear_hours_by_day", {}),
                             min_daily_wear_hours=cfg.min_daily_wear_hours),
        width='stretch',
        key=f"invalid_wear_bar_{key_suffix}",
    )
