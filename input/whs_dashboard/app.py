"""Streamlit entry point for the WHS preprocessor dashboard.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from dashboard import components, plots
from dashboard.pipeline_runner import run_pipeline
from whs_preprocessor import PreprocessConfig, WHSPreprocessor


st.set_page_config(page_title="WHS Accelerometer Dashboard", layout="wide")
st.title("WHS Accelerometer Explorer")
st.caption(
    "Visual validation for the hip-worn ActiGraph GT3X+ preprocessing pipeline "
    "(ENMO + macro/micro windows + Choi non-wear + gait FFT filter)."
)

sb = components.sidebar()
data, start_datetime, subject_id = sb["patient"]
config = sb["config"]
records = sb["patients"]
cfg = PreprocessConfig(**config)

result = run_pipeline(subject_id, data, start_datetime, config)

tab_cohort, tab_raw, tab_macro, tab_micro = st.tabs(
    ["Cohort overview", "Raw signal", "Macro (mortality / circadian)", "Micro (gait / falls)"]
)

# ---------------------------------------------------------------------------
# 1. Cohort overview
# ---------------------------------------------------------------------------
with tab_cohort:
    st.subheader("Discovered patients")
    components.cohort_table(records, config, synthetic_available=True)

# ---------------------------------------------------------------------------
# 2. Raw signal
# ---------------------------------------------------------------------------
with tab_raw:
    st.subheader(f"Raw triaxial — {subject_id}")
    total_hours = data.shape[0] / (cfg.fs * 3600.0)

    col_a, col_b, col_c = st.columns([3, 1, 1])
    with col_a:
        t_range_h = st.slider(
            "Time window (hours from recording start)",
            min_value=0.0, max_value=float(total_hours),
            value=(0.0, float(total_hours)),
            step=0.25,
            help=("Drag to zoom. When the window is shorter than 1 hour the plot "
                  "switches to full-resolution; otherwise it is stride-downsampled "
                  "to ~50k points."),
        )
    with col_b:
        show_enmo = st.checkbox("Overlay ENMO", value=False)
    with col_c:
        show_nonwear = st.checkbox("Shade non-wear", value=True,
                                   disabled=not result["valid"])

    time_range_sec = (t_range_h[0] * 3600.0, t_range_h[1] * 3600.0)
    enmo_arr = WHSPreprocessor.compute_enmo(data) if show_enmo else None
    wear_mask = result["macro"]["wear_mask"] if (show_nonwear and result["valid"]) else None

    fig = plots.raw_triaxial_figure(
        data, fs=cfg.fs,
        enmo=enmo_arr,
        wear_mask=wear_mask,
        macro_size_sec=cfg.macro_size_sec,
        time_range_sec=time_range_sec,
    )
    st.plotly_chart(fig, width='stretch')

    st.caption(
        f"Selected window: {t_range_h[1] - t_range_h[0]:.2f} h. "
        f"Full-res threshold: 1 h (set on `raw_triaxial_figure`)."
    )

# ---------------------------------------------------------------------------
# 3. Macro view
# ---------------------------------------------------------------------------
with tab_macro:
    st.subheader(f"Macro features — {subject_id}")
    if not result["valid"]:
        components.invalid_subject_panel(result, config, key_suffix="macro")
    else:
        macro = result["macro"]
        feats = result["macro_features"]

        components.kpi_row([
            ("Total sedentary (min)", f"{feats['total_sedentary_min']:.0f}"),
            ("Mean bout (min)", f"{feats['mean_bout_min']:.1f}"),
            ("Max bout (min)", f"{feats['max_bout_min']:.1f}"),
            ("# bouts", f"{feats['n_bouts']}"),
        ])
        components.kpi_row([
            ("Light (min)", f"{feats['light_min']:.0f}"),
            ("Moderate (min)", f"{feats['moderate_min']:.0f}"),
            ("Vigorous (min)", f"{feats['vigorous_min']:.0f}"),
            ("MVPA (min)", f"{feats['mvpa_min']:.0f}"),
        ])

        st.plotly_chart(
            plots.actigraphy_double_plot(
                macro["full_enmo"], macro["full_timestamps"], macro["wear_mask"],
                start_datetime, macro_size_sec=cfg.macro_size_sec,
            ),
            width='stretch',
        )

        st.plotly_chart(
            plots.sedentary_bout_histogram(feats["bout_lengths_min"]),
            width='stretch',
        )

# ---------------------------------------------------------------------------
# 4. Micro view
# ---------------------------------------------------------------------------
with tab_micro:
    st.subheader(f"Micro features (gait bouts) — {subject_id}")
    if not result["valid"]:
        components.invalid_subject_panel(result, config, key_suffix="micro")
    else:
        micro = result["micro"]
        gfeats = result["micro_features"]
        n_bouts = micro["data"].shape[0]

        if n_bouts == 0:
            st.warning("Subject passed wear-validity but the gait filter retained no bouts.")
        else:
            rms_xyz = np.stack([gfeats["rms_x"], gfeats["rms_y"], gfeats["rms_z"]], axis=1)
            p50, p90, p99 = np.percentile(np.linalg.norm(rms_xyz, axis=1), [50, 90, 99])

            components.kpi_row([
                ("# gait bouts", f"{gfeats['n_gait_bouts']}"),
                ("Mean dom freq (Hz)", f"{gfeats['dom_freq_mean']:.2f}"),
                ("Std dom freq (Hz)", f"{gfeats['dom_freq_std']:.2f}"),
                ("RMS p50 / p90 / p99 (g)", f"{p50:.2f} / {p90:.2f} / {p99:.2f}"),
            ])

            scatter_fig = plots.gait_scatter(
                gfeats["dom_freq_hz"], rms_xyz, micro["timestamps"], start_datetime,
            )
            event = st.plotly_chart(
                scatter_fig, width='stretch',
                key="gait_scatter", on_select="rerun",
                selection_mode=("points", "box", "lasso"),
            )

            selected_points = []
            try:
                selected_points = event["selection"]["points"]
            except (KeyError, TypeError):
                pass

            if selected_points:
                bout_idx = int(selected_points[-1]["point_index"])
                st.session_state["last_bout_idx"] = bout_idx
            elif "last_bout_idx" not in st.session_state:
                st.session_state["last_bout_idx"] = int(
                    np.random.default_rng(0).integers(0, n_bouts)
                )
            bout_idx = min(st.session_state["last_bout_idx"], n_bouts - 1)

            st.markdown(
                f"**Inspected bout #{bout_idx}** — "
                f"dom freq {gfeats['dom_freq_hz'][bout_idx]:.2f} Hz, "
                f"t = {micro['timestamps'][bout_idx]:.0f} s "
                f"({len(selected_points)} point(s) selected)"
            )

            col_l, col_r = st.columns(2)
            with col_l:
                st.plotly_chart(
                    plots.gait_freq_histogram(
                        gfeats["dom_freq_hz"],
                        gait_band=(cfg.gait_freq_low_hz, cfg.gait_freq_high_hz),
                    ),
                    width='stretch',
                )
            with col_r:
                st.plotly_chart(
                    plots.bout_signal_and_spectrum(
                        micro["data"][bout_idx], micro["enmo"][bout_idx],
                        fs=cfg.fs,
                        gait_band=(cfg.gait_freq_low_hz, cfg.gait_freq_high_hz),
                    ),
                    width='stretch',
                )
