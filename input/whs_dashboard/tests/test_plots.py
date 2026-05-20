"""Smoke tests for every plot in ``dashboard.plots``.

We feed the synthetic subject through the real pipeline (no Streamlit-cache
indirection) and assert each figure has the expected structure.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from dashboard import plots
from dashboard.data_loader import generate_synthetic_subject
from whs_preprocessor import (
    MacroFeatureExtractor,
    MicroFeatureExtractor,
    PreprocessConfig,
    WHSPreprocessor,
)


# ---------------------------------------------------------------------------
# Shared fixture — one synthetic subject run end-to-end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_result():
    cfg = PreprocessConfig()
    data, start_dt, subject_id = generate_synthetic_subject(seed=0)
    pre = WHSPreprocessor(cfg)
    result = pre.process(data, start_datetime=start_dt, subject_id=subject_id)
    assert result["valid"], "synthetic subject should pass wear-validity"
    result["macro_features"] = MacroFeatureExtractor(cfg).extract_all(result["macro"])
    result["micro_features"] = MicroFeatureExtractor(cfg).extract_all(result["micro"])
    return cfg, data, start_dt, result


def _assert_nonempty_figure(fig, expected_min_traces: int = 1):
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= expected_min_traces, (
        f"expected at least {expected_min_traces} traces, got {len(fig.data)}"
    )
    # at least one trace must carry data
    has_data = any(
        (getattr(t, "x", None) is not None and len(t.x) > 0)
        or (getattr(t, "z", None) is not None and np.asarray(t.z).size > 0)
        for t in fig.data
    )
    assert has_data, "all traces are empty"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_wear_hours_bar(synthetic_result):
    cfg, _, _, result = synthetic_result
    fig = plots.wear_hours_bar(result["wear_hours_by_day"],
                               min_daily_wear_hours=cfg.min_daily_wear_hours)
    _assert_nonempty_figure(fig, expected_min_traces=1)
    assert fig.data[0].type == "bar"


def test_wear_hours_bar_handles_empty():
    fig = plots.wear_hours_bar({}, min_daily_wear_hours=10.0)
    assert isinstance(fig, go.Figure)


def test_raw_triaxial_overview(synthetic_result):
    cfg, data, _, result = synthetic_result
    fig = plots.raw_triaxial_figure(
        data, fs=cfg.fs,
        wear_mask=result["macro"]["wear_mask"],
        macro_size_sec=cfg.macro_size_sec,
    )
    _assert_nonempty_figure(fig, expected_min_traces=3)  # x, y, z
    types = {t.type for t in fig.data}
    assert types == {"scattergl"}


def test_raw_triaxial_full_res_zoom(synthetic_result):
    cfg, data, _, _ = synthetic_result
    fig = plots.raw_triaxial_figure(
        data, fs=cfg.fs,
        time_range_sec=(0.0, 60.0),  # 1-min window -> full res, stride=1
    )
    _assert_nonempty_figure(fig, expected_min_traces=3)
    # full-res for 60s @ 30Hz -> 1800 samples per trace
    assert len(fig.data[0].x) == 60 * cfg.fs


def test_raw_triaxial_with_enmo_overlay(synthetic_result):
    cfg, data, _, _ = synthetic_result
    enmo = WHSPreprocessor.compute_enmo(data)
    fig = plots.raw_triaxial_figure(data, fs=cfg.fs, enmo=enmo)
    _assert_nonempty_figure(fig, expected_min_traces=4)  # x, y, z, ENMO


def test_actigraphy_double_plot(synthetic_result):
    cfg, _, start_dt, result = synthetic_result
    macro = result["macro"]
    fig = plots.actigraphy_double_plot(
        macro["full_enmo"], macro["full_timestamps"], macro["wear_mask"],
        start_dt, macro_size_sec=cfg.macro_size_sec,
    )
    _assert_nonempty_figure(fig, expected_min_traces=2)  # ENMO + non-wear overlay
    assert all(t.type == "heatmap" for t in fig.data)


def test_sedentary_bout_histogram(synthetic_result):
    _, _, _, result = synthetic_result
    feats = result["macro_features"]
    fig = plots.sedentary_bout_histogram(feats["bout_lengths_min"])
    assert isinstance(fig, go.Figure)
    # empty arrays are handled gracefully — only assert trace count when non-empty
    if feats["bout_lengths_min"].size > 0:
        assert fig.data[0].type == "histogram"


def test_gait_scatter(synthetic_result):
    _, _, start_dt, result = synthetic_result
    gfeats = result["micro_features"]
    rms = np.stack([gfeats["rms_x"], gfeats["rms_y"], gfeats["rms_z"]], axis=1)
    fig = plots.gait_scatter(gfeats["dom_freq_hz"], rms,
                             result["micro"]["timestamps"], start_dt)
    _assert_nonempty_figure(fig, expected_min_traces=1)
    assert fig.data[0].type == "scattergl"
    assert fig.layout.dragmode == "lasso"


def test_gait_freq_histogram(synthetic_result):
    cfg, _, _, result = synthetic_result
    gfeats = result["micro_features"]
    fig = plots.gait_freq_histogram(
        gfeats["dom_freq_hz"],
        gait_band=(cfg.gait_freq_low_hz, cfg.gait_freq_high_hz),
    )
    _assert_nonempty_figure(fig, expected_min_traces=1)


def test_bout_signal_and_spectrum(synthetic_result):
    cfg, _, _, result = synthetic_result
    micro = result["micro"]
    assert micro["data"].shape[0] > 0, "synthetic subject should yield gait bouts"
    fig = plots.bout_signal_and_spectrum(
        micro["data"][0], micro["enmo"][0], fs=cfg.fs,
        gait_band=(cfg.gait_freq_low_hz, cfg.gait_freq_high_hz),
    )
    # 3 axes + ENMO on the left, |FFT| on the right -> 5 traces
    _assert_nonempty_figure(fig, expected_min_traces=5)


def test_nonwear_segments_diff_only(synthetic_result):
    _, _, _, result = synthetic_result
    segs = plots.nonwear_segments(result["macro"]["wear_mask"],
                                  macro_size_sec=PreprocessConfig().macro_size_sec)
    # Synthetic subject has overnight off-wrist periods -> at least one run.
    assert all(e > s for s, e in segs), "every non-wear segment must be non-empty"
