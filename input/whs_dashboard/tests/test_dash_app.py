"""Smoke tests for ``app_dash.py``: layout builds + each callback returns a Figure.

Dash callbacks are decorated module-level functions; the decorator returns the
original function so we can invoke them directly without spinning up the server.
"""
from __future__ import annotations

import plotly.graph_objects as go
import pytest

dash = pytest.importorskip("dash")  # gracefully skip if Dash isn't available

import app_dash as A  # noqa: E402


def test_module_layout_builds():
    layout = A._build_layout()
    assert layout is not None
    # Sidebar Div is the first child, content Div is the second.
    assert len(layout.children) == 2


def test_subjects_discovered():
    # Real example file + synthetic fallback should both be present.
    ids = [s["subject_id"] for s in A._SUBJECTS]
    assert A.SUBJECT_SYNTH_ID in ids
    assert len(ids) >= 1


def test_aggregate_config_round_trip():
    specs = A.config_field_specs()
    values = [A.DEFAULT_CONFIG[name] for name, _, _ in specs]
    ids = [{"type": "cfg", "name": name} for name, _, _ in specs]
    out = A.aggregate_config(values, ids)
    assert out == A.DEFAULT_CONFIG


def test_on_subject_change_synthetic():
    info, max_h, value, date_iso, time_str = A.on_subject_change(A.SUBJECT_SYNTH_ID)
    assert "samples" in info
    assert max_h > 100  # synthetic is 7 days
    assert value[0] == 0 and value[1] == max_h


def test_render_raw_returns_figure():
    fig, caption = A.render_raw(
        A.SUBJECT_SYNTH_ID,
        [0, 168],          # 7 days
        ["nonwear"],
        A.DEFAULT_CONFIG,
        "2024-01-01",
        "00:00:00",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 3  # x, y, z
    assert "h." in caption


def test_render_macro_synthetic():
    body = A.render_macro(
        A.SUBJECT_SYNTH_ID,
        A.DEFAULT_CONFIG,
        "2024-01-01",
        "00:00:00",
    )
    assert body is not None


def test_render_micro_synthetic():
    body = A.render_micro(
        A.SUBJECT_SYNTH_ID,
        A.DEFAULT_CONFIG,
        None,
        "2024-01-01",
        "00:00:00",
    )
    assert body is not None


def test_update_cohort_yields_one_row_per_subject():
    rows, bars = A.update_cohort(A.DEFAULT_CONFIG)
    assert len(rows) == len(A._SUBJECTS)
    assert len(bars) == len(A._SUBJECTS)
    cols = {"subject_id", "samples", "duration_h", "valid", "reason",
            "n_days_logged", "max_wear_h"}
    assert set(rows[0].keys()) == cols


def test_cohort_row_clicked_promotes_subject():
    rows, _ = A.update_cohort(A.DEFAULT_CONFIG)
    # Pick the synthetic row, ensure its id is returned.
    synth_idx = next(i for i, r in enumerate(rows)
                     if r["subject_id"] == A.SUBJECT_SYNTH_ID)
    out = A.cohort_row_clicked([synth_idx], rows)
    assert out == A.SUBJECT_SYNTH_ID


def test_invalid_subject_short_recording():
    """Real example file is ~18 h — it must fail validation and show the panel."""
    # Find a non-synthetic subject if any (the real example).
    real = [s for s in A._SUBJECTS if s["subject_id"] != A.SUBJECT_SYNTH_ID]
    if not real:
        pytest.skip("no real example file present")
    sid = real[0]["subject_id"]
    body = A.render_macro(sid, A.DEFAULT_CONFIG, "2020-01-12", "00:00:00")
    # The invalid panel contains the "Subject failed validation:" banner.
    serialized = str(body)
    assert "failed validation" in serialized


def test_reset_config_restores_defaults():
    ids = [{"type": "cfg", "name": name}
           for name, _, _ in A.config_field_specs()]
    out = A.reset_config(1, ids)
    assert len(out) == len(ids)
    # Bool defaults are returned as the list form ["on"] or [], everything else
    # is returned as the raw default value.
    for v, id_ in zip(out, ids):
        name = id_["name"]
        type_ = A.CONFIG_TYPES[name]
        default = A.DEFAULT_CONFIG[name]
        if type_ is bool:
            assert v == (["on"] if default else [])
        else:
            assert v == default
