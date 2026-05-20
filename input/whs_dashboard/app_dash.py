"""Dash port of the WHS preprocessor dashboard.

Sister entry point to ``app.py``. Layout, plots, and pipeline are identical;
the only difference is the reactivity model — Dash spells out the callback
graph explicitly instead of re-running the script on every interaction.

Run:
    python app_dash.py
"""
from __future__ import annotations

import os
# Opt out of Streamlit's cache layer BEFORE importing the dashboard package,
# so its ``@cache_data`` decorators no-op instead of routing through Streamlit
# (whose "No runtime found, using MemoryCacheStorageManager" warnings would
# otherwise spam the log on every cached call). Our own ``functools.lru_cache``
# wrappers below provide the caching.
os.environ.setdefault("WHS_DASHBOARD_NO_STREAMLIT_CACHE", "1")

import functools
from datetime import datetime, time as dt_time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# Modern Dash (2.0+) bundles dcc / html / dash_table at the top level; the
# pre-2.0 packages are still installable as compat shims, so we accept both.
try:
    from dash import Dash, Input, Output, State, ALL, no_update, dcc, html, dash_table
except ImportError:  # Dash 1.x with separate sub-packages
    from dash import Dash, no_update
    from dash.dependencies import Input, Output, State, ALL
    import dash_core_components as dcc  # type: ignore
    import dash_html_components as html  # type: ignore
    import dash_table  # type: ignore

from dashboard import plots
from dashboard.data_loader import (
    _resolve_loader,
    discover_patients,
    generate_synthetic_subject,
)
from dashboard.pipeline_runner import config_field_specs, run_pipeline_core
from whs_preprocessor import PreprocessConfig, WHSPreprocessor


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "./example_data/"
SUBJECT_SYNTH_KEY = "__synth__"
SUBJECT_SYNTH_ID = "SYNTH001"

DEFAULT_CONFIG: dict = {name: default for name, _, default in config_field_specs()}
CONFIG_TYPES: dict[str, type] = {name: type_ for name, type_, _ in config_field_specs()}


# ---------------------------------------------------------------------------
# Subject discovery + caching
# ---------------------------------------------------------------------------

def _list_subjects() -> list[dict]:
    """Discover real subjects + always include the synthetic demo at the end."""
    records = discover_patients(DEFAULT_DATA_DIR)
    out = [
        {"subject_id": r.subject_id, "path": str(r.path), "mtime": r.mtime,
         "size_bytes": r.size_bytes, "suffix": r.suffix}
        for r in records
    ]
    out.append({
        "subject_id": SUBJECT_SYNTH_ID,
        "path": SUBJECT_SYNTH_KEY,
        "mtime": 0.0,
        "size_bytes": 0,
        "suffix": "synthetic",
    })
    return out


_SUBJECTS = _list_subjects()
_SUBJECT_BY_ID = {s["subject_id"]: s for s in _SUBJECTS}


@functools.lru_cache(maxsize=8)
def _load_data_cached(path: str, mtime: float):
    if path == SUBJECT_SYNTH_KEY:
        return generate_synthetic_subject()
    p = Path(path)
    _, loader = _resolve_loader(p)
    if loader is None:
        raise ValueError(f"No loader registered for {p.name!r}")
    return loader(p)


def _resolve_subject(subject_id: str) -> tuple[np.ndarray, datetime, str]:
    s = _SUBJECT_BY_ID.get(subject_id)
    if s is None:
        raise ValueError(f"Unknown subject_id: {subject_id}")
    return _load_data_cached(s["path"], s["mtime"])


@functools.lru_cache(maxsize=16)
def _pipeline_cached(subject_id: str, start_dt_iso: str, config_items: tuple) -> dict:
    data, _, _ = _resolve_subject(subject_id)
    start_dt = datetime.fromisoformat(start_dt_iso)
    return run_pipeline_core(subject_id, data, start_dt, dict(config_items))


def get_result(subject_id: str, start_dt: datetime, config_dict: dict) -> dict:
    return _pipeline_cached(subject_id, start_dt.isoformat(),
                            tuple(sorted(config_dict.items())))


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _config_inputs() -> list:
    """One labeled input per PreprocessConfig field, with pattern-matched ids."""
    children = []
    for name, type_, default in config_field_specs():
        cid = {"type": "cfg", "name": name}
        if type_ is bool:
            input_el = dcc.Checklist(
                id=cid,
                options=[{"label": "", "value": "on"}],
                value=["on"] if default else [],
                style={"marginLeft": "0.5rem"},
            )
        elif type_ is int:
            input_el = dcc.Input(id=cid, type="number", step=1,
                                 value=int(default),
                                 style={"width": "100%"})
        elif type_ is float:
            input_el = dcc.Input(id=cid, type="number", step=0.001,
                                 value=float(default),
                                 style={"width": "100%"})
        else:
            input_el = dcc.Input(id=cid, type="text", value=str(default),
                                 style={"width": "100%"})
        children.append(html.Div([
            html.Label(name.replace("_", " "),
                       style={"fontSize": "0.78em", "color": "#444"}),
            input_el,
        ], style={"marginBottom": "0.35rem"}))
    return children


def _kpi_grid(items: list[tuple[str, str]]):
    return html.Div([
        html.Div([
            html.Div(label, style={"fontSize": "0.75em", "color": "#888"}),
            html.Div(value, style={"fontSize": "1.35em", "fontWeight": 600,
                                   "marginTop": "0.15rem"}),
        ], style={"padding": "0.55rem 0.8rem",
                  "border": "1px solid #eee",
                  "borderRadius": "4px",
                  "background": "#fafafa"})
        for label, value in items
    ], style={"display": "grid",
              "gridTemplateColumns": "repeat(auto-fill, minmax(170px, 1fr))",
              "gap": "0.5rem",
              "margin": "0.5rem 0 1rem"})


def _invalid_subject_panel(result: dict, cfg: PreprocessConfig):
    return html.Div([
        html.Div(f"Subject failed validation: {result.get('reason', 'unknown')}",
                 style={"color": "#a00", "fontWeight": 600,
                        "padding": "0.6rem 0.8rem",
                        "background": "#fff3e0",
                        "border": "1px solid #ffcc80",
                        "borderRadius": "4px",
                        "marginBottom": "0.5rem"}),
        html.Div(
            f"Requirement: ≥ {cfg.min_valid_days} days with ≥ "
            f"{cfg.min_daily_wear_hours:g} h daytime wear "
            f"({cfg.daytime_start_hour:02d}:00–{cfg.daytime_end_hour:02d}:00).",
            style={"color": "#666", "fontSize": "0.85em",
                   "marginBottom": "0.5rem"},
        ),
        dcc.Graph(
            figure=plots.wear_hours_bar(
                result.get("wear_hours_by_day", {}),
                min_daily_wear_hours=cfg.min_daily_wear_hours,
            ),
            config={"displayModeBar": False},
        ),
    ])


def _combine_date_time(date_str: str | None, time_str: str | None,
                       fallback: datetime) -> datetime:
    """Robustly combine the date-picker + text time field; fall back on bad input."""
    try:
        d = datetime.fromisoformat(date_str).date() if date_str else fallback.date()
    except (ValueError, TypeError):
        d = fallback.date()
    try:
        parts = (time_str or "").split(":")
        t = dt_time(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, TypeError, IndexError):
        t = fallback.time()
    return datetime.combine(d, t)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _initial_subject() -> str | None:
    if not _SUBJECTS:
        return None
    return _SUBJECTS[0]["subject_id"]


def _initial_start_iso() -> str:
    sid = _initial_subject()
    if sid is None:
        return "2020-01-12"
    _, default_start, _ = _resolve_subject(sid)
    return default_start.date().isoformat()


def _initial_start_time() -> str:
    sid = _initial_subject()
    if sid is None:
        return "00:00:00"
    _, default_start, _ = _resolve_subject(sid)
    return default_start.strftime("%H:%M:%S")


def _build_sidebar():
    return html.Div([
        html.H3("Patient", style={"marginTop": 0}),
        dcc.Dropdown(
            id="subject-dropdown",
            options=[{"label": s["subject_id"], "value": s["subject_id"]}
                     for s in _SUBJECTS],
            value=_initial_subject(),
            clearable=False,
        ),
        html.Div(id="subject-info",
                 style={"fontSize": "0.8em", "color": "#666",
                        "marginTop": "0.3rem", "marginBottom": "0.8rem"}),

        html.Label("Recording start date"),
        dcc.DatePickerSingle(
            id="start-date",
            date=_initial_start_iso(),
            display_format="YYYY-MM-DD",
            style={"width": "100%"},
        ),
        html.Label("Recording start time", style={"marginTop": "0.4rem"}),
        dcc.Input(id="start-time", type="text",
                  value=_initial_start_time(),
                  placeholder="HH:MM:SS",
                  style={"width": "100%"}),

        html.Hr(),
        html.Details([
            html.Summary("Preprocessing config",
                         style={"cursor": "pointer", "fontWeight": 600}),
            html.Div(_config_inputs(),
                     style={"maxHeight": "460px",
                            "overflowY": "auto",
                            "padding": "0.4rem 0.2rem"}),
            html.Button("Reset to defaults", id="cfg-reset", n_clicks=0,
                        style={"marginTop": "0.4rem", "width": "100%"}),
        ], open=False),

        dcc.Store(id="store-config", data=DEFAULT_CONFIG),
        dcc.Store(id="store-last-bout-idx", data=None),
    ], style={"width": "300px",
              "padding": "1rem",
              "borderRight": "1px solid #ddd",
              "height": "100vh",
              "overflowY": "auto",
              "position": "fixed", "top": 0, "left": 0,
              "background": "#fafafa",
              "boxSizing": "border-box"})


def _build_cohort_tab():
    return html.Div([
        html.H4("Discovered patients"),
        dash_table.DataTable(
            id="cohort-table",
            columns=[
                {"name": "Subject", "id": "subject_id"},
                {"name": "Samples", "id": "samples", "type": "numeric"},
                {"name": "Duration (h)", "id": "duration_h", "type": "numeric"},
                {"name": "Valid", "id": "valid"},
                {"name": "Days", "id": "n_days_logged", "type": "numeric"},
                {"name": "Max wear (h)", "id": "max_wear_h", "type": "numeric"},
                {"name": "Reason", "id": "reason"},
            ],
            row_selectable="single",
            selected_rows=[0],
            sort_action="native",
            style_cell={"fontSize": "0.85em", "padding": "0.4rem",
                        "fontFamily": "system-ui, sans-serif"},
            style_header={"fontWeight": 600, "background": "#f0f0f0"},
            style_data_conditional=[
                {"if": {"filter_query": '{valid} = "FAIL"'},
                 "backgroundColor": "#fff3e0"},
                {"if": {"filter_query": '{valid} = "PASS"'},
                 "backgroundColor": "#f1faf1"},
            ],
        ),
        html.H5("Wear hours per day", style={"marginTop": "1.2rem"}),
        html.Div(id="cohort-wear-bars",
                 style={"display": "grid",
                        "gridTemplateColumns": "repeat(auto-fill, minmax(240px, 1fr))",
                        "gap": "0.5rem"}),
    ], style={"padding": "1rem 0.5rem"})


def _build_raw_tab():
    return html.Div([
        html.Div([
            html.Label("Time window (hours from recording start)"),
            dcc.RangeSlider(id="raw-time-range",
                            min=0, max=24, step=0.25,
                            value=[0, 24],
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": False},
                            allowCross=False),
        ], style={"marginBottom": "0.6rem"}),
        dcc.Checklist(
            id="raw-toggles",
            options=[
                {"label": " Overlay ENMO", "value": "enmo"},
                {"label": " Shade non-wear", "value": "nonwear"},
            ],
            value=["nonwear"],
            inline=True,
            inputStyle={"marginRight": "0.3rem", "marginLeft": "1rem"},
        ),
        dcc.Loading(
            id="raw-loading",
            type="default",
            children=dcc.Graph(id="raw-graph"),
        ),
        html.Div(id="raw-caption",
                 style={"fontSize": "0.8em", "color": "#666",
                        "marginTop": "0.3rem"}),
    ], style={"padding": "1rem 0.5rem"})


def _build_macro_tab():
    return html.Div([
        html.Div(id="macro-content"),
    ], style={"padding": "1rem 0.5rem"})


def _build_micro_tab():
    return html.Div([
        html.Div(id="micro-content"),
    ], style={"padding": "1rem 0.5rem"})


def _build_layout():
    return html.Div([
        _build_sidebar(),
        html.Div([
            html.H2("WHS Accelerometer Explorer",
                    style={"marginTop": 0, "marginBottom": "0.2rem"}),
            html.P(
                "Visual validation for the hip-worn ActiGraph GT3X+ preprocessing "
                "pipeline (ENMO + macro/micro windows + Choi non-wear + gait FFT filter).",
                style={"color": "#666", "marginTop": 0},
            ),
            dcc.Tabs(id="main-tabs", value="tab-cohort", children=[
                dcc.Tab(label="Cohort overview", value="tab-cohort",
                        children=_build_cohort_tab()),
                dcc.Tab(label="Raw signal", value="tab-raw",
                        children=_build_raw_tab()),
                dcc.Tab(label="Macro (mortality / circadian)", value="tab-macro",
                        children=_build_macro_tab()),
                dcc.Tab(label="Micro (gait / falls)", value="tab-micro",
                        children=_build_micro_tab()),
            ]),
        ], style={"marginLeft": "320px", "padding": "1rem"})
    ])


app = Dash(__name__, suppress_callback_exceptions=True)  # micro-scatter is rendered
                                                          # inside a callback, so its
                                                          # id isn't in the initial layout
app.title = "WHS Accelerometer Dashboard (Dash)"
app.layout = _build_layout


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("store-config", "data"),
    Input({"type": "cfg", "name": ALL}, "value"),
    State({"type": "cfg", "name": ALL}, "id"),
)
def aggregate_config(values, ids):
    """Roll the 16 individual config inputs into one dict in ``store-config``."""
    out: dict = {}
    for value, id_ in zip(values, ids):
        name = id_["name"]
        type_ = CONFIG_TYPES.get(name)
        if value is None:
            out[name] = DEFAULT_CONFIG[name]
            continue
        if type_ is bool:
            out[name] = bool(value) if isinstance(value, bool) else ("on" in (value or []))
        elif type_ is int:
            try:
                out[name] = int(value)
            except (TypeError, ValueError):
                out[name] = DEFAULT_CONFIG[name]
        elif type_ is float:
            try:
                out[name] = float(value)
            except (TypeError, ValueError):
                out[name] = DEFAULT_CONFIG[name]
        else:
            out[name] = value
    return out


@app.callback(
    Output({"type": "cfg", "name": ALL}, "value"),
    Input("cfg-reset", "n_clicks"),
    State({"type": "cfg", "name": ALL}, "id"),
    prevent_initial_call=True,
)
def reset_config(n_clicks, ids):
    """Restore every config field to its PreprocessConfig default."""
    out = []
    for id_ in ids:
        name = id_["name"]
        default = DEFAULT_CONFIG[name]
        type_ = CONFIG_TYPES.get(name)
        if type_ is bool:
            out.append(["on"] if default else [])
        else:
            out.append(default)
    return out


@app.callback(
    Output("subject-info", "children"),
    Output("raw-time-range", "max"),
    Output("raw-time-range", "value"),
    Output("start-date", "date"),
    Output("start-time", "value"),
    Input("subject-dropdown", "value"),
)
def on_subject_change(subject_id):
    """When the subject changes: refresh caption + reset the raw-tab time slider
    + seed the recording-start widgets from the loader's default."""
    if subject_id is None:
        return "no subject", 24.0, [0, 24.0], no_update, no_update
    data, default_start, _ = _resolve_subject(subject_id)
    total_h = float(data.shape[0] / (30 * 3600))
    info = f"{data.shape[0]:,} samples ({total_h:.2f} h @ 30 Hz)"
    return (info, total_h, [0.0, total_h],
            default_start.date().isoformat(),
            default_start.strftime("%H:%M:%S"))


@app.callback(
    Output("cohort-table", "data"),
    Output("cohort-wear-bars", "children"),
    Input("store-config", "data"),
)
def update_cohort(config):
    """Refresh the cohort table whenever the config changes. Cached pipeline calls
    keep this cheap on repeat renders."""
    cfg = PreprocessConfig(**config)
    rows: list = []
    bars: list = []
    for s in _SUBJECTS:
        sid = s["subject_id"]
        data, default_start, _ = _resolve_subject(sid)
        result = get_result(sid, default_start, config)
        wh = result.get("wear_hours_by_day", {})
        rows.append({
            "subject_id": sid,
            "samples": int(data.shape[0]),
            "duration_h": round(data.shape[0] / (30 * 3600), 2),
            "valid": "PASS" if result["valid"] else "FAIL",
            "reason": result.get("reason", ""),
            "n_days_logged": len(wh),
            "max_wear_h": round(max(wh.values()), 2) if wh else 0.0,
        })
        bars.append(html.Div([
            html.Div(sid, style={"fontSize": "0.8em", "color": "#666",
                                  "marginBottom": "0.2rem"}),
            dcc.Graph(
                figure=plots.wear_hours_bar(
                    wh, min_daily_wear_hours=cfg.min_daily_wear_hours, height=160),
                config={"displayModeBar": False},
            ),
        ]))
    return rows, bars


@app.callback(
    Output("subject-dropdown", "value"),
    Input("cohort-table", "selected_rows"),
    State("cohort-table", "data"),
    prevent_initial_call=True,
)
def cohort_row_clicked(selected_rows, table_data):
    """Clicking a row in the cohort table promotes that subject to active."""
    if not selected_rows or not table_data:
        return no_update
    return table_data[selected_rows[0]]["subject_id"]


@app.callback(
    Output("raw-graph", "figure"),
    Output("raw-caption", "children"),
    Input("subject-dropdown", "value"),
    Input("raw-time-range", "value"),
    Input("raw-toggles", "value"),
    Input("store-config", "data"),
    Input("start-date", "date"),
    Input("start-time", "value"),
)
def render_raw(subject_id, time_range, toggles, config, start_date, start_time):
    if subject_id is None or time_range is None:
        return go.Figure(), ""
    cfg = PreprocessConfig(**config)
    data, default_start, _ = _resolve_subject(subject_id)
    start_dt = _combine_date_time(start_date, start_time, default_start)
    toggles = toggles or []
    enmo = WHSPreprocessor.compute_enmo(data) if "enmo" in toggles else None
    wear_mask = None
    if "nonwear" in toggles:
        result = get_result(subject_id, start_dt, config)
        if result["valid"]:
            wear_mask = result["macro"]["wear_mask"]
    fig = plots.raw_triaxial_figure(
        data, fs=cfg.fs,
        enmo=enmo, wear_mask=wear_mask,
        macro_size_sec=cfg.macro_size_sec,
        time_range_sec=(time_range[0] * 3600.0, time_range[1] * 3600.0),
    )
    caption = (f"Selected window: {time_range[1] - time_range[0]:.2f} h. "
               f"Full-res threshold: 1 h.")
    return fig, caption


@app.callback(
    Output("macro-content", "children"),
    Input("subject-dropdown", "value"),
    Input("store-config", "data"),
    Input("start-date", "date"),
    Input("start-time", "value"),
)
def render_macro(subject_id, config, start_date, start_time):
    if subject_id is None:
        return html.Div("no subject selected")
    cfg = PreprocessConfig(**config)
    data, default_start, _ = _resolve_subject(subject_id)
    start_dt = _combine_date_time(start_date, start_time, default_start)
    result = get_result(subject_id, start_dt, config)
    if not result["valid"]:
        return _invalid_subject_panel(result, cfg)

    macro = result["macro"]
    feats = result["macro_features"]
    return html.Div([
        _kpi_grid([
            ("Total sedentary (min)", f"{feats['total_sedentary_min']:.0f}"),
            ("Mean bout (min)", f"{feats['mean_bout_min']:.1f}"),
            ("Max bout (min)", f"{feats['max_bout_min']:.1f}"),
            ("# bouts", f"{feats['n_bouts']}"),
            ("Light (min)", f"{feats['light_min']:.0f}"),
            ("Moderate (min)", f"{feats['moderate_min']:.0f}"),
            ("Vigorous (min)", f"{feats['vigorous_min']:.0f}"),
            ("MVPA (min)", f"{feats['mvpa_min']:.0f}"),
        ]),
        dcc.Loading(dcc.Graph(figure=plots.actigraphy_double_plot(
            macro["full_enmo"], macro["full_timestamps"], macro["wear_mask"],
            start_dt, macro_size_sec=cfg.macro_size_sec,
        ))),
        dcc.Loading(dcc.Graph(figure=plots.sedentary_bout_histogram(
            feats["bout_lengths_min"]))),
    ])


@app.callback(
    Output("micro-content", "children"),
    Input("subject-dropdown", "value"),
    Input("store-config", "data"),
    Input("store-last-bout-idx", "data"),
    Input("start-date", "date"),
    Input("start-time", "value"),
)
def render_micro(subject_id, config, last_bout_idx, start_date, start_time):
    if subject_id is None:
        return html.Div("no subject selected")
    cfg = PreprocessConfig(**config)
    data, default_start, _ = _resolve_subject(subject_id)
    start_dt = _combine_date_time(start_date, start_time, default_start)
    result = get_result(subject_id, start_dt, config)
    if not result["valid"]:
        return _invalid_subject_panel(result, cfg)

    micro = result["micro"]
    gfeats = result["micro_features"]
    n_bouts = micro["data"].shape[0]
    if n_bouts == 0:
        return html.Div(
            "Subject passed wear-validity but the gait filter retained no bouts.",
            style={"color": "#a60", "padding": "0.8rem",
                   "background": "#fff8e1", "borderRadius": "4px"},
        )

    rms_xyz = np.stack([gfeats["rms_x"], gfeats["rms_y"], gfeats["rms_z"]], axis=1)
    p50, p90, p99 = np.percentile(np.linalg.norm(rms_xyz, axis=1), [50, 90, 99])

    if last_bout_idx is None:
        bout_idx = int(np.random.default_rng(0).integers(0, n_bouts))
    else:
        bout_idx = min(int(last_bout_idx), n_bouts - 1)

    return html.Div([
        _kpi_grid([
            ("# gait bouts", f"{gfeats['n_gait_bouts']}"),
            ("Mean dom freq (Hz)", f"{gfeats['dom_freq_mean']:.2f}"),
            ("Std dom freq (Hz)", f"{gfeats['dom_freq_std']:.2f}"),
            ("RMS p50/p90/p99 (g)", f"{p50:.2f} / {p90:.2f} / {p99:.2f}"),
        ]),
        dcc.Loading(dcc.Graph(
            id="micro-scatter",
            figure=plots.gait_scatter(gfeats["dom_freq_hz"], rms_xyz,
                                      micro["timestamps"], start_dt),
            config={"modeBarButtonsToAdd": ["lasso2d", "select2d"]},
        )),
        html.Div(
            f"Inspected bout #{bout_idx} — "
            f"dom freq {gfeats['dom_freq_hz'][bout_idx]:.2f} Hz, "
            f"t = {micro['timestamps'][bout_idx]:.0f} s",
            style={"padding": "0.4rem 0.6rem", "color": "#444",
                   "fontSize": "0.9em", "background": "#f5f5f5",
                   "borderRadius": "4px", "margin": "0.5rem 0"},
        ),
        html.Div([
            dcc.Graph(figure=plots.gait_freq_histogram(
                gfeats["dom_freq_hz"],
                gait_band=(cfg.gait_freq_low_hz, cfg.gait_freq_high_hz)),
                style={"flex": 1}),
            dcc.Graph(figure=plots.bout_signal_and_spectrum(
                micro["data"][bout_idx], micro["enmo"][bout_idx],
                fs=cfg.fs,
                gait_band=(cfg.gait_freq_low_hz, cfg.gait_freq_high_hz)),
                style={"flex": 1}),
        ], style={"display": "flex", "gap": "0.5rem"}),
    ])


@app.callback(
    Output("store-last-bout-idx", "data"),
    Input("micro-scatter", "selectedData"),
    Input("micro-scatter", "clickData"),
    prevent_initial_call=True,
)
def on_bout_selected(selected, clicked):
    """Persist the most recently lasso- or click-selected bout index."""
    pts = (selected or {}).get("points") or (clicked or {}).get("points") or []
    if not pts:
        return no_update
    return int(pts[-1]["pointIndex"])


if __name__ == "__main__":
    app.run(debug=True, port=8050)
