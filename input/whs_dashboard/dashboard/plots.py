"""Pure Plotly figure builders.

Every function takes arrays / dicts and returns a ``go.Figure``. No Streamlit
calls — this keeps the plots unit-testable and reusable from notebooks.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nonwear_segments(wear_mask: np.ndarray, macro_size_sec: int) -> list[tuple[float, float]]:
    """Return [(start_sec, end_sec), ...] runs where ``wear_mask`` is False.

    Computed via ``np.diff`` — no Python loop over windows.
    """
    if wear_mask.size == 0:
        return []
    not_wear = (~wear_mask).astype(np.int8)
    diff = np.diff(not_wear, prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(float(s * macro_size_sec), float(e * macro_size_sec)) for s, e in zip(starts, ends)]


def _hour_of_day(start_dt: datetime, ts_sec: np.ndarray) -> np.ndarray:
    epoch = start_dt.timestamp() + ts_sec.astype(np.float64)
    return (epoch % 86400) / 3600.0


# ---------------------------------------------------------------------------
# Wear-hours-per-day (used in cohort sparklines + invalid-subject fallback)
# ---------------------------------------------------------------------------

def wear_hours_bar(
    wear_hours_by_day: dict[int, float],
    min_daily_wear_hours: float | None = None,
    height: int = 220,
) -> go.Figure:
    if not wear_hours_by_day:
        fig = go.Figure()
        fig.add_annotation(text="No wear data", showarrow=False, x=0.5, y=0.5)
        fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10))
        return fig

    days = sorted(wear_hours_by_day.keys())
    hours = [wear_hours_by_day[d] for d in days]
    rel_days = [d - days[0] for d in days]
    colors = ["#2E86AB"] * len(days)
    if min_daily_wear_hours is not None:
        colors = ["#2E86AB" if h >= min_daily_wear_hours else "#C73E1D" for h in hours]

    fig = go.Figure(go.Bar(x=rel_days, y=hours, marker_color=colors, name="wear hours"))
    if min_daily_wear_hours is not None:
        fig.add_hline(y=min_daily_wear_hours, line_dash="dash", line_color="gray",
                      annotation_text=f"{min_daily_wear_hours:g} h threshold",
                      annotation_position="top right")
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=10, t=10, b=30),
        xaxis_title="Day of recording",
        yaxis_title="Wear hours",
        showlegend=False,
        bargap=0.2,
    )
    return fig


# ---------------------------------------------------------------------------
# Raw triaxial signal (Scattergl + optional ENMO + non-wear shading)
# ---------------------------------------------------------------------------

def raw_triaxial_figure(
    data: np.ndarray,
    fs: int,
    *,
    target_points: int = 50_000,
    enmo: np.ndarray | None = None,
    wear_mask: np.ndarray | None = None,
    macro_size_sec: int = 60,
    time_range_sec: tuple[float, float] | None = None,
    full_res_threshold_sec: float = 3600.0,
) -> go.Figure:
    """Three overlaid XYZ traces using Scattergl, with optional ENMO overlay.

    Downsampling: if a ``time_range_sec`` window shorter than
    ``full_res_threshold_sec`` is given, render the full-resolution slice;
    otherwise stride-downsample so the overview has roughly ``target_points``.
    """
    n = data.shape[0]
    total_sec = n / fs

    if time_range_sec is not None:
        t0, t1 = time_range_sec
        i0 = max(0, int(t0 * fs))
        i1 = min(n, int(t1 * fs))
    else:
        i0, i1 = 0, n
        t0, t1 = 0.0, total_sec

    span_sec = max(0.0, t1 - t0)
    if span_sec <= full_res_threshold_sec:
        stride = 1
    else:
        view_n = i1 - i0
        stride = max(1, view_n // target_points)

    sl = slice(i0, i1, stride)
    t = np.arange(i0, i1, stride, dtype=np.float64) / fs

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for k, name, color in [(0, "x", "#1f77b4"), (1, "y", "#ff7f0e"), (2, "z", "#2ca02c")]:
        fig.add_trace(
            go.Scattergl(x=t, y=data[sl, k], mode="lines", name=name,
                         line=dict(width=1, color=color), opacity=0.8),
            secondary_y=False,
        )

    if enmo is not None:
        fig.add_trace(
            go.Scattergl(x=t, y=enmo[sl], mode="lines", name="ENMO",
                         line=dict(width=1.2, color="#8B0000")),
            secondary_y=True,
        )

    if wear_mask is not None:
        for s, e in nonwear_segments(wear_mask, macro_size_sec):
            if e < t0 or s > t1:
                continue
            fig.add_vrect(x0=max(s, t0), x1=min(e, t1),
                          fillcolor="rgba(120,120,120,0.18)", line_width=0,
                          annotation_text="non-wear", annotation_position="top left",
                          annotation=dict(font_size=10))

    fig.update_layout(
        height=440,
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title="Time (s from recording start)",
                   rangeslider=dict(visible=True, thickness=0.06)),
        title=dict(text=f"Raw triaxial (stride={stride}, span={span_sec / 3600:.2f} h)",
                   x=0.01, font_size=12),
    )
    fig.update_yaxes(title_text="Acceleration (g)", secondary_y=False)
    if enmo is not None:
        fig.update_yaxes(title_text="ENMO (g)", secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Actigraphy double-plot heatmap (day x hour-of-day)
# ---------------------------------------------------------------------------

def actigraphy_double_plot(
    full_enmo: np.ndarray,
    full_timestamps: np.ndarray,
    wear_mask: np.ndarray,
    start_datetime: datetime,
    macro_size_sec: int = 60,
) -> go.Figure:
    """Per-minute mean ENMO laid out as day-of-recording × hour-of-day."""
    if full_enmo.shape[0] == 0:
        fig = go.Figure()
        fig.add_annotation(text="No macro data", showarrow=False, x=0.5, y=0.5)
        return fig

    mean_enmo = full_enmo.mean(axis=1)
    epoch = start_datetime.timestamp() + full_timestamps.astype(np.float64)
    day_idx = (epoch // 86400).astype(np.int64)
    day_idx -= day_idx.min()
    sec_of_day = (epoch % 86400)
    bin_of_day = (sec_of_day // macro_size_sec).astype(np.int64)

    n_days = int(day_idx.max()) + 1
    n_bins = 86400 // macro_size_sec
    grid = np.full((n_days, n_bins), np.nan, dtype=np.float64)
    grid[day_idx, bin_of_day] = mean_enmo

    nonwear_grid = np.zeros((n_days, n_bins), dtype=np.float64)
    nonwear_grid[day_idx[~wear_mask], bin_of_day[~wear_mask]] = 1.0
    nonwear_grid[nonwear_grid == 0.0] = np.nan

    hours = np.arange(n_bins) * (macro_size_sec / 3600.0)
    days = np.arange(n_days)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=grid, x=hours, y=days,
        colorscale="Viridis", colorbar=dict(title="ENMO (g)"),
        zmin=0.0, zmax=float(np.nanpercentile(mean_enmo, 99)) if np.isfinite(mean_enmo).any() else 0.1,
        hovertemplate="day %{y}, hour %{x:.2f}<br>ENMO=%{z:.3f}<extra></extra>",
        name="ENMO",
    ))
    fig.add_trace(go.Heatmap(
        z=nonwear_grid, x=hours, y=days,
        colorscale=[[0.0, "rgba(150,150,150,0.55)"], [1.0, "rgba(150,150,150,0.55)"]],
        showscale=False, hoverinfo="skip", name="non-wear",
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=60, r=20, t=40, b=40),
        xaxis=dict(title="Hour of day", dtick=2, range=[0, 24]),
        yaxis=dict(title="Day of recording", autorange="reversed", dtick=1),
        title=dict(text="Per-minute mean ENMO (non-wear shaded gray)",
                   x=0.01, font_size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# Sedentary-bout histogram
# ---------------------------------------------------------------------------

def sedentary_bout_histogram(bout_lengths_min: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if bout_lengths_min.size == 0:
        fig.add_annotation(text="No sedentary bouts", showarrow=False, x=0.5, y=0.5)
        fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
        return fig
    fig.add_trace(go.Histogram(x=bout_lengths_min, nbinsx=40, marker_color="#5D4E75",
                               name="sedentary bouts"))
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis_title="Bout length (minutes)",
        yaxis_title="Count",
        title=dict(text="Sedentary bout length distribution", x=0.01, font_size=12),
        bargap=0.05,
    )
    return fig


# ---------------------------------------------------------------------------
# Gait scatter (dom freq vs RMS, colored by time-of-day)
# ---------------------------------------------------------------------------

def gait_scatter(
    dom_freq_hz: np.ndarray,
    rms_xyz: np.ndarray,
    timestamps_sec: np.ndarray,
    start_datetime: datetime,
) -> go.Figure:
    """Dominant gait frequency vs vector-norm RMS, colored by hour-of-day."""
    fig = go.Figure()
    if dom_freq_hz.size == 0:
        fig.add_annotation(text="No gait bouts retained", showarrow=False, x=0.5, y=0.5)
        fig.update_layout(height=380, margin=dict(l=40, r=20, t=30, b=40))
        return fig

    rms_norm = np.linalg.norm(rms_xyz, axis=1)
    hod = _hour_of_day(start_datetime, timestamps_sec)

    fig.add_trace(go.Scattergl(
        x=dom_freq_hz, y=rms_norm, mode="markers",
        marker=dict(
            color=hod, colorscale="Twilight", cmin=0, cmax=24,
            colorbar=dict(title="Hour of day"),
            size=6, opacity=0.65,
            line=dict(width=0),
        ),
        customdata=np.stack([timestamps_sec, hod], axis=1),
        hovertemplate=(
            "dom freq: %{x:.2f} Hz<br>"
            "RMS: %{y:.3f} g<br>"
            "t: %{customdata[0]:.0f} s<br>"
            "hour-of-day: %{customdata[1]:.2f}<extra></extra>"
        ),
        name="gait bout",
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=60, r=20, t=30, b=40),
        xaxis_title="Dominant frequency (Hz)",
        yaxis_title="Vector-norm RMS (g)",
        title=dict(text="Gait bouts — lasso-select a cluster",
                   x=0.01, font_size=12),
        dragmode="lasso",
    )
    return fig


# ---------------------------------------------------------------------------
# Gait-frequency histogram
# ---------------------------------------------------------------------------

def gait_freq_histogram(dom_freq_hz: np.ndarray, gait_band: tuple[float, float] = (1.0, 3.0)) -> go.Figure:
    fig = go.Figure()
    if dom_freq_hz.size == 0:
        fig.add_annotation(text="No gait bouts retained", showarrow=False, x=0.5, y=0.5)
        fig.update_layout(height=300, margin=dict(l=40, r=20, t=30, b=40))
        return fig
    fig.add_trace(go.Histogram(x=dom_freq_hz, nbinsx=40, marker_color="#2E86AB",
                               name="dominant freq"))
    fig.add_vrect(x0=gait_band[0], x1=gait_band[1],
                  fillcolor="rgba(46,134,171,0.08)", line_width=0)
    fig.update_layout(
        height=300,
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis_title="Dominant frequency (Hz)",
        yaxis_title="Count",
        title=dict(text="Dominant gait frequencies", x=0.01, font_size=12),
        bargap=0.05,
    )
    return fig


# ---------------------------------------------------------------------------
# Single-bout signal + spectrum
# ---------------------------------------------------------------------------

def bout_signal_and_spectrum(
    bout_data: np.ndarray,
    bout_enmo: np.ndarray,
    fs: int,
    gait_band: tuple[float, float] = (1.0, 3.0),
) -> go.Figure:
    """Side-by-side time-domain triaxial + FFT magnitude spectrum of one bout."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=("2-s bout (triaxial + ENMO)", "FFT magnitude (ENMO)"))
    if bout_data.size == 0:
        fig.add_annotation(text="No bout selected", showarrow=False, x=0.5, y=0.5)
        return fig

    L = bout_data.shape[0]
    t = np.arange(L) / fs

    for k, name, color in [(0, "x", "#1f77b4"), (1, "y", "#ff7f0e"), (2, "z", "#2ca02c")]:
        fig.add_trace(
            go.Scatter(x=t, y=bout_data[:, k], mode="lines", name=name,
                       line=dict(width=1.2, color=color)),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(x=t, y=bout_enmo, mode="lines", name="ENMO",
                   line=dict(width=1.5, color="#8B0000")),
        row=1, col=1,
    )

    spectra = np.abs(np.fft.rfft(bout_enmo))
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    fig.add_trace(
        go.Scatter(x=freqs, y=spectra, mode="lines+markers", name="|FFT|",
                   line=dict(width=1.5, color="#5D4E75")),
        row=1, col=2,
    )
    fig.add_vrect(x0=gait_band[0], x1=gait_band[1],
                  fillcolor="rgba(46,134,171,0.18)", line_width=0,
                  row=1, col=2)

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration (g)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2, range=[0, fs / 2])
    fig.update_yaxes(title_text="Magnitude", row=1, col=2)
    fig.update_layout(
        height=360,
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.10),
    )
    return fig
