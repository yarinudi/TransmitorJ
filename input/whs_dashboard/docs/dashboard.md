# Dashboard reference

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10+. Plotly handles ~50k WebGL points smoothly via `Scattergl`; the
raw page stride-downsamples to that target unless the selected zoom window is
shorter than 1 hour, at which point it loads the full-resolution slice.

## Project layout

```
whs_dashboard/
├── app.py                       # Streamlit entry point — four tabs
├── app_dash.py                  # Dash entry point — same four tabs, callback graph
├── whs_preprocessor.py          # frozen copy — DO NOT MODIFY
├── dashboard/
│   ├── _streamlit_compat.py     # @cache_data / @cache_resource shim
│   ├── data_loader.py           # discovery + raw loading (cached)
│   ├── pipeline_runner.py       # WHSPreprocessor + extractors, exposes
│   │                            #   run_pipeline_core (framework-agnostic)
│   ├── plots.py                 # pure Plotly figure builders (shared)
│   └── components.py            # Streamlit-only UI blocks
├── docs/dashboard.md            # this file
├── tests/
│   ├── test_plots.py            # smoke tests for every plot
│   └── test_dash_app.py         # layout + callback invocation tests
├── requirements.txt
└── README.md
```

## Streamlit vs Dash — what differs and what doesn't

| Layer | Streamlit (`app.py`) | Dash (`app_dash.py`) |
| ----- | -------------------- | -------------------- |
| **Plots** (`dashboard/plots.py`) | unchanged | unchanged |
| **Pipeline** (`pipeline_runner.run_pipeline_core`) | wrapped by `@cache_data` (Streamlit's cache) | wrapped by `@functools.lru_cache(maxsize=16)` keyed on `(subject_id, start_dt_iso, sorted config tuple)` |
| **Data load** (`data_loader.load_patient`) | wrapped by `@cache_data((path, mtime))` | wrapped by `@functools.lru_cache(maxsize=8)` |
| **Reactivity** | re-runs the whole script on every interaction | static `app.layout` + 9 `@app.callback` functions wiring the dependency graph explicitly |
| **State across pages** | `st.session_state` (e.g. `last_bout_idx`) | `dcc.Store` (`store-config`, `store-last-bout-idx`) |
| **Lasso bout selection** | `st.plotly_chart(..., on_select="rerun")` → `event["selection"]["points"]` | `dcc.Graph(..., id="micro-scatter")` → `Input("micro-scatter", "selectedData")` |
| **Cohort table** | `st.dataframe` | `dash_table.DataTable` (sortable, `row_selectable="single"`, click → sets dropdown) |
| **Streamlit decorators** | `streamlit.cache_data` / `cache_resource` | shim no-ops them so the modules import in a Streamlit-free env |

## Dash callback graph

```
subject-dropdown.value  ─┐
start-date.date          ├─►  render_raw     →  raw-graph.figure + caption
start-time.value         ├─►  render_macro   →  macro-content.children
raw-time-range.value     ├─►  render_micro   →  micro-content.children
raw-toggles.value        │
store-config.data        ┘

cfg-inputs.value (× 16) ──►  aggregate_config  →  store-config.data
cfg-reset.n_clicks      ──►  reset_config      →  cfg-inputs.value (× 16)
subject-dropdown.value  ──►  on_subject_change →  subject-info, raw slider, start-date, start-time
store-config.data       ──►  update_cohort    →  cohort-table.data + cohort-wear-bars
cohort-table.selected_rows ─►  cohort_row_clicked → subject-dropdown.value
micro-scatter.selectedData ─►  on_bout_selected   → store-last-bout-idx.data
store-last-bout-idx.data   ─►  (consumed by render_micro)
```

## Four views

| Tab | What it answers | Plots |
| --- | --------------- | ----- |
| **Cohort overview** | "Which subjects pass the wear-validity gate?" | per-subject wear-hours bar (red below `min_daily_wear_hours`); table of validity verdicts |
| **Raw signal** | "Does the raw stream look right, and are non-wear runs flagged where I expect?" | overlaid X/Y/Z `Scattergl` + optional ENMO + non-wear vrects |
| **Macro view** | "What does mortality / circadian feature input look like?" | day × hour-of-day mean-ENMO heatmap, sedentary bout-length histogram, intensity KPIs |
| **Micro view** | "Which gait bouts survive the FFT filter, and do they look like real walking?" | dom-freq × RMS scatter (lasso), dom-freq histogram, per-bout signal + spectrum, RMS percentile KPIs |

## Sidebar

* **Patient selector** — populated by `data_loader.discover_patients`, falls
  back to a built-in synthetic subject when the directory is empty.
* **Recording start (date + time)** — anchors the calendar-day / hour-of-day
  axes for circadian and double-plot computations.
* **Preprocessing config** — collapsible expander generated automatically
  from `PreprocessConfig`'s dataclass fields. Editing any value re-keys the
  `run_pipeline` cache so the pipeline re-runs but the raw-data cache stays
  warm (config changes do not touch raw loading).

## Cache layers

| Cache | Streamlit decorator | Dash equivalent | Key |
| ----- | ------------------- | --------------- | --- |
| Raw load | `@cache_data` (shim → `st.cache_data`) | `@functools.lru_cache(maxsize=8)` in `app_dash._load_data_cached` | `(path_str, mtime)` |
| Synthetic subject | `@cache_resource` (shim → `st.cache_resource`) | reuses the same `_load_data_cached` keyed on a sentinel path | n/a |
| Pipeline | `@cache_data` on `pipeline_runner.run_pipeline` | `@functools.lru_cache(maxsize=16)` in `app_dash._pipeline_cached` | `(subject_id, start_datetime_iso, sorted_config_tuple)` |

In both apps the raw numpy array is **not** in the cache key — Streamlit
skips it via the leading-underscore `_data` parameter; Dash never passes
it at all because the cached pipeline wrapper resolves data from
`subject_id` internally via `_resolve_subject`.

## Adding a new patient-file loader

`data_loader._LOADERS` is a `{suffix: callable}` registry. Decorate a new
function with `@register_loader(".your_ext")` returning
`(data: np.ndarray, start_datetime: datetime, subject_id: str)`:

```python
from datetime import datetime
from pathlib import Path
import numpy as np
from dashboard.data_loader import register_loader

@register_loader(".gt3x")
def load_gt3x(path: Path) -> tuple[np.ndarray, datetime, str]:
    # use pygt3x or actipy to decode the GT3X container
    data, start = decode_gt3x(path)         # your decoder
    return data.astype(np.float32), start, path.stem
```

Multi-suffix extensions (e.g. `.csv.gz`) are supported — the resolver picks
the longest registered suffix that matches the filename, so a `.csv.gz`
loader takes precedence over a `.gz` loader for the same file.

Optional sidecar JSON next to the data file provides the recording start:

```json
{ "start_datetime": "2024-01-01T08:00:00" }
```

Without a sidecar the loader falls back to the file's mtime, aligned to
midnight. The bundled `.csv.gz` loader instead parses the ActiGraph preamble
(rows 0–8) for the start date and time directly.

### Built-in loaders

| Suffix    | Reader | Subject-id source | Start-datetime source |
| --------- | ------ | ----------------- | --------------------- |
| `.npy`    | `np.load` (expects `(N, 3)`) | filename minus suffix | sidecar JSON → mtime |
| `.csv`    | `np.loadtxt` (3 columns, 1 header row) | filename minus suffix | sidecar JSON → mtime |
| `.csv.gz` | `pandas.read_csv` (skips 9 preamble rows, `header='infer'`) | filename minus suffixes | preamble `Start Date` + `Start Time` lines → mtime |

## Memory notes

* 7 days × 30 Hz = 18,144,000 samples. Float32 keeps a subject under ~220 MB.
* Plot functions never materialise the full stream into a DataFrame — they
  stride-index into the NumPy array directly (`data[::stride]`).
* Non-wear shading is computed once via a single `np.diff` over `wear_mask`,
  not per-window in Python.

## Things to verify visually

* **Non-wear shading** should align with flat overnight regions of the X/Y/Z
  traces. If they drift, check `nonwear_std_threshold_g` (default 0.013 g).
* **Macro heatmap** should show clear daytime ENMO above ~0.03 g and dark
  nighttime bands; the gray non-wear cells should cluster off-hours.
* **Gait scatter** should concentrate inside the configured gait band — any
  drift outside [1, 3] Hz means the filter is letting artefacts through and
  is worth investigating in the per-bout signal + spectrum panel.
