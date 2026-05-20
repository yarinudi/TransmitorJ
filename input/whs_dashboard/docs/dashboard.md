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
├── whs_preprocessor.py          # frozen copy — DO NOT MODIFY
├── dashboard/
│   ├── data_loader.py           # discovery + raw loading (cached)
│   ├── pipeline_runner.py       # WHSPreprocessor + extractors (cached)
│   ├── plots.py                 # pure Plotly figure builders
│   └── components.py            # sidebar, KPIs, cohort table, config editor
├── docs/dashboard.md            # this file
├── tests/test_plots.py          # smoke tests for every plot
├── requirements.txt
└── README.md
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

| Cache | Decorator | Key | Why |
| ----- | --------- | --- | --- |
| Raw load (`load_patient`) | `@st.cache_data` | `(path_str, mtime)` | Reload only when the file changes on disk. |
| Synthetic subject (`get_synthetic_patient`) | `@st.cache_resource` | none | Built once per Streamlit session; ~216 MB float32 per subject. |
| Pipeline (`run_pipeline`) | `@st.cache_data` | `(subject_id, start_datetime, config_dict)` | The raw array is passed with a leading underscore (`_data`) so Streamlit's hasher skips it — keeping the cache key small and fast. |

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
