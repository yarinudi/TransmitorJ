# WHS Accelerometer Dashboard

Streamlit dashboard for visually validating the
[`whs_preprocessor.py`](whs_preprocessor.py) pipeline on raw 7-day triaxial
accelerometer data from the Women's Health Study (WHS) hip-worn ActiGraph
GT3X+ devices.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell:  .venv\Scripts\Activate.ps1
                                # bash/zsh:    source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If no patient files are found under `./data/raw/` the dashboard will boot
with a built-in synthetic 7-day subject so it is demoable out of the box.

## What you get

Four tabs sharing the same sidebar (patient selector + recording-start +
collapsible `PreprocessConfig` editor):

1. **Cohort overview** — table of all discovered patients (samples, duration,
   validity verdict) with a wear-hours-per-day bar.
2. **Raw signal** — three overlaid `Scattergl` X/Y/Z traces, optional ENMO
   overlay on a secondary y-axis, non-wear shading from `wear_mask`. Use the
   time-window slider to zoom; if the selected span is shorter than 1 hour
   the plot switches to full resolution.
3. **Macro view** — per-minute mean ENMO double plot (day × hour-of-day),
   sedentary-bout length histogram, intensity-minute KPIs.
4. **Micro view** — dominant-frequency vs RMS scatter (lasso-select a
   cluster), gait-frequency histogram, and a per-bout 2-second triaxial
   signal + FFT spectrum with the [1, 3] Hz gait band shaded.

## Tests

```bash
pytest tests/
```

## Documentation

See [`docs/dashboard.md`](docs/dashboard.md) for layout details, cache layers,
and how to plug in a new patient-file loader.
