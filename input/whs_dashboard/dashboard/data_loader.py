"""Patient discovery and raw-data loading.

The loader registry pattern lets us add new file formats (e.g. `.gt3x`) by
registering a single function — see ``register_loader`` at the bottom.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# Patient record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatientRecord:
    subject_id: str
    path: Path
    suffix: str
    size_bytes: int
    mtime: float


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

LoaderFn = Callable[[Path], "tuple[np.ndarray, datetime, str]"]
_LOADERS: dict[str, LoaderFn] = {}


def register_loader(suffix: str) -> Callable[[LoaderFn], LoaderFn]:
    def decorator(fn: LoaderFn) -> LoaderFn:
        _LOADERS[suffix.lower()] = fn
        return fn
    return decorator


def supported_suffixes() -> list[str]:
    return sorted(_LOADERS.keys())


def _resolve_loader(path: Path) -> tuple[str, LoaderFn] | tuple[None, None]:
    """Pick the loader matching the longest registered suffix (so .csv.gz wins over .gz)."""
    name = path.name.lower()
    for suffix in sorted(_LOADERS, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix, _LOADERS[suffix]
    return None, None


def _derive_subject_id(path: Path) -> str:
    """Strip every registered suffix from the filename to get a clean subject id."""
    name = path.name
    suffix, _ = _resolve_loader(path)
    if suffix is not None:
        name = name[: -len(suffix)]
    return name


# ---------------------------------------------------------------------------
# Built-in loaders
# ---------------------------------------------------------------------------

@register_loader(".npy")
def _load_npy(path: Path) -> tuple[np.ndarray, datetime, str]:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path.name}: expected (N, 3) array, got {arr.shape}.")
    sidecar = path.with_suffix(".json")
    start = _read_sidecar_start(sidecar) or _mtime_to_recording_start(path)
    return arr.astype(np.float32, copy=False), start, _derive_subject_id(path)


@register_loader(".csv")
def _load_csv(path: Path) -> tuple[np.ndarray, datetime, str]:
    arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path.name}: expected 3 columns (x,y,z), got shape {arr.shape}.")
    sidecar = path.with_suffix(".json")
    start = _read_sidecar_start(sidecar) or _mtime_to_recording_start(path)
    return arr, start, _derive_subject_id(path)


# ActiGraph GT3X+ CSV export, gzipped.
# Preamble layout (matches the WHS export format):
#   row 0..8 : metadata (serial number, start time/date, epoch, battery, ...)
#   row 9    : column header  "Accelerometer X","Accelerometer Y","Accelerometer Z"
#   row 10.. : triaxial g-units, one row per sample
_ACTIGRAPH_PREAMBLE_LINES = 9


@register_loader(".csv.gz")
def _load_actigraph_csv_gz(path: Path) -> tuple[np.ndarray, datetime, str]:
    import pandas as pd
    start = _parse_actigraph_preamble(path) or _mtime_to_recording_start(path)
    df = pd.read_csv(
        path,
        compression="gzip",
        header="infer",
        skiprows=list(range(_ACTIGRAPH_PREAMBLE_LINES)),
        dtype=np.float32,
    )
    if df.shape[1] < 3:
        raise ValueError(f"{path.name}: expected 3 axis columns, got {df.shape[1]}.")
    arr = df.iloc[:, :3].to_numpy(dtype=np.float32, copy=False)
    return arr, start, _derive_subject_id(path)


def _parse_actigraph_preamble(path: Path) -> datetime | None:
    """Read just the preamble bytes and pull out start date + start time.

    Tolerates the WHS-anonymized 'Fake Start Time' / 'Fake Start Valid Date'
    line variants — the dates are still real-shape ISO strings.
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline() for _ in range(_ACTIGRAPH_PREAMBLE_LINES)]
    except OSError:
        return None
    blob = "\n".join(lines)
    date_match = re.search(r"Start\s+(?:Valid\s+)?Date\s+([0-9]{4}-[0-9]{2}-[0-9]{2})", blob)
    time_match = re.search(r"Start\s+Time\s+([0-9]{2}:[0-9]{2}:[0-9]{2})", blob)
    if not (date_match and time_match):
        return None
    try:
        return datetime.fromisoformat(f"{date_match.group(1)}T{time_match.group(1)}")
    except ValueError:
        return None


def _read_sidecar_start(sidecar: Path) -> datetime | None:
    if not sidecar.exists():
        return None
    import json
    try:
        meta = json.loads(sidecar.read_text())
        return datetime.fromisoformat(meta["start_datetime"])
    except Exception:
        return None


def _mtime_to_recording_start(path: Path) -> datetime:
    """Fallback when no sidecar JSON is present: use file mtime, midnight-aligned."""
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_patients(directory: str | Path = "./example_data/") -> list[PatientRecord]:
    """Scan a directory for files matching any registered loader.

    Returns a list sorted by subject_id. Missing directory -> empty list.
    Multi-suffix loaders (e.g. ``.csv.gz``) win over single-suffix matches
    so the same file isn't recorded twice.
    """
    root = Path(directory)
    if not root.exists():
        return []
    seen: dict[Path, PatientRecord] = {}
    for suffix in sorted(_LOADERS, key=len, reverse=True):
        for p in sorted(root.glob(f"*{suffix}")):
            if p in seen:
                continue
            stat = p.stat()
            seen[p] = PatientRecord(
                subject_id=_derive_subject_id(p),
                path=p,
                suffix=suffix,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
            )
    records = sorted(seen.values(), key=lambda r: r.subject_id)
    return records


# ---------------------------------------------------------------------------
# Cached raw loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading raw accelerometer data...")
def load_patient(path_str: str, mtime: float) -> tuple[np.ndarray, datetime, str]:
    """Load a patient file. Cache key is (path_str, mtime) so file edits invalidate."""
    del mtime  # used only for cache invalidation
    path = Path(path_str)
    _, loader = _resolve_loader(path)
    if loader is None:
        raise ValueError(f"No loader registered for {path.name!r}.")
    return loader(path)


# ---------------------------------------------------------------------------
# Synthetic subject — same shape as the smoke-test block in whs_preprocessor.py
# ---------------------------------------------------------------------------

def generate_synthetic_subject(
    seed: int = 0,
    fs: int = 30,
    n_days: int = 7,
    subject_id: str = "SYNTH001",
    start_datetime: datetime | None = None,
) -> tuple[np.ndarray, datetime, str]:
    """Build a synthetic 7-day stream: flat at night, walking bouts during the day.

    Mirrors the WHS off-at-night protocol used in the preprocessor's smoke test.
    """
    rng = np.random.default_rng(seed)
    N = fs * 60 * 60 * 24 * n_days
    data = np.zeros((N, 3), dtype=np.float32)
    data[:, 2] = -1.0  # gravity on Z

    for d in range(n_days):
        day_start = d * 86400 * fs + 7 * 3600 * fs
        day_end = d * 86400 * fs + 23 * 3600 * fs
        data[day_start:day_end] += rng.normal(0, 0.03, (day_end - day_start, 3)).astype(np.float32)
        for _ in range(20):
            t0 = rng.integers(day_start, day_end - 60 * fs)
            t = np.arange(60 * fs) / fs
            walk = (0.4 * np.sin(2 * np.pi * 2.0 * t)).astype(np.float32)
            data[t0:t0 + 60 * fs, 2] += walk

    start = start_datetime or datetime(2024, 1, 1, 0, 0, 0)
    return data, start, subject_id


@st.cache_resource(show_spinner="Generating synthetic subject...")
def get_synthetic_patient() -> tuple[np.ndarray, datetime, str]:
    """Streamlit-cached wrapper so the synthetic stream is built once per session."""
    return generate_synthetic_subject()
