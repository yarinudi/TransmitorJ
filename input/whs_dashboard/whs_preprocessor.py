"""
whs_preprocessor.py
===================

Production-ready preprocessing pipeline for raw 7-day triaxial accelerometer
data from the Women's Health Study (WHS) hip-worn ActiGraph GT3X+ devices.

Implements a dual-window framework:

    * Macro windows (60 s, non-overlapping)  -> Mortality / circadian features
    * Micro windows (2 s,  50% overlap)      -> Parkinson's / Fall-risk gait features

Cleaning constraints (all vectorised with NumPy):

    1. ENMO = max(0, ||a|| - 1)                       (no global normalisation)
    2. Choi non-wear: drop runs of >= 90 min with     [Choi et al. 2011]
       per-window ENMO std < 0.013 g
    3. Subject filter: keep only if >= 4 days with    [WHS/UKB convention]
       >= 10 h of valid daytime wear
    4. Gait bout filter on micro windows:
            std(ENMO) > 0.05 g  AND  1.0 Hz <= dom_freq <= 3.0 Hz

Memory note
-----------
7 days @ 30 Hz = 18,144,000 samples. Pass float32 in -- you keep ~216 MB
per subject instead of ~432 MB with float64, and `sliding_window_view` /
`reshape` give views, not copies, so the peak footprint stays close to that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    """All thresholds in one place so they are easy to audit / change."""

    # Sampling
    fs: int = 30                           # Hz (WHS resampled rate)

    # Window geometry
    macro_size_sec: int = 60               # non-overlapping
    micro_size_sec: int = 2
    micro_overlap_pct: float = 0.5

    # Choi non-wear filter (per-window ENMO std threshold + run length)
    nonwear_std_threshold_g: float = 0.013
    nonwear_duration_min: int = 90

    # Subject-level wear validity
    min_valid_days: int = 4
    min_daily_wear_hours: float = 10.0
    daytime_start_hour: int = 7            # 07:00 .. 23:00 counted as "daytime"
    daytime_end_hour: int = 23

    # Gait-bout verification on micro windows
    gait_std_threshold_g: float = 0.05
    gait_freq_low_hz: float = 1.0
    gait_freq_high_hz: float = 3.0


# Zero-padding target for the gait-band FFT. A 2-s × 30-Hz window only holds
# 60 samples -> a 0.5 Hz native bin grid which would snap every dominant
# frequency onto {1.0, 1.5, 2.0, 2.5, 3.0} Hz. Padding to 256 buys
# 30/256 ≈ 0.117 Hz resolution, enough to distinguish typical step cadences.
GAIT_FFT_LENGTH = 256


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class WHSPreprocessor:
    """Stateless, vectorised dual-window preprocessor for WHS hip data.

    Typical usage
    -------------
    >>> pre = WHSPreprocessor()
    >>> result = pre.process(
    ...     data=raw_xyz_array,                # shape (N, 3), raw g-units
    ...     start_datetime=recording_start,    # datetime, anchors clock-time
    ...     subject_id="ZU000001",
    ... )
    >>> if result["valid"]:
    ...     macro = result["macro"]            # dict with data/enmo/timestamps
    ...     micro = result["micro"]            # dict with data/timestamps
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.cfg = config or PreprocessConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        c = self.cfg
        if not (0.0 <= c.micro_overlap_pct < 1.0):
            raise ValueError("micro_overlap_pct must be in [0, 1).")
        if c.gait_freq_high_hz > c.fs / 2:
            raise ValueError(
                f"gait_freq_high_hz ({c.gait_freq_high_hz}) exceeds Nyquist "
                f"({c.fs / 2})."
            )
        if (c.nonwear_duration_min * 60) % c.macro_size_sec != 0:
            raise ValueError(
                "nonwear_duration_min*60 must be divisible by macro_size_sec."
            )

    # ------------------------------------------------------------------ #
    # 1. ENMO
    # ------------------------------------------------------------------ #
    @staticmethod
    def compute_enmo(data: np.ndarray) -> np.ndarray:
        """Euclidean Norm Minus One, clipped at zero.

        ENMO_t = max(0, sqrt(x_t^2 + y_t^2 + z_t^2) - 1.0)

        Robust to orientation, removes the static gravity floor without
        touching dynamic g-force amplitude (so 0.05 g still *means* 0.05 g).
        """
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) array, got shape {data.shape}.")
        vm = np.linalg.norm(data, axis=1)
        return np.clip(vm - 1.0, 0.0, None)

    # ------------------------------------------------------------------ #
    # 2. Macro window extraction (non-overlapping, reshape is a view)
    # ------------------------------------------------------------------ #
    def extract_macro_windows(
        self,
        data: np.ndarray,
        enmo: np.ndarray,
        start_offset_sec: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Non-overlapping 60-s windows.

        Returns
        -------
        macro_data       : (W, macro_size, 3)
        macro_enmo       : (W, macro_size)
        macro_timestamps : (W,)   seconds since recording start
                                   (computed BEFORE any windows are dropped)
        """
        fs = self.cfg.fs
        macro_size = int(self.cfg.macro_size_sec * fs)
        n_windows = data.shape[0] // macro_size
        if n_windows == 0:
            return (
                np.empty((0, macro_size, 3), dtype=data.dtype),
                np.empty((0, macro_size), dtype=enmo.dtype),
                np.empty((0,), dtype=np.float64),
            )

        trim = n_windows * macro_size
        # Reshape is a true view -- no copy, no extra memory.
        macro_data = data[:trim].reshape(n_windows, macro_size, 3)
        macro_enmo = enmo[:trim].reshape(n_windows, macro_size)

        # Structural timestamps -- chronology preserved even after dropping.
        macro_timestamps = (
            start_offset_sec
            + np.arange(n_windows, dtype=np.float64) * self.cfg.macro_size_sec
        )
        return macro_data, macro_enmo, macro_timestamps

    # ------------------------------------------------------------------ #
    # 3. Micro window extraction (overlapping)
    # ------------------------------------------------------------------ #
    def extract_micro_windows(
        self,
        data: np.ndarray,
        enmo: np.ndarray,
        start_offset_sec: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2-s windows with 50 % overlap.

        sliding_window_view returns a view -- only the FFT / std reductions
        below materialise memory, and they output (n_windows, ...) which is
        tiny relative to the raw timeline.
        """
        fs = self.cfg.fs
        micro_size = int(self.cfg.micro_size_sec * fs)
        step = max(1, int(micro_size * (1.0 - self.cfg.micro_overlap_pct)))

        if data.shape[0] < micro_size:
            return (
                np.empty((0, micro_size, 3), dtype=data.dtype),
                np.empty((0, micro_size), dtype=enmo.dtype),
                np.empty((0,), dtype=np.float64),
            )

        # sliding_window_view over axis=0 -> shape (N-L+1, 3, L); subsample
        # by `step`, then transpose so the time-within-window axis is in
        # the middle: (n_windows, L, 3).
        sw = sliding_window_view(data, window_shape=micro_size, axis=0)[::step]
        micro_data = np.transpose(sw, (0, 2, 1))

        micro_enmo = sliding_window_view(
            enmo, window_shape=micro_size, axis=0
        )[::step]

        n_windows = micro_data.shape[0]
        micro_timestamps = (
            start_offset_sec
            + np.arange(n_windows, dtype=np.float64) * step / fs
        )
        return micro_data, micro_enmo, micro_timestamps

    # ------------------------------------------------------------------ #
    # 4. Choi non-wear filter (vectorised, no Python loop over time)
    # ------------------------------------------------------------------ #
    def apply_choi_nonwear(self, macro_enmo: np.ndarray) -> np.ndarray:
        """Mark non-wear macro windows.

        A run of >= `nonwear_duration_min` minutes where per-window
        std(ENMO) < `nonwear_std_threshold_g` is flagged as non-wear.

        Returns
        -------
        wear_mask : (W,) bool   True  -> wear, keep
                                False -> non-wear, drop
        """
        if macro_enmo.shape[0] == 0:
            return np.zeros(0, dtype=bool)

        macro_stds = macro_enmo.std(axis=1)
        is_still = macro_stds < self.cfg.nonwear_std_threshold_g

        # How many macro windows correspond to the non-wear duration?
        n_required = int(
            self.cfg.nonwear_duration_min * 60 / self.cfg.macro_size_sec
        )
        if is_still.size < n_required:
            return np.ones(is_still.size, dtype=bool)

        # Convolution trick: at position i, run_sums[i] = sum(is_still[i:i+n_required]).
        kernel = np.ones(n_required, dtype=np.int32)
        run_sums = np.convolve(is_still.astype(np.int32), kernel, mode="valid")

        # Mark start positions of full stillness runs.
        run_starts_bool = np.zeros(is_still.size, dtype=np.int32)
        run_starts_bool[: run_sums.size] = (run_sums == n_required).astype(
            np.int32
        )

        # Propagate each run-start forward by `n_required` windows
        # via a second convolution (this is equivalent to a rolling OR).
        nonwear_int = np.convolve(run_starts_bool, kernel, mode="full")[
            : is_still.size
        ]
        return ~(nonwear_int > 0)

    # ------------------------------------------------------------------ #
    # 5. Subject-level validity (>= 4 days with >= 10 h wear)
    # ------------------------------------------------------------------ #
    def validate_subject(
        self,
        macro_timestamps: np.ndarray,
        wear_mask: np.ndarray,
        start_datetime: datetime | None = None,
    ) -> tuple[bool, dict[int, float]]:
        """Aggregate wear by calendar day and count valid days.

        If `start_datetime` is given we bucket by real calendar day and
        restrict to the daytime window so sleep-off-wrist periods don't
        unfairly penalise the subject. Otherwise we fall back to
        recording-day indexing (24-h blocks from t=0).
        """
        if macro_timestamps.size == 0:
            return False, {}

        macro_dt_sec = self.cfg.macro_size_sec

        if start_datetime is not None:
            epoch_sec = (
                start_datetime.timestamp()
                + macro_timestamps.astype(np.float64)
            )
            day_idx = (epoch_sec // 86400).astype(np.int64)
            hour_of_day = (epoch_sec % 86400) / 3600.0
            daytime_mask = (
                (hour_of_day >= self.cfg.daytime_start_hour)
                & (hour_of_day < self.cfg.daytime_end_hour)
            )
        else:
            day_idx = (macro_timestamps // 86400).astype(np.int64)
            daytime_mask = np.ones_like(macro_timestamps, dtype=bool)

        eligible = wear_mask & daytime_mask
        days, inverse = np.unique(day_idx, return_inverse=True)
        wear_per_day_sec = (
            np.bincount(inverse, weights=eligible.astype(np.float64))
            * macro_dt_sec
        )

        wear_hours_by_day = {
            int(d): float(s / 3600.0) for d, s in zip(days, wear_per_day_sec)
        }
        n_valid_days = int(
            np.sum(wear_per_day_sec >= self.cfg.min_daily_wear_hours * 3600)
        )
        is_valid = n_valid_days >= self.cfg.min_valid_days
        return is_valid, wear_hours_by_day

    # ------------------------------------------------------------------ #
    # 6. Map micro -> macro wear (cheap index arithmetic, no np.isin)
    # ------------------------------------------------------------------ #
    def _micro_in_wear(
        self,
        micro_timestamps: np.ndarray,
        macro_timestamps: np.ndarray,
        wear_mask: np.ndarray,
    ) -> np.ndarray:
        if macro_timestamps.size == 0 or micro_timestamps.size == 0:
            return np.zeros(micro_timestamps.size, dtype=bool)
        macro_start = macro_timestamps[0]
        macro_dt = self.cfg.macro_size_sec
        idx = np.floor((micro_timestamps - macro_start) / macro_dt).astype(
            np.int64
        )
        valid = (idx >= 0) & (idx < wear_mask.size)
        out = np.zeros(micro_timestamps.size, dtype=bool)
        out[valid] = wear_mask[idx[valid]]
        return out

    # ------------------------------------------------------------------ #
    # 7. Gait-bout verification (batched FFT, no per-window loop)
    # ------------------------------------------------------------------ #
    def verify_gait_bouts(self, micro_enmo: np.ndarray) -> np.ndarray:
        """Vectorised FFT-based gait filter.

        Step 1: std(ENMO) > 0.05 g   -> drop sedentary / quiet windows.
        Step 2: zero-padded FFT to GAIT_FFT_LENGTH (~0.117 Hz bins at fs=30)
                so step cadences off the 0.5 Hz native grid are still
                resolved. Then require: in-band peak >= out-of-band peak
                (both after zeroing DC). This rejects windows where a
                strong out-of-band tone (handwashing tremor, vehicle
                vibration ~5-10 Hz) coexists with weaker in-band content.
        """
        n_windows = micro_enmo.shape[0]
        gait_mask = np.zeros(n_windows, dtype=bool)
        if n_windows == 0:
            return gait_mask

        # Cheap pre-filter -- cuts FFT cost ~10-100x on real data.
        stds = micro_enmo.std(axis=1)
        active = stds > self.cfg.gait_std_threshold_g
        if not active.any():
            return gait_mask

        active_idx = np.flatnonzero(active)
        active_enmo = micro_enmo[active_idx]                                 # (A, L)

        # ENMO is half-wave rectified, so DC carries the mean amplitude rather
        # than any oscillatory information. Subtract the per-window mean before
        # the rFFT — just zeroing the f=0 bin after the fact is not enough once
        # we zero-pad, because the DC sinc leaks into adjacent (out-of-band)
        # bins (~0.117 Hz) and can dominate the gate.
        active_enmo = active_enmo - active_enmo.mean(axis=1, keepdims=True)

        # Batched zero-padded rFFT -- one allocation, one call.
        spectra = np.abs(np.fft.rfft(active_enmo, n=GAIT_FFT_LENGTH, axis=1))  # (A, K)
        freqs = np.fft.rfftfreq(GAIT_FFT_LENGTH, d=1.0 / self.cfg.fs)          # (K,)

        # Belt-and-braces: f=0 bin is already ~0 after demean, set it exactly so
        # downstream argmax / max comparisons are robust to float-precision noise.
        spectra[:, 0] = 0.0

        in_band = (
            (freqs >= self.cfg.gait_freq_low_hz)
            & (freqs <= self.cfg.gait_freq_high_hz)
        )
        in_band_peak = np.where(in_band[None, :], spectra, -np.inf).max(axis=1)
        out_band_peak = np.where(~in_band[None, :], spectra, -np.inf).max(axis=1)
        passes = in_band_peak >= out_band_peak

        gait_mask[active_idx[passes]] = True
        return gait_mask

    # ------------------------------------------------------------------ #
    # 8. End-to-end pipeline
    # ------------------------------------------------------------------ #
    def process(
        self,
        data: np.ndarray,
        start_datetime: datetime | None = None,
        subject_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the full preprocessing pipeline for one subject.

        Parameters
        ----------
        data           : (N, 3) raw triaxial accelerometer in g-units.
                         Pass float32 to halve memory.
        start_datetime : recording start (used for circadian + day-bucketing).
        subject_id     : optional id passed through to the result.
        """
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)

        enmo = self.compute_enmo(data)

        # --- Macro ----------------------------------------------------- #
        macro_data, macro_enmo, macro_ts = self.extract_macro_windows(
            data, enmo
        )
        wear_mask = self.apply_choi_nonwear(macro_enmo)

        is_valid, wear_hours_by_day = self.validate_subject(
            macro_ts, wear_mask, start_datetime=start_datetime
        )
        if not is_valid:
            return {
                "valid": False,
                "subject_id": subject_id,
                "reason": "fails 4-days x 10-hours wear criterion",
                "wear_hours_by_day": wear_hours_by_day,
            }

        # --- Micro ----------------------------------------------------- #
        micro_data, micro_enmo, micro_ts = self.extract_micro_windows(
            data, enmo
        )
        in_wear = self._micro_in_wear(micro_ts, macro_ts, wear_mask)

        # Verify gait only on wear-time micro windows (cheaper FFT batch).
        gait_mask_local = self.verify_gait_bouts(micro_enmo[in_wear])
        wear_indices = np.flatnonzero(in_wear)
        final_micro_mask = np.zeros(micro_ts.size, dtype=bool)
        final_micro_mask[wear_indices[gait_mask_local]] = True

        return {
            "valid": True,
            "subject_id": subject_id,
            "start_datetime": start_datetime,
            "wear_hours_by_day": wear_hours_by_day,
            "macro": {
                # Wear-only views (most features want these):
                "data": macro_data[wear_mask],
                "enmo": macro_enmo[wear_mask],
                "timestamps": macro_ts[wear_mask],
                # Full timeline kept so circadian features can see structure:
                "full_enmo": macro_enmo,
                "full_timestamps": macro_ts,
                "wear_mask": wear_mask,
            },
            "micro": {
                "data": micro_data[final_micro_mask],
                "enmo": micro_enmo[final_micro_mask],
                "timestamps": micro_ts[final_micro_mask],
            },
        }


# ===========================================================================
# Feature extractors -- templates with vectorised implementations where
# obvious; clearly-marked placeholders where the method choice is still open.
# ===========================================================================

class MacroFeatureExtractor:
    """Mortality-relevant features from 60-s macro windows.

    The macro stream preserves the absolute time-of-day timeline, so
    circadian features (rest-activity rhythm, acrophase, intra-daily
    variability) can be extracted alongside intensity / sedentary metrics.
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.cfg = config or PreprocessConfig()

    # ---- intensity / sedentary -------------------------------------- #
    def sedentary_bout_stats(
        self,
        macro_enmo: np.ndarray,
        wear_mask: np.ndarray,
        sedentary_threshold_g: float = 0.02,
    ) -> dict[str, Any]:
        """Distribution of continuous sedentary bouts (minutes).

        A macro window counts as sedentary if its mean ENMO is below the
        threshold AND the device was being worn. Bouts are maximal runs of
        sedentary wear windows separated by any non-sedentary or non-wear
        window.
        """
        mean_enmo = macro_enmo.mean(axis=1)
        sedentary = (mean_enmo < sedentary_threshold_g) & wear_mask

        # Run-length extraction without Python loops.
        diff = np.diff(sedentary.astype(np.int8), prepend=0, append=0)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        bout_lengths_min = (ends - starts) * (self.cfg.macro_size_sec / 60.0)

        return {
            "bout_lengths_min": bout_lengths_min,
            "mean_bout_min": float(bout_lengths_min.mean())
            if bout_lengths_min.size
            else 0.0,
            "median_bout_min": float(np.median(bout_lengths_min))
            if bout_lengths_min.size
            else 0.0,
            "max_bout_min": float(bout_lengths_min.max())
            if bout_lengths_min.size
            else 0.0,
            "total_sedentary_min": float(bout_lengths_min.sum()),
            "n_bouts": int(bout_lengths_min.size),
        }

    def intensity_minutes(
        self, macro_enmo: np.ndarray, wear_mask: np.ndarray
    ) -> dict[str, float]:
        """Light / Moderate / Vigorous minutes via hip ENMO cutpoints.

        Cutpoints follow the conventional hip-ENMO calibration used in
        WHS-style analyses. Adjust for cohort-specific calibration.
        """
        if not wear_mask.any():
            return {"light_min": 0.0, "moderate_min": 0.0,
                    "vigorous_min": 0.0, "mvpa_min": 0.0}
        mean_enmo = macro_enmo[wear_mask].mean(axis=1)
        light = (mean_enmo >= 0.030) & (mean_enmo < 0.100)
        moderate = (mean_enmo >= 0.100) & (mean_enmo < 0.430)
        vigorous = mean_enmo >= 0.430
        minutes = self.cfg.macro_size_sec / 60.0
        return {
            "light_min": float(light.sum() * minutes),
            "moderate_min": float(moderate.sum() * minutes),
            "vigorous_min": float(vigorous.sum() * minutes),
            "mvpa_min": float((moderate | vigorous).sum() * minutes),
        }

    # ---- circadian -------------------------------------------------- #
    def circadian_metrics(
        self,
        full_enmo: np.ndarray,
        full_timestamps: np.ndarray,
        start_datetime: datetime,
    ) -> dict[str, float]:
        """Cosinor + nonparametric rest-activity metrics.

        PLACEHOLDER -- recommended implementations:
          * Cosinor (mesor, amplitude, acrophase): least-squares fit of a
            24-h sinusoid to mean ENMO per macro window
            (scipy.optimize.curve_fit).
          * Nonparametric: M10, L5, RA = (M10 - L5)/(M10 + L5), IS, IV
            (Witting et al. 1990; Van Someren et al. 1999).

        See `pyActigraphy` for reference implementations.
        """
        # TODO: implement cosinor and nonparametric RA metrics.
        return {
            "mesor": np.nan,
            "amplitude": np.nan,
            "acrophase_hours": np.nan,
            "M10": np.nan,
            "L5": np.nan,
            "RA": np.nan,
            "IS": np.nan,  # interdaily stability
            "IV": np.nan,  # intradaily variability
        }

    # ---- composite -------------------------------------------------- #
    def extract_all(self, macro_block: dict[str, Any]) -> dict[str, Any]:
        feats: dict[str, Any] = {}
        feats.update(
            self.sedentary_bout_stats(
                macro_block["full_enmo"], macro_block["wear_mask"]
            )
        )
        feats.update(
            self.intensity_minutes(
                macro_block["full_enmo"], macro_block["wear_mask"]
            )
        )
        return feats


class MicroFeatureExtractor:
    """Gait-related features from 2-s micro windows (Parkinson's + falls).

    Inputs are pre-filtered: every window passed in is already verified as
    a gait bout (active std AND dominant frequency in [1, 3] Hz).
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.cfg = config or PreprocessConfig()

    # ---- spectral --------------------------------------------------- #
    def dominant_frequency(self, micro_enmo: np.ndarray) -> np.ndarray:
        """Per-window dominant gait frequency (Hz), batched zero-padded FFT.

        Padded to GAIT_FFT_LENGTH so the ~0.117 Hz bin grid resolves cadences
        that fall between the native 0.5 Hz bins of a 60-sample window.
        """
        if micro_enmo.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        spectra = np.abs(np.fft.rfft(micro_enmo, n=GAIT_FFT_LENGTH, axis=1))
        freqs = np.fft.rfftfreq(GAIT_FFT_LENGTH, d=1.0 / self.cfg.fs)
        band = (freqs >= self.cfg.gait_freq_low_hz) & (
            freqs <= self.cfg.gait_freq_high_hz
        )
        masked = np.where(band[None, :], spectra, -np.inf)
        return freqs[masked.argmax(axis=1)]

    def rms_acceleration(self, micro_data: np.ndarray) -> np.ndarray:
        """Per-window per-axis RMS. Shape (W, 3)."""
        if micro_data.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.sqrt((micro_data ** 2).mean(axis=1))

    # ---- gait regularity / asymmetry -------------------------------- #
    def step_regularity(
        self, micro_data: np.ndarray, vertical_axis: int = 2
    ) -> dict[str, np.ndarray]:
        """Step and stride regularity via vertical-axis autocorrelation.

        PLACEHOLDER -- standard method (Moe-Nilssen & Helbostad 2004):
          1. Unbiased autocorrelation of vertical acceleration.
          2. First dominant peak  -> step regularity  (Ad1).
          3. Second dominant peak -> stride regularity (Ad2).
          4. Step asymmetry      = |Ad1 - Ad2| (or 1 - Ad1/Ad2).

        Implementation: `scipy.signal.find_peaks` on each window's autocorr.
        """
        # TODO: implement with scipy.signal.find_peaks.
        n = micro_data.shape[0]
        return {
            "step_regularity": np.full(n, np.nan),
            "stride_regularity": np.full(n, np.nan),
            "step_asymmetry": np.full(n, np.nan),
        }

    def harmonic_ratio(
        self, micro_data: np.ndarray, vertical_axis: int = 2
    ) -> np.ndarray:
        """Even / odd harmonic ratio of vertical acceleration.

        PLACEHOLDER (Smidt 1971; Lowry 2012):
            HR = sum(|even harmonics|) / sum(|odd harmonics|)
        Higher HR -> smoother, more rhythmic gait. Reduced in fall-prone
        and PD-affected individuals.
        """
        # TODO: implement once dominant frequency is locked in per window.
        return np.full(micro_data.shape[0], np.nan)

    # ---- composite -------------------------------------------------- #
    def extract_all(self, micro_block: dict[str, Any]) -> dict[str, Any]:
        micro_data = micro_block["data"]
        micro_enmo = micro_block["enmo"]
        dom_f = self.dominant_frequency(micro_enmo)
        rms = self.rms_acceleration(micro_data)
        reg = self.step_regularity(micro_data)
        hr = self.harmonic_ratio(micro_data)
        return {
            "dom_freq_hz": dom_f,
            "rms_x": rms[:, 0],
            "rms_y": rms[:, 1],
            "rms_z": rms[:, 2],
            **reg,
            "harmonic_ratio": hr,
            # Aggregations the downstream model usually wants:
            "dom_freq_mean": float(dom_f.mean()) if dom_f.size else np.nan,
            "dom_freq_std": float(dom_f.std()) if dom_f.size else np.nan,
            "n_gait_bouts": int(micro_data.shape[0]),
        }


# ---------------------------------------------------------------------------
# Minimal smoke-test (synthetic data) -- run with `python whs_preprocessor.py`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs = 30
    n_days = 7
    N = fs * 60 * 60 * 24 * n_days

    # Build a synthetic 7-day stream: device flat on table at night, walking
    # bouts during the day. Mimics the WHS off-at-night protocol.
    data = np.zeros((N, 3), dtype=np.float32)
    data[:, 2] = -1.0                           # gravity on Z axis
    for d in range(n_days):
        day_start = d * 86400 * fs + 7 * 3600 * fs
        day_end = d * 86400 * fs + 23 * 3600 * fs
        data[day_start:day_end] += rng.normal(0, 0.03, (day_end - day_start, 3))
        # Inject 20 one-minute 2-Hz walking bouts each day.
        for _ in range(20):
            t0 = rng.integers(day_start, day_end - 60 * fs)
            t = np.arange(60 * fs) / fs
            walk = 0.4 * np.sin(2 * np.pi * 2.0 * t).astype(np.float32)
            data[t0 : t0 + 60 * fs, 2] += walk

    pre = WHSPreprocessor()
    result = pre.process(
        data,
        start_datetime=datetime(2024, 1, 1, 0, 0, 0),
        subject_id="SYNTH001",
    )

    print(f"Subject valid: {result['valid']}")
    if result["valid"]:
        print(f"  wear hours / day: {result['wear_hours_by_day']}")
        print(f"  macro (wear-only): {result['macro']['data'].shape}")
        print(f"  micro gait bouts:  {result['micro']['data'].shape}")

        m_feats = MacroFeatureExtractor().extract_all(result["macro"])
        g_feats = MicroFeatureExtractor().extract_all(result["micro"])
        print(f"  sedentary bouts:  n={m_feats['n_bouts']}, "
              f"mean={m_feats['mean_bout_min']:.1f} min, "
              f"max={m_feats['max_bout_min']:.1f} min")
        print(f"  MVPA min:         {m_feats['mvpa_min']:.1f}")
        print(f"  gait dom-freq:    mean={g_feats['dom_freq_mean']:.2f} Hz, "
              f"n={g_feats['n_gait_bouts']}")
