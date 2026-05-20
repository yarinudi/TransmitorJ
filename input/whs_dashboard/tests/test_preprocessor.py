"""Tests for the gait verifier and dominant-frequency estimator.

Pinpoint the two algorithmic fixes:
  1. Zero-pad rFFT to ``GAIT_FFT_LENGTH`` so 1.8 Hz (and other off-grid
     cadences) survive instead of snapping to the native 0.5 Hz grid.
  2. Gait gate requires the in-band peak to be >= the out-of-band peak
     (after zeroing DC) — strong out-of-band tones with weak in-band
     content must be rejected.
"""
from __future__ import annotations

import numpy as np

from whs_preprocessor import (
    GAIT_FFT_LENGTH,
    MicroFeatureExtractor,
    PreprocessConfig,
    WHSPreprocessor,
)


# ---------------------------------------------------------------------------
# Zero-padded resolution
# ---------------------------------------------------------------------------

def test_dominant_frequency_recovers_1_8hz():
    """A 1.8 Hz sinusoid must be recovered to within ±0.15 Hz.

    On the native 2-s × 30-Hz grid (0.5 Hz bins) the dominant bin would snap
    to 1.5 or 2.0 Hz — both 0.3 Hz away from the truth — so this assertion
    only holds with the zero-padded FFT.
    """
    cfg = PreprocessConfig()
    L = cfg.micro_size_sec * cfg.fs  # 60 samples
    target_hz = 1.8

    t = np.arange(L) / cfg.fs
    # DC + 1.8 Hz oscillation. dominant_frequency restricts the argmax to
    # the gait band so the DC bin is harmless here.
    enmo = (0.3 + 0.3 * np.sin(2 * np.pi * target_hz * t)).astype(np.float32)
    micro_enmo = enmo[np.newaxis, :]

    dom = MicroFeatureExtractor(cfg).dominant_frequency(micro_enmo)
    assert dom.shape == (1,)
    assert abs(float(dom[0]) - target_hz) <= 0.15, (
        f"recovered {float(dom[0]):.3f} Hz; expected within 0.15 Hz of {target_hz} "
        f"(off-grid cadence: zero-pad missing?)"
    )


def test_gait_fft_bin_grid_is_zero_padded():
    """Sanity: the dominant-frequency bin spacing comes from the padded length."""
    cfg = PreprocessConfig()
    freqs = np.fft.rfftfreq(GAIT_FFT_LENGTH, d=1.0 / cfg.fs)
    native_freqs = np.fft.rfftfreq(cfg.micro_size_sec * cfg.fs, d=1.0 / cfg.fs)
    assert (freqs[1] - freqs[0]) < (native_freqs[1] - native_freqs[0])
    assert abs((freqs[1] - freqs[0]) - cfg.fs / GAIT_FFT_LENGTH) < 1e-9


# ---------------------------------------------------------------------------
# In-band vs out-of-band gate
# ---------------------------------------------------------------------------

def test_verify_gait_bouts_rejects_strong_out_of_band_peak():
    """A weak 2 Hz tone under a stronger 6 Hz tone must be rejected.

    Out-of-band peak (6 Hz, amplitude 0.30) dominates the in-band peak
    (2 Hz, amplitude 0.05), so the gate's ``in_band_peak >= out_band_peak``
    fails and the window is filtered out.
    """
    cfg = PreprocessConfig()
    L = cfg.micro_size_sec * cfg.fs
    t = np.arange(L) / cfg.fs

    enmo = (0.5
            + 0.05 * np.sin(2 * np.pi * 2.0 * t)
            + 0.30 * np.sin(2 * np.pi * 6.0 * t))
    enmo = np.clip(enmo, 0.0, None).astype(np.float32)
    micro_enmo = enmo[np.newaxis, :]

    gait_mask = WHSPreprocessor(cfg).verify_gait_bouts(micro_enmo)
    assert gait_mask.shape == (1,)
    assert not bool(gait_mask[0]), "window dominated by 6 Hz tone must be rejected"


def test_verify_gait_bouts_accepts_clean_in_band_signal():
    """A clean 2.0 Hz signal must pass — in-band peak dominates by a wide margin."""
    cfg = PreprocessConfig()
    L = cfg.micro_size_sec * cfg.fs
    t = np.arange(L) / cfg.fs

    enmo = (0.3 + 0.3 * np.sin(2 * np.pi * 2.0 * t)).astype(np.float32)
    micro_enmo = enmo[np.newaxis, :]

    gait_mask = WHSPreprocessor(cfg).verify_gait_bouts(micro_enmo)
    assert bool(gait_mask[0]), "clean 2 Hz window must pass the gait filter"


def test_verify_gait_bouts_filters_quiet_window():
    """The std pre-filter still applies — sub-threshold std rejects the window
    before the FFT even runs."""
    cfg = PreprocessConfig()
    L = cfg.micro_size_sec * cfg.fs

    # std well below gait_std_threshold_g (0.05 g)
    enmo = (0.001 * np.random.default_rng(0).standard_normal(L)).astype(np.float32)
    micro_enmo = enmo[np.newaxis, :]

    gait_mask = WHSPreprocessor(cfg).verify_gait_bouts(micro_enmo)
    assert not bool(gait_mask[0])
