"""
Hand-crafted PPG features: sVRI, IPA and SQI (skewness), plus a simple
peak-interval heart rate.

These are the three indices PaPaGei-S is trained to capture (paper Sec. 3.2),
and the "PPG" baseline in the paper's Appendix E / Table 15. The formulas are
copied from the repo's morphology.py. We re-implement them here instead of
importing morphology.py because that file imports `scipy.integrate.trapz`,
which was removed in recent SciPy versions, and it also needs torch_ecg.
"""

import warnings

import numpy as np
from scipy.signal import argrelmin, find_peaks
from scipy.stats import skew

try:
    from numpy import trapezoid as _trapz
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapz


def svri(x):
    """Same as morphology.extract_svri."""
    x = np.asarray(x, dtype=float)
    max_index = int(np.argmax(x))
    if max_index == 0:
        return np.nan
    scaled = (x - x.min()) / (x.max() - x.min())
    return float(np.mean(scaled[max_index:]) / np.mean(scaled[:max_index]))


def sqi(x):
    """Same as morphology.skewness_sqi (scipy skew, bias=True)."""
    return float(skew(np.asarray(x, dtype=float)))


def ipa(x, fs):
    """Same logic as morphology.compute_ipa, with two changes:

    - the beat is shifted so its minimum is 0 before taking areas. The
      original takes areas of the filtered, z-scored signal, which is
      negative for much of the beat, so the "area ratio" can be negative or
      huge and is not meaningful;
    - NaN instead of 0 when no full beat / notch is found (imputed later).
    """
    x = np.asarray(x, dtype=float)
    try:
        minima = argrelmin(x, order=fs // 5)[0]
        beat = x[minima[0]:minima[1]]
        beat = beat - beat.min()
        notch = argrelmin(beat)[0][0]
        sys_phase = _trapz(beat[:notch])
        dias_phase = _trapz(beat[notch:])
        return float(sys_phase / dias_phase)
    except (IndexError, ZeroDivisionError):
        return np.nan


def peak_hr(x, fs):
    """Heart rate (beats/min) from the mean interval between systolic peaks.

    Same peak detection as the tile_cycles variant in experiment 1. A 2.1 s
    recording holds only 2-4 peaks, so this is a crude estimate."""
    x = np.asarray(x, dtype=float)
    peaks, _ = find_peaks(x, distance=int(0.33 * fs),
                          prominence=0.5 * np.std(x))
    if len(peaks) < 2:
        return np.nan
    return float(60.0 * fs / np.mean(np.diff(peaks)))


# IPA is computed and saved for reference, but NOT used as a feature: on
# 2.1 s PPG-BP recordings the released notch-detection logic finds a full
# beat in only ~1/3 of recordings, and the values it gives are implausible
# (0 to >4000). See the experiment-2 notes in improvements/README.md.
ALL_MORPH_NAMES = ["svri", "ipa", "sqi"]
MORPH_NAMES = ["svri", "sqi"]


def morphology_features(recordings, fs):
    """Mean sVRI, IPA, SQI over a participant's recordings (NaNs ignored).

    Returns values in the order of ALL_MORPH_NAMES."""
    rows = np.array([[svri(r), ipa(r, fs), sqi(r)] for r in recordings])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(rows, axis=0)


def peak_hr_feature(recordings, fs):
    """Mean peak-interval heart rate over a participant's recordings."""
    vals = np.array([peak_hr(r, fs) for r in recordings])
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
