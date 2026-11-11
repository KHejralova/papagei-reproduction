"""
Simple (handcrafted) features for each participant.

- Demographics: age, sex, height, weight, BMI (from the PPG-BP spreadsheet).
- Pulse-shape indices sVRI, IPA and SQI: the three indices PaPaGei-S is
  trained to capture (paper Sec. 3.2) and the "PPG" baseline in the paper's
  Appendix E / Table 15. Formulas copied from the repo's morphology.py, which
  we can't import: it uses `scipy.integrate.trapz` (removed in recent SciPy)
  and needs torch_ecg.
- Heart rate from the interval between systolic peaks.

"Hypertension" and the disease columns are never used as features:
hypertension is defined from BP, so using it would leak the target.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelmin, find_peaks
from scipy.stats import skew

try:
    from numpy import trapezoid as _trapz
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapz

DEMO_COLS = ["age", "sex", "height", "weight", "bmi"]

# IPA is computed and saved for reference, but NOT used as a feature: on
# 2.1 s PPG-BP recordings the released notch-detection logic finds a full
# beat in only ~1/3 of recordings, and the values it gives are implausible
# (0 to >4000). See improvements/README.md.
ALL_MORPH_NAMES = ["svri", "ipa", "sqi"]
MORPH_NAMES = ["svri", "sqi"]
HANDCRAFTED_CACHE = "handcrafted_svri_ipa_sqi_peakhr.csv"


# --------------------------------------------------------------------------
# Single-recording features
# --------------------------------------------------------------------------

def find_systolic_peaks(x, fs):
    """Indices of the systolic peaks: at least 0.33 s apart (max 180 bpm)
    and at least half a standard deviation more prominent than their
    surroundings, which skips the smaller diastolic bumps."""
    peaks, _ = find_peaks(x, distance=int(0.33 * fs),
                          prominence=0.5 * np.std(x))
    return peaks


def svri(x):
    """Same as morphology.extract_svri: mean after the peak / mean before."""
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
    - NaN instead of 0 when no full beat / notch is found.
    """
    x = np.asarray(x, dtype=float)
    try:
        minima = argrelmin(x, order=fs // 5)[0]
        beat = x[minima[0]:minima[1]]
        beat = beat - beat.min()
        notch = argrelmin(beat)[0][0]
        return float(_trapz(beat[:notch]) / _trapz(beat[notch:]))
    except (IndexError, ZeroDivisionError):
        return np.nan


def peak_hr(x, fs):
    """Heart rate (beats/min) from the mean interval between systolic peaks.

    A 2.1 s recording holds only 2-4 peaks, so this is a crude estimate."""
    peaks = find_systolic_peaks(np.asarray(x, dtype=float), fs)
    if len(peaks) < 2:
        return np.nan
    return float(60.0 * fs / np.mean(np.diff(peaks)))


# --------------------------------------------------------------------------
# Participant-level features
# --------------------------------------------------------------------------

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


def demographic_features(meta):
    """(n, 5) array: age, sex (male = 1), height, weight, BMI."""
    demo = meta[DEMO_COLS].copy()
    demo["sex"] = demo["sex"].map({"Male": 1.0, "Female": 0.0})
    if demo.isna().any().any():
        raise ValueError("Unexpected missing demographic values")
    return demo.to_numpy(float)


def handcrafted_features(subject_ids, get_recordings, cache_dir, fs):
    """{'morph': (n, 2), 'peak_hr': (n, 1)} for the given participants.

    Values are cached per participant in cache_dir, so recordings are only
    preprocessed (via get_recordings()) for participants not seen before.
    """
    cols = ALL_MORPH_NAMES + ["peak_hr"]
    cache_csv = Path(cache_dir) / HANDCRAFTED_CACHE
    cached = (pd.read_csv(cache_csv).set_index("subject_ID")
              if cache_csv.exists() else pd.DataFrame())
    missing = [i for i in subject_ids if i not in cached.index]
    if missing:
        recs = get_recordings()
        new = pd.DataFrame(
            [list(morphology_features(recs[i], fs)) +
             [peak_hr_feature(recs[i], fs)] for i in missing],
            index=pd.Index(missing, name="subject_ID"), columns=cols)
        cached = pd.concat([cached, new])
        cached.to_csv(cache_csv)

    used = MORPH_NAMES + ["peak_hr"]
    n_fail = int(np.isnan(cached.loc[subject_ids, used].to_numpy(float))
                 .any(axis=1).sum())
    n_ipa = int(cached.loc[subject_ids, "ipa"].isna().sum())
    print(f"Handcrafted features: {n_fail} of {len(subject_ids)} participants "
          "have a missing value (filled with the training median). IPA (not "
          f"used) could not be computed for {n_ipa} of {len(subject_ids)}.")
    return {"morph": cached.loc[subject_ids, MORPH_NAMES].to_numpy(float),
            "peak_hr": cached.loc[subject_ids, ["peak_hr"]].to_numpy(float)}
