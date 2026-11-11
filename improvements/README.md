# Improvements

Experiments that build on the reproduction in `01_ppgbp_reprod.ipynb`.
They reuse the same data, supplied 205-participant split, preprocessing,
model configurations and `evaluation_utils.evaluate_features()`, so the
numbers are directly comparable with the reproduction.

Results are appended to `results/improvement_results.csv`; per-participant
predictions and summary tables go to `results/improvements/`.

## Setup

1. Download the PPG-BP data from
   [figshare](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299)
   and unzip it so that the repo contains `data_PPG/Data File/PPG-BP dataset.xlsx`
   and `data_PPG/Data File/0_subject/`.
2. Download `papagei_s.pt`, `papagei_p.pt` (and optionally
   `papagei_s_svri.pt`) from [Zenodo](https://zenodo.org/records/13983110)
   into `weights/`.

`data_PPG/` and `weights/` are git-ignored.

## 1. Input representation for short recordings

PPG-BP recordings are 2.1 s long (~263 samples at 125 Hz). The paper and the
released notebook zero-pad them to the 10 s (1250 samples) used in
pre-training, so ~79% of each input is zeros. This experiment compares
zero padding with no padding, tiling, tiling whole heartbeats (cut at
systolic peaks so the joins don't create a fake beat), mirror padding, and
joining the three recordings before tiling.

```bash
# choose on validation
python improvements/01_input_representation.py --data-dir data_PPG --split val

# only after the choice is frozen
python improvements/01_input_representation.py --data-dir data_PPG --split test
```

Options: `--models papagei_s papagei_p papagei_s_svri`,
`--variants zero_pad native tile tile_cycles reflect concat3_tile`,
`--targets sysbp diasbp hr`. Embeddings are cached in
`data_PPG/features_improvements/`, so re-runs are fast.

`--no-weights` runs the pipeline with random weights; it is only a smoke test
and its rows are labelled `PIPELINE_TEST_...` in the CSV.

## 2. Feature fusion and course baselines

Compares PaPaGei embeddings with demographics (age, sex, height, weight,
BMI), handcrafted PPG features (sVRI, SQI, peak-interval heart rate), the
standard time series regression methods from Lab 8 (ROCKET, MiniROCKET,
Hydra, QUANT via `aeon`) and two ways of combining PaPaGei with the
handcrafted features: early fusion (concatenation, as in the paper's
Table 15) and stacked/late fusion (a ridge on the embedding gives one
out-of-fold "PPG score" that is combined with the other features).
`Hypertension` and the disease columns are never used as features.

Evaluation follows the Lab 7 / Week 9 project checklist:

- split by participant; scaling, tuning and the stacked score fitted inside
  each training fold; participant 24 left out (`--drop-ids`, default 24)
  because one of its recordings is identical to one of participant 23's;
- choose with repeated 5-fold CV on train + validation; the test set is used
  once, after retraining on train + validation;
- MAE (the paper's metric), RMSE, R^2 and gain vs the Dummy (training mean)
  baseline; fit + predict time per fold and feature-extraction time per
  participant;
- paired comparisons on the same folds: corrected resampled t-test
  (Nadeau & Bengio, 2003) with Holm correction, plus Friedman test, critical
  difference diagram and multiple comparison matrix (Wilcoxon-Holm,
  alpha = 0.05) on the per-repeat MAE (exploratory: one dataset).

```bash
# choose: repeated CV on train + validation (test set untouched)
python improvements/02_feature_fusion.py --data-dir data_PPG --mode cv

# only after the choice is frozen: retrain on train + val, test once
python improvements/02_feature_fusion.py --data-dir data_PPG --mode test
```

Outputs in `results/improvements/`: `feature_fusion_cv_summary.csv`,
`feature_fusion_cv_folds.csv`, `feature_fusion_cv_predictions.csv` (every
held-out prediction, for plots), `feature_extraction_cost.csv`, and
`figures/cd_<target>.png`, `figures/mcm_<target>.png`. Test mode writes
`feature_fusion_summary_test.csv`, `feature_fusion_predictions_test.csv` and
logs to `results/improvement_results.csv`.

Quick extra modes: `--mode figures` redraws the CD diagrams and MCMs from a
saved cv run in seconds (works with aeon 1.1 to 1.6, whose MCM arguments
differ), and `--mode timing` only measures feature extraction cost.

Useful options: `--aeon` (with nothing after it) skips the ROCKET family,
`--repeats`, `--jobs`, `--references`, `--stat-sets`, `--no-timing`,
`--out-tag` (keeps outputs of different runs apart). Padding variants from
experiment 1 can be compared under the same CV with `model@variant`, e.g.
`--embeddings papagei_s papagei_s@tile_cycles`.

### Installing aeon safely

The ROCKET-family baselines need `aeon` (`pip install aeon`). Installing it
can change NumPy / SciPy / pandas versions. That matters because pyPPG's
band-pass filter is numerically fragile (see below). Check first with
`pip install --dry-run aeon`; if it would change numpy, scipy or pandas,
install aeon into a copy of the environment instead
(`conda create -n papagei_aeon --clone papagei_env`).

### Notes

- **pyPPG's band-pass filter is numerically unstable in some environments.**
  It designs a Chebyshev II band-pass (0.5-12 Hz at 1000 Hz) in (b, a) form
  and applies it with `filtfilt`. With a cut-off this low relative to the
  sampling rate, that form is fragile: on Linux with SciPy 1.14-1.17 the
  output blew up (median correlation 0.29 with the same filter in the stable
  SOS form) and pulse peaks vanished; SciPy 1.18 was fine. Every run now
  checks the filter against the SOS version and stops if they disagree
  (`ppgbp_common.check_preprocessing`).
- IPA is computed but not used: the released logic finds a full beat in only
  about a third of the 2.1 s recordings and gives implausible values.
- `morphology.py` imports `scipy.integrate.trapz`, which recent SciPy versions
  removed, so the features are re-implemented in `improvements/ppg_features.py`.
