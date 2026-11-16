# Improvements

Part B of the project. For the overview and setup, see the
[main README](../README.md). This file explains the code in this folder
and how to run it.

## Files

| File | What it does |
| --- | --- |
| `data.py` | Loads the PPG-BP metadata, the supplied split and the recordings, and preprocesses them the same way as the reproduction notebook |
| `features.py` | Simple features: demographics, sVRI, SQI and heart rate from pulse peaks |
| `representations.py` | Different ways to turn a 2.1 s recording into PaPaGei's 10 s input |
| `embeddings.py` | Loads PaPaGei and computes one embedding per participant (cached) |
| `estimators.py` | The compared methods: ridge on feature sets, stacking, ROCKET family |
| `evaluation.py` | Repeated CV, metrics, corrected t-test with Holm correction, timing |
| `plots.py` | CD diagrams and multiple comparison matrices |
| `01_input_representation.py` | Experiment 1 |
| `02_feature_fusion.py` | Experiment 2 |
| `03_report_figures.py` | Figures and results table for the report |

We named them `estimators.py` and `plots.py` instead of `models.py` and
`figures.py` because the repo already has a `models` package and a
`figures` folder.

## Setup

You need the PPG-BP data in `data_PPG/Data File/` and `papagei_s.pt` and
`papagei_p.pt` in `weights/` (see the main README). Experiment 2 also needs
`aeon` for the ROCKET baselines. Installing it can change the NumPy/SciPy
versions, so check with `pip install --dry-run aeon` first!

## Running the experiments

```bash
# Experiment 1: input representations, validation set only
python improvements/01_input_representation.py --data-dir data_PPG --split val

# Experiment 2: repeated CV on train + validation (test set not used)
python improvements/02_feature_fusion.py --data-dir data_PPG --mode cv

# Experiment 2: final test, run once after choosing
python improvements/02_feature_fusion.py --data-dir data_PPG --mode test

# Report figures and results table
python improvements/03_report_figures.py
```

Embeddings and features are cached in `data_PPG/features_improvements/`,
so later runs are much faster. `--mode figures` redraws the CD diagrams
without rerunning the CV. Run any script with `--help` to see all options.

## Outputs

Everything is saved in `results/improvements/`:

- `feature_fusion_cv_summary.csv`: CV results and significance tests
- `feature_fusion_summary_test.csv`: test results
- `input_representation_summary_val.csv`: experiment 1 results
- `figures/`: CD diagrams and multiple comparison matrices
- `report/`: the figures and table used in the report

Files ending in `_concat_vs_stacked` and `_padding` are from earlier test
runs and aren't used in the report.

## Notes

- **pyPPG's filter can silently break.** With some SciPy versions on Linux
  (1.14 to 1.17) the band-pass filter output blew up and all features were
  wrong, without any error. `data.py` now checks the filter on every run and
  stops if it's broken. Our results were made with SciPy 1.15.3 on macOS.
- **IPA is not used.** On the short 2.1 s recordings the original code
  only finds a full heartbeat in about a third of the recordings, and the
  values are not realistic.
- **Participant 24 is left out of experiment 2** because one of its
  recordings is identical to one of participant 23's.