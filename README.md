# Reproducing and improving PaPaGei on PPG-BP

Group project for COMP41850 (AI for Time Series), University College Dublin, by **Karolina Hejralova** and **Petra Polakova**.

The project has two parts:

- **Part A: Reproduction.** Downstream evaluation of the pretrained PaPaGei-S and PaPaGei-P foundation models ([Pillai et al., ICLR 2025](https://arxiv.org/abs/2410.20542)) on the PPG-BP dataset, compared with the published results and with simple baselines, random forests and MOMENT.
- **Part B: Improvements.** (1) Alternatives to zero padding of the short 2.1 s recordings, and (2) a stacking method that combines PaPaGei with inexpensive participant and signal features, compared with ROCKET-family time-series baselines under repeated cross-validation with corrected statistical tests.

[Section 3](#3-reproducing-each-result-in-the-report) lists, for every table and figure in the report, where it comes from and how to reproduce it.

## Contents

1. [Environment](#1-environment)
2. [Data, weights and saved inputs](#2-data-weights-and-saved-inputs)
3. [Reproducing each result in the report](#3-reproducing-each-result-in-the-report)
4. [Part A: Reproduction](#4-part-a-reproduction)
5. [Part B: Improvements](#5-part-b-improvements)
6. [Results compared with the paper](#6-results-compared-with-the-paper)
7. [Interpretation and reproducibility limits](#7-interpretation-and-reproducibility-limits)
8. [Attribution and AI assistance](#8-attribution-and-ai-assistance)

## Repository map

| File or directory | Purpose |
| --- | --- |
| `PPG-BP_reprod_tidy_v.ipynb` | Main report notebook: inspect results, generate comparisons and optionally rerun selected stages. |
| `01_ppgbp_reprod.ipynb` | Original development notebook containing the experimental code and exploratory analysis. Retained as a record; it is not a guaranteed clean-kernel, Run All entry point. |
| `evaluation_utils.py` | Shared Ridge evaluator used for the duplicate sensitivity check and compatible feature evaluations. |
| `results_utils.py` | Prints regression summaries and appends metrics and run details to a CSV log. |
| `results/` | Saved metrics, predictions, split information and reference values used in the report. |
| `reproduction_artifacts.zip` | Saved features, processed signals, released split CSVs and historical regression dictionaries required by the report notebook. Extract into the repository root. |
| `requirements.txt` | Original repository dependency list; versions are mostly unpinned. |
| `improvements/` | Part B: code for the two improvement experiments and the report figures (see [`improvements/README.md`](improvements/README.md)). |
| `results/improvements/` | Part B outputs: CV and test summaries, predictions, CD diagrams / MCMs and the report figures and results table (`report/`). |
| `results/improvement_results.csv` | Part B log of every validation / test evaluation (one row per run, method and target). |
| `environment-improvements.yml`, `requirements-improvements-lock.txt` | Exported environment that produced all Part B results (exact versions). |
| `models/`, `linearprobing/`, `preprocessing/`, `baselines/` | Original PaPaGei implementation used by the reproduction. Keep these packages in the repository. |
| `README_upstream.md` and `LICENSE` | Original project documentation and licence. |


## 1. Environment

The work was run on CPU using the Conda environment `papagei_env`. The supplied report notebook records Python 3.10.21. Use that working environment when available.

A starting installation for a new environment is:

```bash
conda create -n papagei_env python=3.10.21
conda activate papagei_env
python -m pip install -r requirements.txt
python -m pip install "pyPPG==1.0.41" "matplotlib-inline==0.1.6" jupyterlab ipykernel jinja2
python -m ipykernel install --user --name papagei_env --display-name "Python (papagei_env)"
```

This is a dependency-installation recipe, not a version-locked recreation of the original environment. Dependency versions can affect preprocessing and fitted results. Before archiving the final experiment, export the actual working environment:

```bash
conda env export --no-builds > environment.yml
python -m pip freeze > requirements-reproduction-lock.txt
```

These exports should be made from the environment that successfully runs the notebook. The report notebook uses saved MOMENT embeddings and does not need to download the MOMENT model. Repeating MOMENT feature extraction in the original development workflow additionally requires `momentfm` and the pretrained model `AutonLab/MOMENT-1-large`.


### Additional packages for Part B

The improvement experiments use the same environment plus [`aeon`](https://www.aeon-toolkit.org/) for the ROCKET-family baselines:

```bash
python -m pip install --dry-run aeon   # check first: it must not change numpy, scipy or pandas
python -m pip install aeon
```

If the dry run would change `numpy`, `scipy` or `pandas`, install `aeon` into a copy of the environment instead (`conda create -n papagei_aeon --clone papagei_env`). This matters because pyPPG's band-pass filter is numerically fragile with some SciPy builds (see [Section 7](#7-interpretation-and-reproducibility-limits)).

All Part B results were produced with **Python 3.10, SciPy 1.15.3, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, torch 2.14.1 and aeon 1.3.0 on macOS (Apple Silicon), CPU only**. The exact environment is exported in `environment-improvements.yml` (conda) and `requirements-improvements-lock.txt` (pip). Every Part B run checks the filter against a numerically stable version and stops if they disagree (`improvements/data.py`).

## 2. Data, weights and saved inputs

#### Raw PPG-BP data

Download [PPG-BP from Figshare](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299). Extract it so these paths exist relative to the repository root:

- `data_PPG/Data File/PPG-BP dataset.xlsx`
- `data_PPG/Data File/0_subject/2_1.txt` and the remaining raw recordings.

The raw data are required even in saved-results mode: the notebook inspects a waveform, reads participant metadata and checks recording hashes. Raw data are not bundled in the saved-artifact archive.

#### Saved reproduction artefacts

Download [`reproduction_artifacts.zip`](reproduction_artifacts.zip) and extract its **contents into the repository root**, preserving the paths inside it. For example:

```bash
python -m zipfile -e reproduction_artifacts.zip .
```

The archive contains:

- Released `train.csv`, `val.csv` and `test.csv` under `data_PPG/Data File/`.
- Preprocessed released recordings under `data_PPG/Data File/ppg/`.
- Participant embedding dictionaries for PaPaGei-S, PaPaGei-P and MOMENT under `data_PPG/features/`.
- Adapted PaPaGei embeddings and cleaned padded recordings under `data_PPG/reproduction_219_seed42/`.
- Eighteen saved regression dictionaries under `results/`: two models × three targets × original/rerun/scaling procedures.
- A SHA-256 manifest, `reproduction_artifacts_manifest.json`, identifying the bundled files.

The remaining result CSVs must be present under `results/` from the repository checkout. The archive does not replace those CSVs. Keep the archive and result CSVs from the same project version together.

#### PaPaGei checkpoints

Download `papagei_s.pt` and `papagei_p.pt` from the [authors' Zenodo record](https://zenodo.org/records/13983110), and place them under `weights/`.

Checkpoints are required to regenerate embeddings; they are not required when loading all saved embeddings. The unused S-sVRI variant is not needed for the main report notebook. Raw data and downloaded checkpoints remain outside Git through `.gitignore`.


Part B only needs the raw PPG-BP data and the two checkpoints above; it does not use `reproduction_artifacts.zip`. Its preprocessed signals, features and embeddings are computed from the raw data and cached in `data_PPG/features_improvements/` (git-ignored), so repeated runs are fast.

## 3. Reproducing each result in the report

Run everything from the repository root. Table and figure numbers refer to the submitted report.

| Report result | Part | Source in this repository | How to reproduce |
| --- | --- | --- | --- |
| Reproduced BP and HR MAEs vs the paper; setups compared (Appendix table) | A | `PPG-BP_reprod_tidy_v.ipynb`, Sections 4, 5.2 and 6.4; `results/reproduction_219_seed42/` | Run the notebook ([Section 4.1](#41-display-the-reported-results)); recompute switches in [Section 4.2](#42-recompute-individual-experiments) |
| Regularisation grid and scaling checks | A | Notebook Sections 5.2 and 5.4; `results/alpha_grid_comparison.csv`, `results/scaling_comparison.csv` | Notebook, Sections 5.2 and 5.4 |
| Validation / test / combined MAE (Appendix table) | A | Notebook Section 5.3; `results/evaluation_split_comparison.csv` | Notebook, Section 5.3 |
| Duplicate recordings (participants 23 and 24) | A | Notebook Section 5.5 | Notebook, Section 5.5 |
| BP MAE of PaPaGei and baselines (Table I) | A | Notebook Sections 8 and 9; `results/baselines/` | Notebook, Sections 8.1, 8.2, 8.4 and 9 |
| Hypertension classification (Table II) | A | Notebook Sections 7, 8.3, 8.4 and 9; `results/classification_205_released/`, `results/baselines/` | Notebook, Sections 7, 8.3 and 9 |
| Participant-level prediction errors (Fig. 1) | A | Notebook Section 6.5 | Notebook, Section 6.5 |
| Experiment 1: input representations | B | `results/improvements/input_representation_summary_val.csv` | `python improvements/01_input_representation.py --data-dir data_PPG --split val` |
| Experiment 2: CV results and corrected t-tests | B | `results/improvements/feature_fusion_cv_summary.csv` | `python improvements/02_feature_fusion.py --data-dir data_PPG --mode cv` |
| Experiment 2: test-set results | B | `results/improvements/feature_fusion_summary_test.csv` | `python improvements/02_feature_fusion.py --data-dir data_PPG --mode test` |
| Experiment 2 results table (Table III) | B | `results/improvements/report/results_table.tex` | `python improvements/03_report_figures.py` (after the two commands above) |
| Input representation figure (Fig. 2) | B | `results/improvements/report/fig_padding.pdf` | `python improvements/03_report_figures.py` |
| Accuracy vs cost figure (Fig. 3) and extraction times | B | `results/improvements/report/fig_cost_sysbp.pdf`, `results/improvements/feature_extraction_cost.csv` | `03_report_figures.py`; the timing is measured at the end of `02 --mode cv` (or alone with `--mode timing`) |

## 4. Part A: Reproduction
Part A investigates the downstream performance of pretrained **PaPaGei-S** and **PaPaGei-P** on the PPG-BP dataset. It covers systolic and diastolic blood-pressure regression, heart-rate regression, normal versus non-normal blood-pressure classification, data-integrity checks, and comparisons with constant predictors, random forests and MOMENT.

**Start with [`PPG-BP_reprod_tidy_v.ipynb`](PPG-BP_reprod_tidy_v.ipynb).** It presents the experiments, saved results and discussion in order. It also allows selected calculations to be repeated. PaPaGei pretraining is not repeated.

### Experimental setups

| Setup | Training | Validation | Test | Main differences |
| --- | ---: | ---: | ---: | --- |
| Original 205 participants | 123 | 41 | 41 | Released participant lists; Ridge alpha includes 1000; scaling before CV. |
| 205-participant rerun | 123 | 41 | 41 | Same split, with Ridge alpha restricted to 0.1, 1, 10 and 100. |
| Adapted 219 participants | 131 | 44 | 44 | Six within-participant duplicate recordings removed; participants sharing a recording grouped together; scaling inside grouped CV. |

Unless a section explicitly reports separate splits, evaluation combines validation and test participants: **82 participants for the 205 setup and 88 for the 219 setup**. Hyperparameter selection uses training participants only. The adapted experiment changes several factors together; it does not isolate the effect of adding participants.

Classification uses `Normal = 0`; prehypertension and both hypertension stages are grouped as `non-normal = 1`. The final classification baseline table uses only the 205 setup.


### 4.1 Display the reported results

From the repository root:

```bash
conda activate papagei_env
jupyter lab PPG-BP_reprod_tidy_v.ipynb
```

1. Select **Python (papagei_env)**.
2. Leave recomputation switches set to `False`.
3. Skip the installation cell when dependencies are already installed. If installation is needed, run it separately and restart the kernel afterwards.
4. Run the remaining cells in order.

The notebook loads saved model results, recalculates lightweight checks and constant/majority baselines, and displays the comparisons. Section 9 contains the combined regression and classification tables.

A saved-results run requires the dataset and saved artefacts above. It is not a CSV-only report or an end-to-end run from raw data with no saved inputs.


### 4.2 Recompute individual experiments

A `True` switch enables its stage only when that cell is executed. It does not automatically run prerequisites, enable downstream stages or save the new metrics. When upstream inputs change, rerun their dependent cells in order. Returning a switch to `False` reloads the archived result.

| Result or check | Notebook section | How to reproduce it |
| --- | --- | --- |
| Original signal preprocessing | 2.2 | Set `RECOMPUTE_RELEASED_PREPROCESSING=True`; run preprocessing from raw data. Intermediate signal files are overwritten. |
| Released S/P embeddings | 3 | Set `RECOMPUTE_RELEASED_EMBEDDINGS=True`; run with processed signals and checkpoints. Intermediate feature files are overwritten. |
| Original BP and HR MAEs | 4 | Change the actual call to `get_original_results(recompute=True)` and run Section 4 using the released embeddings. In the supplied version, the call still contains `False`; the separately declared `RECOMPUTE_ORIGINAL_REGRESSION` flag is not connected to it. |
| Smaller-alpha-grid rerun | 5.2 | Set the local `recompute_rerun=True` in that section and execute its result loop and comparison. |
| Validation/test error comparison | 5.3 | Run the section after Section 4; it reuses predictions without fitting another model. |
| Scaling inside CV | 5.4 | Set the local `recompute_scaling=True`; run its loop and comparison. |
| Duplicate audit | 5.5 | Run the hash and label inspection cells on raw recordings. |
| Excluding participant 24 | 5.5 | Set the local `recompute_duplicate_check=True`; evaluate the two conditions. |
| Excluded-participant audit | 5.6 | Run the recording-availability and label checks. |
| Adapted grouped split | 6 | Set `REBUILD_219_SPLIT=True`; reconstruct and compare against the saved reference split. The reference CSV is still required. |
| Adapted embeddings | 6.2 | Set `RECOMPUTE_219_EMBEDDINGS=True`; preprocess retained raw signals and extract embeddings in memory. |
| Nested training CV | 6.3 | Set `RECOMPUTE_NESTED_CV=True`; run four outer/four inner grouped folds on training participants. |
| Final adapted BP models | 6.4 | Set `RECOMPUTE_219_REGRESSION=True`; tune on training participants and evaluate the combined held-out participants. |
| Individual and split-specific errors | 6.5 | Run using `final_predictions` from Section 6.4. |
| PaPaGei classification | 7 | Set `RECOMPUTE_CLASSIFICATION=True`; run the helpers and both setup-specific cells. |
| Mean/median BP baselines | 8.1 | Run the section; constants are always calculated using training labels only. |
| Statistical features | 8.2 | Set `RECOMPUTE_STATISTICAL_FEATURES=True`; requires existing cleaned padded recordings from the archive. |
| RF regression | 8.2 | Set `RECOMPUTE_RF_REGRESSION=True`; fit and tune forests from the statistical features. |
| Majority/RF classification | 8.3 | Majority results are always recalculated. Set `RECOMPUTE_RF_CLASSIFICATION=True` to refit the forest. |
| MOMENT BP/classification models | 8.4 | Set `RECOMPUTE_MOMENT_MODELS=True`; refit the downstream models from saved MOMENT embeddings. This does not rerun MOMENT extraction. |
| Combined report tables | 9 | Run after the preceding sections; tables use the selected in-memory results. |

The three lowercase sensitivity switches are assigned again in their own sections, so change those local assignments rather than only the copies in the setup cell.

The original development notebook documents additional feature-generation code and historical experiments. It contains exploratory cells, local paths and dependencies on previous kernel state, so it should be read and run by section. Exact clean-kernel regeneration of every historical artefact has not been verified. In particular, rebuilding MOMENT embeddings requires checking the original input-list definitions. The archive provides the features actually used for the saved results.


### 4.3 The custom helpers

#### `evaluation_utils.evaluate_features()`

Fits a `StandardScaler`–`Ridge` pipeline with scaling inside four-fold, unshuffled KFold CV. The search tests alpha `[0.1, 1, 10, 100]` and solvers `auto`, `cholesky`, `sparse_cg`, selecting by negative MAE. It returns MAE, predictions, selected parameters and a 95% MAE interval from 500 participant bootstrap samples, with seed 42 by default.


#### `results_utils.record_result()`

Prints a regression summary and appends one row to a CSV, including run ID, model, target, procedure, sample counts, MAE, confidence interval and selected parameters. It checks the existing CSV schema before appending. Calling it repeatedly appends repeated rows; it does not replace an existing run. 


## 5. Part B: Improvements

Full details, options and design notes are in [`improvements/README.md`](improvements/README.md). In short:

- **Experiment 1 (input representation).** PaPaGei expects 10 s inputs (1250 samples at 125 Hz), but PPG-BP recordings last 2.1 s, so the released pipeline zero-pads about 79% of each input. The experiment compares zero padding with the unpadded clip, tiling, tiling whole heartbeats, mirror padding and joining the three recordings, on the validation set only. No representation was consistently better, so Experiment 2 keeps zero padding.
- **Experiment 2 (stacking with simple features).** Compares simple features (demographics, sVRI, SQI, peak-interval heart rate), PaPaGei-S/P alone, early fusion (embedding + demographics, as in the paper) and the proposed late fusion (an out-of-fold "PPG score" from the embedding, combined with the simple features), plus ROCKET, MiniROCKET, Hydra and QUANT on the preprocessed signal.
- **Evaluation.** Uses the supplied 205-participant split with participant 24 removed (one of its recordings is identical to one of participant 23's). Methods are compared with 5-fold CV repeated 10 times on the 163 train + validation participants, with every fitted step inside the training folds. Differences are tested with the Nadeau–Bengio corrected t-test with Holm correction, and exploratory Friedman / CD diagrams / MCMs are produced. The test set (41 participants) is used once, after retraining on train + validation.

```bash
# Experiment 1 (validation only)
python improvements/01_input_representation.py --data-dir data_PPG --split val

# Experiment 2: choose with repeated CV (test set untouched); also measures the cost
python improvements/02_feature_fusion.py --data-dir data_PPG --mode cv

# Experiment 2: after the choices are frozen, retrain on train + val and test once
python improvements/02_feature_fusion.py --data-dir data_PPG --mode test

# Report figures and results table (from the saved outputs, no refitting)
python improvements/03_report_figures.py
```

Outputs go to `results/improvements/` (report figures and table in `results/improvements/report/`, CD diagrams and MCMs in `results/improvements/figures/`). Files with the suffixes `_concat_vs_stacked` and `_padding` are earlier exploratory CV runs (made with `--out-tag`) and are not used in the report.

## 6. Results compared with the paper

### Part A: Reproduction

The published reference values below come from `results/paper_reference_results.csv`; reproduced original and rerun values come from the saved regression dictionaries, and adapted values from `results/reproduction_219_seed42/final_bp_results.csv`. Lower MAE is better.

| Model | Target | Paper MAE | Original 205 | Rerun 205 | Adapted 219 |
| --- | --- | ---: | ---: | ---: | ---: |
| PaPaGei-S | Systolic BP (mmHg) | 14.390 | 14.254 | 14.371 | 16.347 |
| PaPaGei-P | Systolic BP (mmHg) | 13.600 | 14.307 | 17.946 | 15.137 |
| PaPaGei-S | Diastolic BP (mmHg) | 8.710 | 8.910 | 8.950 | 9.043 |
| PaPaGei-P | Diastolic BP (mmHg) | 8.880 | 8.882 | 9.376 | 9.250 |
| PaPaGei-S | Heart rate (bpm) | 4.000 | 4.535 | 4.535 | — |
| PaPaGei-P | Heart rate (bpm) | 3.470 | 4.585 | 4.327 | — |

For normal/non-normal classification, higher AUROC is better. The intervals below are the saved reproduced 95% bootstrap intervals.

| Model | Paper AUROC | Original 205 AUROC (95% CI) | Adapted 219 AUROC (95% CI) |
| --- | ---: | --- | --- |
| PaPaGei-S | 0.770 | 0.758 [0.647, 0.861] | 0.641 [0.479, 0.775] |
| PaPaGei-P | 0.740 | 0.638 [0.505, 0.766] | 0.708 [0.556, 0.833] |

The notebook includes the complete baseline tables and regression intervals. BP errors were close to some published values, but model rankings and heart-rate results were not fully reproduced. Changes to the grid or evaluation setup did not consistently improve agreement.


### Part B: Improvements

MAE of the main methods in Experiment 2 (mmHg for BP, bpm for HR). CV is the mean over 10 repeats of 5-fold CV on 163 participants; test is the single evaluation on the 41 test participants. The full table, with standard deviations and confidence intervals, is in `results/improvements/report/results_table.md`.

| Method | SBP CV | SBP test | DBP CV | DBP test | HR CV | HR test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| *PaPaGei-S, paper (combined val + test)* | — | *14.39* | — | *8.71* | — | *4.00* |
| Dummy (training mean) | 15.92 | 17.34 | 8.16 | 10.15 | 8.82 | 8.03 |
| Simple features | 14.66 | 12.73 | 8.16 | 8.81 | 3.20 | 3.25 |
| PaPaGei-S | 14.84 | 11.07 | 8.83 | 8.75 | 3.97 | 4.12 |
| PaPaGei-S + demographics, concatenated (paper's fusion) | 14.27 | 9.30 | 8.28 | 7.76 | 4.01 | 4.06 |
| **PaPaGei-S + simple features, stacked (proposed)** | **13.95** | **10.53** | **8.03** | **8.38** | **3.09** | **3.33** |
| ROCKET | 12.35 | 11.71 | 7.54 | 8.71 | 3.39 | 3.12 |
| MiniROCKET | 12.38 | 12.13 | 7.64 | 8.78 | 3.34 | 3.10 |

The paper's values are on a different evaluation set (combined validation and test participants of the 205 split), so they are a reference point rather than a like-for-like comparison. In CV, stacking improved PaPaGei-S on all three targets (significantly only for HR after Holm correction), but PaPaGei added little beyond the simple features, and the ROCKET family reached the lowest BP errors. With 41 test participants the test confidence intervals are 6–7 mmHg wide and overlap, so CV is the primary evidence.

## 7. Interpretation and reproducibility limits

- The released split omits 14 labelled participants. Their available recordings and labels do not establish the reason for exclusion.
- One exact recording is shared by participants 23 and 24 across the original training/validation split. The adapted setup keeps those participants together.
- The 205 statistical baseline uses duplicate-cleaned padded recordings, whereas the original PaPaGei embeddings retain the released recording handling. This limits strict like-for-like interpretation.
- Padding occupies much of each short recording and can dominate simple amplitude statistics.
- Bootstrap intervals describe evaluation-sample uncertainty for fixed fitted models, not variation from retraining on new splits.
- The historical RF classification run was unseeded. The verification branch uses seed 42 and may not recover the same score.
- Some historical regression bootstraps were not seeded. Exact intervals can change on recomputation.
- This README and archive were checked against the submitted project snapshot. A fresh environment installation and full model-training workflow have not been validated end to end.
- **pyPPG's band-pass filter is numerically fragile.** On Linux with SciPy 1.14–1.17 its output blew up and the pulse shape was lost, without any error, while SciPy 1.18 on Linux and SciPy 1.15.3 on macOS were fine. Part B checks the filter against a stable implementation on every run (`improvements/data.py`).
- IPA (one of PaPaGei-S's three morphology indices) is computed but not used in Part B: on the 2.1 s recordings the released logic finds a full beat in only about a third of recordings and gives implausible values.
- Part B results come from a single small dataset (163 participants for CV, 41 for test) and should be confirmed on larger cohorts.

## 8. Attribution and AI assistance

The implementation is based on [Nokia Bell Labs' PaPaGei repository](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) and [the PaPaGei paper](https://arxiv.org/abs/2410.20542). Original documentation is retained in `README_upstream.md`; original licence and copyright notices are preserved in `LICENSE` and the source files. MOMENT embeddings use `AutonLab/MOMENT-1-large`.

For Part A (reproduction), ChatGPT (OpenAI) assisted with Python code development and debugging, notebook organisation and comments. For Part B (improvements), Claude (Anthropic) assisted with code development and debugging, and code comments. Responsibility for the submitted analysis and conclusions remains with the project authors. 
