# PaPaGei reproduction on PPG-BP

This part of the project investigates the downstream performance of pretrained **PaPaGei-S** and **PaPaGei-P** on the PPG-BP dataset. It covers systolic and diastolic blood-pressure regression, heart-rate regression, normal versus non-normal blood-pressure classification, data-integrity checks, and comparisons with constant predictors, random forests and MOMENT.

**Start with [`PPG-BP_reprod_tidy_v.ipynb`](PPG-BP_reprod_tidy_v.ipynb).** It presents the experiments, saved results and discussion in order. It also allows selected calculations to be repeated. PaPaGei pretraining is not repeated.

This README documents the reproduction work only. Improvement experiments are outside its scope.

## Files and their roles

| File or directory | Purpose |
| --- | --- |
| `PPG-BP_reprod_tidy_v.ipynb` | Main report notebook: inspect results, generate comparisons and optionally rerun selected stages. |
| `01_ppgbp_reprod.ipynb` | Original development notebook containing the experimental code and exploratory analysis. Retained as a record; it is not a guaranteed clean-kernel, Run All entry point. |
| `evaluation_utils.py` | Shared Ridge evaluator used for the duplicate sensitivity check and compatible feature evaluations. |
| `results_utils.py` | Prints regression summaries and appends metrics and run details to a CSV log. |
| `results/` | Saved metrics, predictions, split information and reference values used in the report. |
| `reproduction_artifacts.zip` | Saved features, processed signals, released split CSVs and historical regression dictionaries required by the report notebook. Extract into the repository root. |
| `requirements.txt` | Original repository dependency list; versions are mostly unpinned. |
| `models/`, `linearprobing/`, `preprocessing/`, `baselines/` | Original PaPaGei implementation used by the reproduction. Keep these packages in the repository. |
| `README_upstream.md` and `LICENSE` | Original project documentation and licence. |

## Experimental setups

| Setup | Training | Validation | Test | Main differences |
| --- | ---: | ---: | ---: | --- |
| Original 205 participants | 123 | 41 | 41 | Released participant lists; Ridge alpha includes 1000; scaling before CV. |
| 205-participant rerun | 123 | 41 | 41 | Same split, with Ridge alpha restricted to 0.1, 1, 10 and 100. |
| Adapted 219 participants | 131 | 44 | 44 | Six within-participant duplicate recordings removed; participants sharing a recording grouped together; scaling inside grouped CV. |

Unless a section explicitly reports separate splits, evaluation combines validation and test participants: **82 participants for the 205 setup and 88 for the 219 setup**. Hyperparameter selection uses training participants only. The adapted experiment changes several factors together; it does not isolate the effect of adding participants.

Classification uses `Normal = 0`; prehypertension and both hypertension stages are grouped as `non-normal = 1`. The final classification baseline table uses only the 205 setup.

## 1. Environment

The work was run on CPU using the Conda environment `papagei_env`. The supplied report notebook records Python **3.10.21**. Use that working environment when available.

A starting installation for a new environment is:

```bash
conda create -n papagei_env python=3.10.21
conda activate papagei_env
python -m pip install -r requirements.txt
python -m pip install "pyPPG==1.0.41" "matplotlib-inline==0.1.6" jupyterlab ipykernel jinja2
python -m ipykernel install --user --name papagei_env --display-name "Python (papagei_env)"
```

This is a dependency-installation recipe, **not a version-locked recreation of the original environment**. Dependency versions can affect preprocessing and fitted results. Before archiving the final experiment, export the actual working environment:

```bash
conda env export --no-builds > environment.yml
python -m pip freeze > requirements-reproduction-lock.txt
```

These exports should be made from the environment that successfully runs the notebook, not from a newly installed environment that has not been checked. If an `environment.yml` from that run is included, it can be recreated with `conda env create -f environment.yml`.

The report notebook uses saved MOMENT embeddings and does not need to download the MOMENT model. Repeating MOMENT feature extraction in the original development workflow additionally requires `momentfm` and the pretrained model `AutonLab/MOMENT-1-large`.

## 2. Data, weights and saved inputs

### Raw PPG-BP data

Download [PPG-BP from Figshare](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299). Extract it so these paths exist relative to the repository root:

- `data_PPG/Data File/PPG-BP dataset.xlsx`
- `data_PPG/Data File/0_subject/2_1.txt` and the remaining raw recordings.

The raw data are required even in saved-results mode: the notebook inspects a waveform, reads participant metadata and checks recording hashes. Raw data are not bundled in the saved-artifact archive.

### Saved reproduction artefacts

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

### PaPaGei checkpoints

Download `papagei_s.pt` and `papagei_p.pt` from the [authors' Zenodo record](https://zenodo.org/records/13983110), and place them under `weights/`.

Checkpoints are required to regenerate embeddings; they are not required when loading all saved embeddings. The unused S-sVRI variant is not needed for the main report notebook. Raw data and downloaded checkpoints remain outside Git through `.gitignore`.

## 3. Display the reported results

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

## 4. Recompute individual experiments

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

## 5. Results compared with the paper

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

## 6. The custom helpers

### `evaluation_utils.evaluate_features()`

Fits a `StandardScaler`–`Ridge` pipeline with scaling inside four-fold, unshuffled KFold CV. The search tests alpha `[0.1, 1, 10, 100]` and solvers `auto`, `cholesky`, `sparse_cg`, selecting by negative MSE. It returns MAE, predictions, selected parameters and a 95% MAE interval from 500 participant bootstrap samples, with seed 42 by default.

The caller supplies already separated training and evaluation arrays. The helper **does not create participant splits or enforce shared-recording groups**. `evaluation_split` labels the supplied evaluation; it does not choose rows automatically. The grouped adapted experiments use their own notebook code.

### `results_utils.record_result()`

Prints a regression summary and appends one row to a CSV, including run ID, model, target, procedure, sample counts, MAE, confidence interval and selected parameters. It checks the existing CSV schema before appending. Calling it repeatedly appends repeated rows; it does not replace an existing run. It does not fit models or save the complete predictions dictionary.

## 7. Interpretation and reproducibility limits

- The released split omits 14 labelled participants. Their available recordings and labels do not establish the reason for exclusion.
- One exact recording is shared by participants 23 and 24 across the original training/validation split. The adapted setup keeps those participants together.
- The 205 statistical baseline uses duplicate-cleaned padded recordings, whereas the original PaPaGei embeddings retain the released recording handling. This limits strict like-for-like interpretation.
- Padding occupies much of each short recording and can dominate simple amplitude statistics.
- Bootstrap intervals describe evaluation-sample uncertainty for fixed fitted models, not variation from retraining on new splits.
- The historical RF classification run was unseeded. The verification branch uses seed 42 and may not recover the same score.
- Some historical regression bootstraps were not seeded. Exact intervals can change on recomputation.
- This README and archive were checked against the submitted project snapshot. A fresh environment installation and full model-training workflow have not been validated end to end.

## Attribution and AI assistance

The implementation is based on [Nokia Bell Labs' PaPaGei repository](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) and [the PaPaGei paper](https://arxiv.org/abs/2410.20542). Original documentation is retained in `README_upstream.md`; original licence and copyright notices are preserved in `LICENSE` and the source files. MOMENT embeddings use `AutonLab/MOMENT-1-large`.

ChatGPT (OpenAI) assisted with Python code development and debugging, notebook organisation and comments. Responsibility for the submitted analysis and conclusions remains with the project author. 
