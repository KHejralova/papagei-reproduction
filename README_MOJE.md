## Regression evaluation

Use `evaluation_utils.py` to evaluate features with the same
regression settings:

- StandardScaler and Ridge in a pipeline, with scaling fitted inside CV.
- Four-fold cross-validation without shuffling.
- Alpha values: [0.1, 1, 10, 100], following the paper.
- Solvers: auto, cholesky, sparse_cg.
- Settings selected using cross-validation MSE.
- Performance reported using participant-level MAE.
- 95% bootstrap confidence interval: 500 resamples, seed 42.

Use the supplied participant splits:
123 training, 41 validation and 41 test participants.
Keep all recordings from each participant in the same split.

Use validation results to choose improvements. Evaluate on test
participants after those choices are fixed. 

Evaluation_utils does all that, just to we use the same stuff. 

### running evaluation

Each feature matrix must have one row per participant. Target values
must match the participant order in the corresponding matrix.

Import the evaluator and result recorder:

```python
from evaluation_utils import evaluate_features
from results_utils import record_result

result = evaluate_features(
    X_train=X_train,
    y_train=y_train,
    X_eval=X_val,
    y_eval=y_val,
    evaluation_split="val",
    bootstrap_seed=42
)

record_result(
    result=result,
    model_name="papagei_s_plus_derivative_features",
    target="diasbp",
    n_train=len(y_train),
    procedure="shared_pipeline_paper_grid_validation_seed42",
    csv_path="results/improvement_results.csv"
)
```

### Saving results

`results_utils.py` has `record_result()`, which prints the result
and appends it to a CSV that can be opened in Excel.

My results are in `results/evaluation_log.csv`, do something like `results/improvement_results.csv`.

Do procedure labels so that we know + every recording call appends a row, including repeated runs.

The recorder saves summary metrics (save the full returned dictionary
separately if you need more)

### Important distinction

Notebook (the example_papagei.ipynb) combines validation/test evaluation and an alpha grid
containing 1000, which is not what the paper reported - have separate labels for them in the csv

This is just so we have the same 'pipeline' for comparison, It does not do the exact
reconstruction of the publsihed evaluation, but is close enough

