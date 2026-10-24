from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd


def record_result(
    result,
    model_name,
    target,
    n_train,
    procedure,
    csv_path="results/evaluation_log.csv",
    run_id=None,
):
    """Print an evaluation summary and append it to a CSV."""

    targets = {
        "diasbp": ("Diastolic BP", "mmHg"),
        "sysbp": ("Systolic BP", "mmHg"),
        "hr": ("Heart rate", "beats/min"),
    }

    target_name, units = targets[target]
    parameters = result.get("parameters", {})

    row = {
        "run_id": run_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        ),
        "model": model_name,
        "target": target,
        "target_name": target_name,
        "units": units,
        "procedure": procedure,
        "n_train": n_train,
        "n_evaluation": len(result["y_test"]),
        "mae": float(result["mae"]),
        "ci_lower": float(result["lower_bound_mae"]),
        "ci_upper": float(result["upper_bound_mae"]),
        "parameters": json.dumps(parameters, default=str),
    }

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_has_data = path.exists() and path.stat().st_size > 0

    if file_has_data:
        existing_columns = pd.read_csv(path, nrows=0).columns.tolist()

        if existing_columns != list(row):
            raise ValueError(
                "Existing CSV has different columns. "
                "Choose a new csv_path."
            )

    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not file_has_data,
        index=False,
    )

    print(f"\n{model_name} — {target_name}")
    print(f"MAE: {row['mae']:.4f} {units}")
    print(
        f"95% CI: [{row['ci_lower']:.4f}, "
        f"{row['ci_upper']:.4f}] {units}"
    )
    print(f"Settings: {parameters}")
    print(f"Evaluation participants: {row['n_evaluation']}")
    print(f"Saved to: {path.resolve()}")

    return row