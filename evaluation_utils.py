from pathlib import Path
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error


def evaluate_features(
    X_train,
    y_train,
    X_eval,
    y_eval,
    evaluation_split="val",
    bootstrap_seed=42,
):
    """Shared Ridge evaluation for embeddings or improved features."""

    if evaluation_split not in ["val", "test"]:
        raise ValueError("evaluation_split must be 'val' or 'test'")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    search = GridSearchCV(
        pipeline,
        param_grid={
            "ridge__alpha": [0.1, 1.0, 10.0, 100.0],
            "ridge__solver": ["auto", "cholesky", "sparse_cg"]
        },
        cv=KFold(n_splits=4, shuffle=False),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        error_score="raise"
    )

    search.fit(X_train, y_train)

    y_true = np.asarray(y_eval).reshape(-1)
    y_pred = np.asarray(search.predict(X_eval)).reshape(-1)

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.choice(
        len(y_true),
        size=(500, len(y_true)),
        replace=True
    )

    bootstrap_maes = np.abs(y_true - y_pred)[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_maes, [2.5, 97.5])

    return {
        "parameters": {
            key.replace("ridge__", ""): value
            for key, value in search.best_params_.items()
        },
        "mae": mean_absolute_error(y_true, y_pred),
        "lower_bound_mae": float(lower),
        "upper_bound_mae": float(upper),
        # Retained for compatibility with record_result().
        "y_test": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "evaluation_split": evaluation_split,
        "n_train": len(y_train),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": 500
    }