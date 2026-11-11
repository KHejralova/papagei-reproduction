"""
The models compared in experiment 2, and how each one is trained.

Every method is described by a "feature set" spec:
  {"kind": "ridge", "direct": [...blocks...], "stacked": <embedding> or None}
      a ridge model on some feature blocks; with "stacked", a first ridge on
      that embedding makes one "PPG score" that is added as a feature
  {"kind": "aeon", "method": "rocket" | "minirocket" | "hydra" | "quant"}
      a standard time series regression method from the course (Lab 8),
      trained on the raw preprocessed PPG

(Named estimators.py, not models.py, because the repo already has a
`models` package holding PaPaGei's network.)
"""

import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
AEON_METHODS = ["rocket", "minirocket", "hydra", "quant"]
SIMPLE_BLOCKS = ["demo", "morph", "peak_hr"]


def feature_sets(embeddings, aeon_methods):
    """All methods compared in experiment 2: name -> spec."""
    def ridge(direct, stacked=None):
        return {"kind": "ridge", "direct": direct, "stacked": stacked}
    sets = {
        "demo": ridge(["demo"]),
        "morph": ridge(["morph"]),
        "demo+morph": ridge(["demo", "morph"]),
        "peak_hr": ridge(["peak_hr"]),
        "handcrafted": ridge(SIMPLE_BLOCKS),
    }
    for e in embeddings:
        sets[e] = ridge([e])
        sets[f"{e}+demo_concat"] = ridge([e, "demo"])
        sets[f"{e}+demo_stacked"] = ridge(["demo"], e)
        sets[f"{e}+handcrafted_stacked"] = ridge(SIMPLE_BLOCKS, e)
    for m in aeon_methods:
        sets[m] = {"kind": "aeon", "method": m}
    return sets


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def ridge_search():
    """Scaler + ridge, alpha chosen by 4-fold CV on the training rows.

    Identical settings to evaluation_utils.evaluate_features(), so every
    ridge model here is the same as in the reproduction."""
    return GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
        param_grid={"ridge__alpha": [0.1, 1.0, 10.0, 100.0],
                    "ridge__solver": ["auto", "cholesky", "sparse_cg"]},
        cv=KFold(n_splits=4, shuffle=False),
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )


def make_aeon(method):
    """aeon defaults, as in Lab 8 (one thread each; folds run in parallel)."""
    from aeon.regression.convolution_based import (
        HydraRegressor,
        MiniRocketRegressor,
        RocketRegressor,
    )
    from aeon.regression.interval_based import QUANTRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    if method == "rocket":
        return RocketRegressor(n_kernels=10000, random_state=SEED, n_jobs=1)
    if method == "minirocket":
        return MiniRocketRegressor(random_state=SEED, n_jobs=1)
    if method == "hydra":
        return HydraRegressor(random_state=SEED, n_jobs=1)
    if method == "quant":
        # QUANTRegressor's default head, with a fixed seed.
        return QUANTRegressor(
            estimator=ExtraTreesRegressor(n_estimators=200, max_features=0.1,
                                          random_state=SEED, n_jobs=1),
            random_state=SEED)
    raise ValueError(method)


# --------------------------------------------------------------------------
# Building the input table
# --------------------------------------------------------------------------

def impute_like_train(X_tr, X_ev):
    """Fill NaNs with the training-column median (fitted on train only)."""
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return (np.where(np.isnan(X_tr), med, X_tr),
            np.where(np.isnan(X_ev), med, X_ev))


def stacked_score(E_tr, y_tr, E_ev):
    """The PPG score: a ridge prediction from the embedding alone.

    Training rows get out-of-fold predictions (each from a model that did
    not see that row), so the second model sees how good the score really
    is on new people. Evaluation rows get the prediction of a model fitted
    on all training rows."""
    oof = cross_val_predict(ridge_search(), E_tr, y_tr,
                            cv=KFold(n_splits=4, shuffle=False))
    ev = ridge_search().fit(E_tr, y_tr).predict(E_ev)
    return oof[:, None], ev[:, None]


def build_matrices(spec, blocks, tr, ev, y_tr, score_cache=None):
    """Training and evaluation tables (rows tr and ev) for a ridge spec."""
    parts_tr, parts_ev = [], []
    for name in spec["direct"]:
        a, b = impute_like_train(blocks[name][tr], blocks[name][ev])
        parts_tr.append(a)
        parts_ev.append(b)
    e = spec["stacked"]
    if e is not None:
        if score_cache is not None and e in score_cache:
            s_tr, s_ev = score_cache[e]
        else:
            s_tr, s_ev = stacked_score(blocks[e][tr], y_tr, blocks[e][ev])
            if score_cache is not None:
                score_cache[e] = (s_tr, s_ev)
        parts_tr.append(s_tr)
        parts_ev.append(s_ev)
    return np.hstack(parts_tr), np.hstack(parts_ev)


def fit_predict(spec, data, tr, ev, y_tr, score_cache=None):
    """Train one method on participants tr, predict participants ev.

    data = {"blocks": {name: (n, k) array}, "raw": (n, 3, length) array}.
    Returns (predictions, fit + predict seconds); feature extraction is not
    included. score_cache lets several stacked sets share one PPG score.
    """
    t0 = time.perf_counter()
    if spec["kind"] == "ridge":
        X_tr, X_ev = build_matrices(spec, data["blocks"], tr, ev, y_tr,
                                    score_cache)
        pred = ridge_search().fit(X_tr, y_tr).predict(X_ev)
    else:
        # Each recording is one series; a participant's prediction is the
        # mean over their 3 recordings, like PaPaGei's averaged embedding.
        raw = data["raw"]
        n_rec = raw.shape[1]
        X_tr = raw[tr].reshape(-1, 1, raw.shape[2])
        X_ev = raw[ev].reshape(-1, 1, raw.shape[2])
        model = make_aeon(spec["method"]).fit(X_tr, np.repeat(y_tr, n_rec))
        pred = model.predict(X_ev).reshape(len(ev), n_rec).mean(axis=1)
    return pred, time.perf_counter() - t0


def warm_up(sets, raw, tr):
    """Fit each aeon method once on a few series so one-off compilation
    (numba / torch) isn't counted as runtime (Lab 7-8 timing rule)."""
    for spec in sets.values():
        if spec["kind"] == "aeon":
            X = raw[tr[:10]].reshape(-1, 1, raw.shape[2])
            make_aeon(spec["method"]).fit(X, np.arange(len(X), dtype=float))
