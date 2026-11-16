"""
How methods are evaluated and compared.
- repeated_cv: repeated k-fold cross-validation over participants. Every
  method sees exactly the same folds, so results are paired.
- per_repeat_metrics: MAE, RMSE and R^2 over all held-out predictions of each repeat.
- summarise: mean metrics, gain vs the Dummy (training-mean) baseline, and
  paired comparisons against reference methods with the corrected resampled
  t-test plus Holm correction.
- result_from_predictions: hold-out results in the same format (and with the
  same bootstrap) as evaluation_utils.evaluate_features().
- measure_extraction_cost: time to turn a participant's recordings into
  features.
"""

import time
import warnings

import data as D
import features as F
import numpy as np
import pandas as pd
from embeddings import build_model, parse_embedding, participant_embeddings
from estimators import SEED, fit_predict, warm_up
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import RepeatedKFold


# Repeated cross-validation
def run_fold(r, k, tr, ev, data, targets, sets, ids):
    """Fit and score every method on one fold, for every target."""
    # r = repeat number, k = fold number, tr / ev = row numbers of the
    # training / held-out people in this fold
    warnings.filterwarnings("ignore")
    # compile the ROCKET-family code once so it isn't counted in the timing of every fold
    if data["raw"] is not None:
        warm_up(sets, data["raw"], tr)
    fold_rows, preds = [], []
    for t in D.TARGETS: # for BP, diastolic BP and heart rate
        y_tr, y_ev = targets[t][tr], targets[t][ev]
        # the dummy baseline -> predict the training mean for everyone
        t0 = time.perf_counter()
        pred_all = {"training_mean": np.full(len(ev), y_tr.mean())}
        times = {"training_mean": time.perf_counter() - t0}
        # one shared PPG score per fold so the stacked methods don't have to recompute it
        score_cache = {}
        # training every method on the training people, predicting the held-out people
        for name, spec in sets.items():
            pred_all[name], times[name] = fit_predict(spec, data, tr, ev,
                                                      y_tr, score_cache)
        for name, pred in pred_all.items():
            # one summary row per method - errors on this fold + how long it took
            fold_rows.append({
                "repeat": r, "fold": k, "target": t, "feature_set": name,
                "mae": float(np.mean(np.abs(y_ev - pred))),
                "rmse": float(np.sqrt(np.mean((y_ev - pred) ** 2))),
                "n_eval": len(ev), "fit_predict_s": times[name]})
            # and every single prediction kept for the plots and per-repeat metrics
            preds.append(pd.DataFrame({
                "repeat": r, "fold": k, "target": t, "feature_set": name,
                "subject_ID": np.asarray(ids)[ev], "y_true": y_ev,
                "y_pred": pred}))
    return fold_rows, preds


def repeated_cv(data, targets, sets, ids, n_splits=5, n_repeats=10,
                n_jobs=-1):
    """Run every method on the same repeated k-fold splits of participants.

    Returns (folds, predictions): one row per fold x target x method with
    MAE / RMSE / time, and every held-out prediction."""
    # 10 repears of 5-fold CV = 50 folds, adding fixed seed so that all runs use the same splits
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                        random_state=SEED)
    # list of all 50 folds
    tasks = [(i // n_splits, i % n_splits, tr, ev)
             for i, (tr, ev) in enumerate(rkf.split(np.arange(len(ids))))]
    # run the folds in parallel
    out = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_fold)(r, k, tr, ev, data, targets, sets, ids)
        for r, k, tr, ev in tasks)
    # put the results of all folds together into two tables
    folds = pd.DataFrame([row for rows, _ in out for row in rows])
    preds = pd.concat([p for _, ps in out for p in ps], ignore_index=True)
    return folds, preds


# Metrics and paired comparisons
def per_repeat_metrics(preds):
    """MAE, RMSE, R^2 over all held-out predictions of each repeat."""
    # each repear predicts every person exactly once, so one repeat = one full set of predictions for all the targets
    rows = []
    for key, g in preds.groupby(["target", "feature_set", "repeat"]):
        err = g["y_true"] - g["y_pred"]
        # total spread of the true values, needed for R^2
        sst = np.sum((g["y_true"] - g["y_true"].mean()) ** 2)
        rows.append(dict(zip(["target", "feature_set", "repeat"], key),
                         mae=np.mean(np.abs(err)), # avg size of the error
                         rmse=np.sqrt(np.mean(err ** 2)), # like MAE but big errors count more
                         r2=1 - np.sum(err ** 2) / sst)) # share of the spread explained (1 = perfect, 0 = no better than the mean)
    return pd.DataFrame(rows)


def corrected_ttest(diff, n_splits):
    """Nadeau & Bengio (2003) corrected resampled t-test on per-fold
    differences. The usual t-test assumes independent folds; CV folds share
    most of their training data, so the variance is inflated by
    n_test / n_train to avoid over-confident p-values."""
    # diff = MAE of method A minus MAE of method B, one value per fold
    diff = np.asarray(diff, dtype=float)
    J = len(diff)
    rho = 1.0 / (n_splits - 1)  # n_test / n_train for k-fold
    var = np.var(diff, ddof=1)
    if var == 0: # identical results on every fold - nothing to test
        return np.nan
    # the correction to make the test more cautious
    t = diff.mean() / np.sqrt((1.0 / J + rho) * var)
    # two-sided p-value from the t-distribution
    return float(2 * stats.t.sf(abs(t), df=J - 1))


def holm(pvals):
    """Holm step-down adjusted p-values (NaNs ignored): the smallest p is
    multiplied by m, the next by m - 1, and so on, keeping the order."""
    # needed because we compare many methods at once - the more tests, the more likely one looks significant by chance
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p) # skip the reference compared with itself
    idx = np.argsort(p[ok]) # smallest p first, largest last
    m = ok.sum() # number of comparisons
    # multiply the smallest p by m, the next by m - 1, and so on
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    out_ok = np.empty(m)
    out_ok[idx] = np.minimum(adj, 1.0) # a p-value can't be larger than 1
    out[ok] = out_ok
    return out


def summarise(folds, preds, references, n_splits):
    """Per-repeat metrics and one summary row per target x method."""
    per_rep = per_repeat_metrics(preds)
    # gain vs dummy -> what % of the dummy's error each method removes, per repeat
    dummy = (per_rep[per_rep.feature_set == "training_mean"]
             .set_index(["target", "repeat"])["mae"])
    per_rep["gain_vs_dummy_pct"] = 100 * (
        1 - per_rep["mae"] / dummy.loc[list(zip(per_rep.target,
                                               per_rep.repeat))].to_numpy())
    # average fit + predict time per fold for each method
    times = folds.groupby(["target", "feature_set"])["fit_predict_s"].mean()

    # one row per target x method - averages over the repeats
    summary = pd.DataFrame([{
        "target": t, "feature_set": s,
        "mae_mean": g["mae"].mean(), "mae_sd_over_repeats": g["mae"].std(ddof=1),
        "rmse_mean": g["rmse"].mean(), "r2_mean": g["r2"].mean(),
        "gain_vs_dummy_pct": g["gain_vs_dummy_pct"].mean(),
        "fit_predict_s_per_fold": times.loc[(t, s)]}
        for (t, s), g in per_rep.groupby(["target", "feature_set"])])

    # Paired comparisons on folds vs each reference, Holm-corrected over all methods compared with that reference for that target
    for ref in references: # e.g. "papagei_s" or "handcrafted"
        for t in D.TARGETS:
            ft = folds[folds.target == t]
            # the reference's MAE on every fold in a fixed order
            other = (ft[ft.feature_set == ref]
                     .sort_values(["repeat", "fold"])["mae"].to_numpy())
            idx = summary.index[summary.target == t]
            ps = []
            for i in idx:
                s = summary.at[i, "feature_set"]
                # thius method's MAE on the same folds, same order -> paired
                mine = (ft[ft.feature_set == s]
                        .sort_values(["repeat", "fold"])["mae"].to_numpy())
                d = mine - other # negative = this method is better on that fold
                p = corrected_ttest(d, n_splits) if s != ref else np.nan
                summary.at[i, f"diff_vs_{ref}"] = d.mean()
                summary.at[i, f"p_vs_{ref}"] = p
                summary.at[i, f"pct_folds_better_than_{ref}"] = \
                    100 * np.mean(d < 0)
                ps.append(p)
            # correct all p-values for this referece + target together
            summary.loc[idx, f"p_holm_vs_{ref}"] = holm(ps)
    return per_rep, summary.sort_values(["target", "mae_mean"])


# Hold-out results 
def result_from_predictions(y_true, y_pred, split, params, seed=42):
    """Same result format and bootstrap as evaluate_features()."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    # bootstrap - resample the evaluation people with replacement 500 times and recompute the MAE each time
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=(500, len(y_true)), replace=True)
    boot = np.abs(y_true - y_pred)[idx].mean(axis=1)
    # the middle 95% of those 500 MAEs = the 95% confidence interval
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"parameters": params,
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "lower_bound_mae": float(lo), "upper_bound_mae": float(hi),
            "y_test": y_true.tolist(), "y_pred": y_pred.tolist(),
            "evaluation_split": split}


def training_mean_result(y_train, y_eval, split, seed=42):
    """The Dummy baseline: predict the training mean for everyone."""
    mean = float(np.mean(y_train))
    return result_from_predictions(y_eval, np.full(len(y_eval), mean), split,
                                   {"constant": mean}, seed)


# Feature extraction cost
def measure_extraction_cost(data_dir, subject_ids, embeddings,
                            weights_dir="weights", device="cpu", reps=3):
    """Median time to turn one participant's 3 raw recordings into features (preprocessing included), on this machine."""
    ids = list(subject_ids)
    n = len(ids)
    # load the raw files first so reading from disk isn't counted in the timing
    raws = {i: [D.load_raw_recording(data_dir, i, r) for r in (1, 2, 3)]
            for i in ids}

    def prep():
        return {i: [D.preprocess_recording(x) for x in raws[i]] for i in ids}

    # run a step reps times and keep the median, divided by the number of participants to get seconds per participant
    def median_time(fn):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return np.median(times) / n

    rows = [("preprocessing (filter + resample)", median_time(prep))]
    recs = prep()
    rows.append(("handcrafted features (sVRI, SQI, IPA, peak HR)",
                 median_time(lambda: [
                     (F.morphology_features(recs[i], D.FS_TARGET),
                      F.peak_hr_feature(recs[i], D.FS_TARGET))
                     for i in ids])))
    for e in embeddings:
        model_name, variant = parse_embedding(e)
        model = build_model(model_name, weights_dir, device)
        participant_embeddings(model, variant, recs, ids[:2], device)  # warm-up run, not timed
        rows.append((f"{e} embedding", median_time(
            lambda: participant_embeddings(model, variant, recs, ids,
                                           device))))
    # seconds -> milliseconds per participant, easier to read
    df = pd.DataFrame(rows, columns=["step", "seconds_per_participant"])
    df["ms_per_participant"] = 1000 * df["seconds_per_participant"]
    return df
