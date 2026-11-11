"""
Improvement experiment 2: what does PaPaGei add beyond simple information
about the person, and how should the two be combined?

Feature sets (one row per participant):

  demo                      age, sex, height, weight, BMI
  morph                     sVRI and SQI (mean over the 3 recordings) - two of
                            the three indices PaPaGei-S is trained on. IPA is
                            left out: it can't be computed reliably on 2.1 s
                            recordings (see ppg_features.py)
  demo+morph                the paper's "Demo + PPG" baseline (Table 15)
  peak_hr                   heart rate from the interval between systolic
                            peaks (one number; a classic signal-processing
                            baseline for the HR target)
  handcrafted               demo + morph + peak_hr (no deep learning)
  <emb>                     PaPaGei embedding alone (zero padding, as in the
                            paper and the reproduction)
  <emb>+demo_concat         early fusion: 512 embedding dims + 5 demo columns
                            in one ridge (what the paper's Table 15 does)
  <emb>+demo_stacked        late fusion: a ridge on the embedding produces one
                            "PPG score" (out-of-fold on the training rows),
                            which is used next to the demo columns in a
                            second ridge. Stops 512 embedding dims from
                            drowning 5 demographic columns.
  <emb>+handcrafted_stacked the same, with all handcrafted features

Standard time series regression baselines from the course (Lab 8), trained
on the raw preprocessed PPG (each 2.1 s recording is one univariate series,
predictions are averaged over a participant's 3 recordings, like PaPaGei's
embeddings):

  rocket, minirocket, hydra, quant   aeon's RocketRegressor,
                            MiniRocketRegressor, HydraRegressor, QUANTRegressor

<emb> is papagei_s and papagei_p. "Hypertension" and the disease columns are
never used as features (hypertension is defined from BP, i.e. leakage).
Every ridge model is the shared pipeline (StandardScaler + Ridge, paper alpha
grid, 4-fold CV, scaler inside CV), exactly as in evaluate_features().

Participant 24 is left out by default (--drop-ids): one of its recordings is
identical to one of participant 23's, so the two could end up on opposite
sides of a split (leakage through duplicates, Week 7 lecture).

Modes (--mode):

  cv    (default, for choosing) repeated 5-fold CV over the train+val
        participants (default 10 repeats = 50 held-out folds). The test set is
        not touched. Reports MAE, RMSE, R^2, gain vs the Dummy (training-mean)
        baseline and fit+predict time; compares sets on the same folds with
        the corrected resampled t-test (Nadeau & Bengio 2003) plus Holm
        correction, and draws a critical difference diagram and a multiple
        comparison matrix (Friedman + Wilcoxon-Holm, as in Labs 7-8).
  val   train on the training participants, evaluate on validation.
  figures  redraw the CD diagrams and MCMs from a saved cv run (fast).
  timing   only measure feature extraction cost per participant.
  test  retrain on training + validation participants, evaluate ONCE on the
        test participants (Week 7: "retrain on train + validation, evaluate
        once on test"). Run only after the choices are frozen.

Example (from the repo root):
    python improvements/02_feature_fusion.py --data-dir data_PPG
"""

import argparse
import importlib.util
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RepeatedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ppg_features as pf
import ppgbp_common as pc

from evaluation_utils import evaluate_features
from results_utils import record_result

_spec = importlib.util.spec_from_file_location(
    "exp1", HERE / "01_input_representation.py")
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)

DEMO_COLS = ["age", "sex", "height", "weight", "bmi"]
TARGETS = ["sysbp", "diasbp", "hr"]
UNITS = {"sysbp": "mmHg", "diasbp": "mmHg", "hr": "bpm"}
EMBEDDINGS = ["papagei_s", "papagei_p"]
AEON_METHODS = ["rocket", "minirocket", "hydra", "quant"]
SEED = 42
RAW_LENGTH = 263  # samples per recording at 125 Hz (2.1 s); longer ones cropped
# Short names for the CD diagrams and MCMs.
DISPLAY = {"training_mean": "Dummy (mean)", "demo": "Demographics",
           "handcrafted": "Simple features", "papagei_s": "PaPaGei-S",
           "papagei_p": "PaPaGei-P", "papagei_s+demo_concat": "S + demo concat",
           "papagei_s+demo_stacked": "S + demo stacked",
           "papagei_s+handcrafted_stacked": "S + simple stacked",
           "papagei_p+handcrafted_stacked": "P + simple stacked",
           "rocket": "ROCKET", "minirocket": "MiniROCKET", "hydra": "Hydra",
           "quant": "QUANT", "peak_hr": "Peak HR", "morph": "sVRI + SQI"}
DEFAULT_STAT_SETS = ["training_mean", "demo", "handcrafted", "papagei_s",
                     "papagei_s+demo_concat", "papagei_s+handcrafted_stacked",
                     "rocket", "minirocket", "hydra", "quant"]


def parse_embedding(name):
    """'papagei_s' -> ('papagei_s', 'zero_pad'); 'papagei_p@tile_cycles' ->
    ('papagei_p', 'tile_cycles')."""
    model_name, _, variant = name.partition("@")
    variant = variant or "zero_pad"
    if model_name not in pc.MODEL_FILES:
        raise SystemExit(f"Unknown model in --embeddings: {name}")
    if variant not in exp1.ALL_VARIANTS:
        raise SystemExit(f"Unknown input representation in --embeddings: "
                         f"{name} (choose from {exp1.ALL_VARIANTS})")
    return model_name, variant


def feature_sets(embeddings, aeon_methods):
    """name -> spec dict."""
    def ridge(direct, stacked=None):
        return {"kind": "ridge", "direct": direct, "stacked": stacked}
    sets = {
        "demo": ridge(["demo"]),
        "morph": ridge(["morph"]),
        "demo+morph": ridge(["demo", "morph"]),
        "peak_hr": ridge(["peak_hr"]),
        "handcrafted": ridge(["demo", "morph", "peak_hr"]),
    }
    for e in embeddings:
        sets[e] = ridge([e])
        sets[f"{e}+demo_concat"] = ridge([e, "demo"])
        sets[f"{e}+demo_stacked"] = ridge(["demo"], e)
        sets[f"{e}+handcrafted_stacked"] = ridge(["demo", "morph", "peak_hr"],
                                                 e)
    for m in aeon_methods:
        sets[m] = {"kind": "aeon", "method": m}
    return sets


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def ridge_search():
    """Identical settings to evaluation_utils.evaluate_features()."""
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


def impute_like_train(X_tr, X_ev):
    """Fill NaNs with the training-column median (fitted on train only)."""
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return (np.where(np.isnan(X_tr), med, X_tr),
            np.where(np.isnan(X_ev), med, X_ev))


def stacked_score(E_tr, y_tr, E_ev):
    """Out-of-fold ridge predictions on train, full-train fit on eval rows."""
    oof = cross_val_predict(ridge_search(), E_tr, y_tr,
                            cv=KFold(n_splits=4, shuffle=False))
    ev = ridge_search().fit(E_tr, y_tr).predict(E_ev)
    return oof[:, None], ev[:, None]


def build_matrices(spec, blocks, tr, ev, y_tr, score_cache=None):
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
    """Fit one feature set on rows tr, predict rows ev. Returns predictions
    and fit+predict time in seconds (feature extraction not included)."""
    t0 = time.perf_counter()
    if spec["kind"] == "ridge":
        X_tr, X_ev = build_matrices(spec, data["blocks"], tr, ev, y_tr,
                                    score_cache)
        pred = ridge_search().fit(X_tr, y_tr).predict(X_ev)
    else:
        raw = data["raw"]  # (n_participants, 3, RAW_LENGTH)
        n_rec = raw.shape[1]
        X_tr = raw[tr].reshape(-1, 1, raw.shape[2])
        X_ev = raw[ev].reshape(-1, 1, raw.shape[2])
        model = make_aeon(spec["method"]).fit(X_tr, np.repeat(y_tr, n_rec))
        pred = model.predict(X_ev).reshape(len(ev), n_rec).mean(axis=1)
    return pred, time.perf_counter() - t0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def apply_drop(splits, drop_ids):
    if not drop_ids:
        return splits
    out = {}
    for name, df in splits.items():
        gone = df["subject_ID"].isin(drop_ids)
        if gone.any():
            print(f"Leaving out participant(s) "
                  f"{df.loc[gone, 'subject_ID'].tolist()} from {name}")
        out[name] = df[~gone].reset_index(drop=True)
    return out


def load_blocks(args, splits, split_names, embeddings, need_raw):
    """Feature blocks for the participants of the given splits, in order."""
    meta = pd.concat([splits[s] for s in split_names], ignore_index=True)
    ids = meta["subject_ID"].astype(int).tolist()
    features_dir = Path(args.features_dir or
                        Path(args.data_dir) / "features_improvements")
    features_dir.mkdir(parents=True, exist_ok=True)

    demo = meta[DEMO_COLS].copy()
    demo["sex"] = demo["sex"].map({"Male": 1.0, "Female": 0.0})
    if demo.isna().any().any():
        raise ValueError("Unexpected missing demographic values")
    blocks = {"demo": demo.to_numpy(float)}

    recordings = None

    def get_recordings():
        nonlocal recordings
        if recordings is None:
            print(f"Preprocessing {len(ids) * 3} recordings ...")
            recordings = pc.load_preprocessed_recordings(args.data_dir, ids)
        return recordings

    # Handcrafted features (cached).
    cols = pf.ALL_MORPH_NAMES + ["peak_hr"]
    cache_csv = features_dir / "handcrafted_svri_ipa_sqi_peakhr.csv"
    cached = (pd.read_csv(cache_csv).set_index("subject_ID")
              if cache_csv.exists() else pd.DataFrame())
    missing = [i for i in ids if i not in cached.index]
    if missing:
        recs = get_recordings()
        new = pd.DataFrame(
            [list(pf.morphology_features(recs[i], pc.FS_TARGET)) +
             [pf.peak_hr_feature(recs[i], pc.FS_TARGET)] for i in missing],
            index=pd.Index(missing, name="subject_ID"), columns=cols)
        cached = pd.concat([cached, new])
        cached.to_csv(cache_csv)
    blocks["morph"] = cached.loc[ids, pf.MORPH_NAMES].to_numpy(float)
    blocks["peak_hr"] = cached.loc[ids, ["peak_hr"]].to_numpy(float)
    used = pf.MORPH_NAMES + ["peak_hr"]
    n_fail = int(np.isnan(cached.loc[ids, used].to_numpy(float))
                 .any(axis=1).sum())
    n_ipa = int(cached.loc[ids, "ipa"].isna().sum())
    print(f"Handcrafted features: {n_fail} of {len(ids)} participants have a "
          f"missing value (filled with the training median). IPA (not used) "
          f"could not be computed for {n_ipa} of {len(ids)}.")

    # Embeddings: reuse experiment 1's caches, compute what's missing.
    for e in embeddings:
        model_name, variant = parse_embedding(e)
        cache = features_dir / f"{model_name}_pretrained_{variant}.npz"
        stored = dict(np.load(cache)) if cache.exists() else {}
        parts, model = [], None
        for s in split_names:
            want = splits[s]["subject_ID"].astype(int).tolist()
            # Reuse cached rows by participant ID (works after dropping IDs).
            if f"X_{s}" in stored:
                lookup = dict(zip(stored[f"ids_{s}"].tolist(),
                                  stored[f"X_{s}"]))
                if all(i in lookup for i in want):
                    parts.append(np.stack([lookup[i] for i in want]))
                    continue
            if model is None:
                model = pc.build_model(model_name, args.weights_dir,
                                       args.device)
            print(f"Embedding {model_name} / {variant} / {s} ...")
            X = exp1.participant_features(model, variant, get_recordings(),
                                          want, args.device)
            stored[f"X_{s}"], stored[f"ids_{s}"] = X, np.array(want)
            parts.append(X)
        np.savez(cache, **stored)
        blocks[e] = np.vstack(parts)

    raw = None
    if need_raw:
        recs = get_recordings()
        raw = np.stack([np.stack([r[:RAW_LENGTH] for r in recs[i]])
                        for i in ids]).astype(np.float64)
        n_long = sum(len(r) > RAW_LENGTH for i in ids for r in recs[i])
        if n_long:
            print(f"Raw series for ROCKET-family models: {n_long} longer "
                  f"recording(s) cropped to the first {RAW_LENGTH} samples")

    targets = {t: meta[t].to_numpy(float) for t in TARGETS}
    return ids, {"blocks": blocks, "raw": raw}, targets, get_recordings


# --------------------------------------------------------------------------
# Feature extraction cost
# --------------------------------------------------------------------------

def measure_extraction_cost(args, splits, embeddings, n=30, reps=3):
    """Median time to turn one participant's 3 raw recordings into features
    (preprocessing included), on this machine's CPU."""
    ids = splits["train"]["subject_ID"].astype(int).tolist()[:n]
    raws = {i: [pc.load_raw_recording(args.data_dir, i, r) for r in (1, 2, 3)]
            for i in ids}
    def prep():
        return {i: [pc.preprocess_recording(x) for x in raws[i]] for i in ids}

    rows = []
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        prep()
        times.append(time.perf_counter() - t0)
    t_prep = np.median(times) / n
    rows.append(("preprocessing (filter + resample)", t_prep))
    recs = prep()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for i in ids:
            pf.morphology_features(recs[i], pc.FS_TARGET)
            pf.peak_hr_feature(recs[i], pc.FS_TARGET)
        times.append(time.perf_counter() - t0)
    rows.append(("handcrafted features (sVRI, SQI, IPA, peak HR)",
                 np.median(times) / n))

    for e in embeddings:
        model_name, variant = parse_embedding(e)
        model = pc.build_model(model_name, args.weights_dir, args.device)
        exp1.participant_features(model, variant, recs, ids[:2], args.device)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            exp1.participant_features(model, variant, recs, ids, args.device)
            times.append(time.perf_counter() - t0)
        rows.append((f"{e} embedding", np.median(times) / n))
    df = pd.DataFrame(rows, columns=["step", "seconds_per_participant"])
    df["ms_per_participant"] = 1000 * df["seconds_per_participant"]
    return df


# --------------------------------------------------------------------------
# Mode: repeated CV on train+val
# --------------------------------------------------------------------------

def run_fold(r, k, tr, ev, data, targets, sets, ids):
    warnings.filterwarnings("ignore")
    # Warm up compiled code (numba / torch) so it isn't counted as runtime.
    for name, spec in sets.items():
        if spec["kind"] == "aeon":
            X = data["raw"][tr[:10]].reshape(-1, 1, data["raw"].shape[2])
            make_aeon(spec["method"]).fit(X, np.arange(len(X), dtype=float))
    fold_rows, preds = [], []
    for t in TARGETS:
        y_tr, y_ev = targets[t][tr], targets[t][ev]
        t0 = time.perf_counter()
        pred_all = {"training_mean": np.full(len(ev), y_tr.mean())}
        times = {"training_mean": time.perf_counter() - t0}
        score_cache = {}
        for name, spec in sets.items():
            pred_all[name], times[name] = fit_predict(spec, data, tr, ev,
                                                      y_tr, score_cache)
        for name, pred in pred_all.items():
            fold_rows.append({
                "repeat": r, "fold": k, "target": t, "feature_set": name,
                "mae": float(np.mean(np.abs(y_ev - pred))),
                "rmse": float(np.sqrt(np.mean((y_ev - pred) ** 2))),
                "n_eval": len(ev), "fit_predict_s": times[name]})
            preds.append(pd.DataFrame({
                "repeat": r, "fold": k, "target": t, "feature_set": name,
                "subject_ID": np.asarray(ids)[ev], "y_true": y_ev,
                "y_pred": pred}))
    return fold_rows, preds


def corrected_ttest(diff, n_splits):
    """Nadeau & Bengio (2003) corrected resampled t-test on per-fold diffs."""
    diff = np.asarray(diff, dtype=float)
    J = len(diff)
    rho = 1.0 / (n_splits - 1)  # n_test / n_train for k-fold
    var = np.var(diff, ddof=1)
    if var == 0:
        return np.nan
    t = diff.mean() / np.sqrt((1.0 / J + rho) * var)
    return float(2 * stats.t.sf(abs(t), df=J - 1))


def holm(pvals):
    """Holm step-down adjusted p-values (NaNs ignored)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    out_ok = np.empty(m)
    out_ok[idx] = np.minimum(adj, 1.0)
    out[ok] = out_ok
    return out


def per_repeat_metrics(preds):
    """MAE, RMSE, R^2 over all held-out predictions of each repeat."""
    def metrics(g):
        err = g["y_true"] - g["y_pred"]
        sst = np.sum((g["y_true"] - g["y_true"].mean()) ** 2)
        return pd.Series({"mae": np.mean(np.abs(err)),
                          "rmse": np.sqrt(np.mean(err ** 2)),
                          "r2": 1 - np.sum(err ** 2) / sst})
    rows = []
    for key, g in preds.groupby(["target", "feature_set", "repeat"]):
        rows.append(dict(zip(["target", "feature_set", "repeat"], key),
                         **metrics(g)))
    return pd.DataFrame(rows)


def _mcm(scores, title, png_path):
    """Draw and save a multiple comparison matrix with whichever aeon
    version is installed (its arguments changed in aeon 1.3 and 1.4)."""
    import inspect

    import matplotlib.pyplot as plt
    from aeon.visualisation import create_multi_comparison_matrix as mcm
    params = inspect.signature(mcm).parameters
    common = {"pvalue_test": "wilcoxon",
                  "pvalue_test_params": {"alternative": "two-sided",
                                      "zero_method": "wilcox"},
                  "pvalue_correction": "Holm", "pvalue_threshold": 0.05,
                  "order_stats": "average-statistic", "fig_size": (22, 13),
                  "font_size": 11}
    if "statistic_name" in params:  # aeon >= 1.4
        fig = mcm(scores, statistic_name="MAE", higher_stat_better=False,
                  order_stats_increasing=True, formats=None, **common)
    elif "higher_stat_better" in params:  # aeon 1.3
        fig = mcm(scores, used_statistic="MAE", higher_stat_better=False,
                  order_stats_increasing=True, formats=None, **common)
    else:  # aeon < 1.3: saves the file itself
        mcm(scores, output_dir=str(png_path.parent) + "/",
            png_savename=png_path.stem, used_statistic="MAE",
            order_win_tie_loss="lower", order_better="increasing", **common)
        plt.close("all")
        return
    fig.suptitle(title, y=1.0)
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stats_figures(per_rep, stat_sets, out_dir, tag):
    """Friedman test, CD diagram and MCM per target on the per-repeat MAE
    (one paired score per repeat, as Lab 8 uses one per fold)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from aeon.visualisation import plot_critical_difference
    except ImportError:
        print("aeon/matplotlib not installed: skipping CD diagrams and MCM")
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    present = [s for s in stat_sets if s in set(per_rep["feature_set"])]
    n_rep = per_rep["repeat"].nunique()
    print(f"\nFriedman test + CD diagram + MCM on per-repeat MAE "
          f"({n_rep} paired scores per method, exploratory), methods: "
          f"{present}")
    for t in TARGETS:
        scores = (per_rep[per_rep.target == t]
                  .pivot(index="repeat", columns="feature_set", values="mae")
                  [present])
        chi2, p = stats.friedmanchisquare(*[scores[c] for c in present])
        print(f"  {t}: Friedman chi2 = {chi2:.1f}, p = {p:.2g}")
        scores = scores.rename(columns=lambda c: DISPLAY.get(c, c))
        labels = list(scores.columns)
        try:
            fig, ax = plot_critical_difference(
                scores.to_numpy(), labels, lower_better=True,
                test="wilcoxon", correction="holm", alpha=0.05, width=9,
                textspace=2.2)
            ax.set_title(f"{t}: MAE over {n_rep} CV repeats, "
                         "Wilcoxon-Holm, alpha = 0.05", pad=25)
            fig.savefig(fig_dir / f"cd_{t}{tag}.png", dpi=200,
                        bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:  # never lose the run over a figure  # noqa: BLE001
            print(f"  CD diagram for {t} failed: {exc!r}")
        try:
            _mcm(scores, f"{t}: MCM on MAE over {n_rep} CV repeats "
                 "(two-sided Wilcoxon, Holm, alpha = 0.05)",
                 fig_dir / f"mcm_{t}{tag}.png")
        except Exception as exc:  # noqa: BLE001
            print(f"  MCM for {t} failed: {exc!r}")
    print(f"  Saved CD diagrams and MCMs to {fig_dir}")


def mode_figures(args):
    """Redraw the CD diagrams / MCMs from a saved cv run (no refitting)."""
    out_dir = Path(args.results_csv).parent / "improvements"
    tag = f"_{args.out_tag}" if args.out_tag else ""
    path = out_dir / f"feature_fusion_cv_predictions{tag}.csv"
    if not path.exists():
        raise SystemExit(f"No saved cv predictions at {path}: run --mode cv "
                         "first (with the same --out-tag).")
    per_rep = per_repeat_metrics(pd.read_csv(path))
    stats_figures(per_rep, args.stat_sets or DEFAULT_STAT_SETS, out_dir, tag)


def mode_timing(args, splits):
    out_dir = Path(args.results_csv).parent / "improvements"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    print("Measuring feature extraction cost (CPU: "
          f"{platform.processor() or platform.machine()}, "
          f"{platform.system()}) ...")
    cost = measure_extraction_cost(args, splits, args.embeddings)
    cost.to_csv(out_dir / f"feature_extraction_cost{tag}.csv", index=False)
    print(cost[["step", "ms_per_participant"]].round(2).to_string(
        index=False))


def mode_cv(args, splits, sets):
    need_raw = any(s["kind"] == "aeon" for s in sets.values())
    ids, data, targets, _ = load_blocks(args, splits, ["train", "val"],
                                        args.embeddings, need_raw)
    n = len(ids)
    rkf = RepeatedKFold(n_splits=args.folds, n_repeats=args.repeats,
                        random_state=SEED)
    tasks = [(i // args.folds, i % args.folds, tr, ev)
             for i, (tr, ev) in enumerate(rkf.split(np.arange(n)))]
    print(f"\nRepeated CV: {args.repeats} x {args.folds}-fold on {n} "
          f"participants, {len(sets) + 1} feature sets, {len(TARGETS)} "
          "targets ...")
    t0 = time.perf_counter()
    out = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_fold)(r, k, tr, ev, data, targets, sets, ids)
        for r, k, tr, ev in tasks)
    folds = pd.DataFrame([row for rows, _ in out for row in rows])
    preds = pd.concat([p for _, ps in out for p in ps], ignore_index=True)
    print(f"Done in {time.perf_counter() - t0:.0f} s")

    out_dir = Path(args.results_csv).parent / "improvements"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    folds.to_csv(out_dir / f"feature_fusion_cv_folds{tag}.csv", index=False)
    preds.to_csv(out_dir / f"feature_fusion_cv_predictions{tag}.csv",
                 index=False)

    per_rep = per_repeat_metrics(preds)
    dummy = (per_rep[per_rep.feature_set == "training_mean"]
             .set_index(["target", "repeat"])["mae"])
    per_rep["gain_vs_dummy_pct"] = 100 * (
        1 - per_rep["mae"] / dummy.loc[list(zip(per_rep.target,
                                               per_rep.repeat))].to_numpy())
    times = folds.groupby(["target", "feature_set"])["fit_predict_s"].mean()

    summary = []
    for (t, s), g in per_rep.groupby(["target", "feature_set"]):
        summary.append({
            "target": t, "feature_set": s,
            "mae_mean": g["mae"].mean(), "mae_sd_over_repeats": g["mae"].std(ddof=1),
            "rmse_mean": g["rmse"].mean(), "r2_mean": g["r2"].mean(),
            "gain_vs_dummy_pct": g["gain_vs_dummy_pct"].mean(),
            "fit_predict_s_per_fold": times.loc[(t, s)]})
    summary = pd.DataFrame(summary)

    # Paired comparisons on the 50 folds vs each reference, Holm-corrected
    # over all sets compared with that reference for that target.
    for ref in args.references:
        for t in TARGETS:
            ft = folds[folds.target == t]
            other = (ft[ft.feature_set == ref]
                     .sort_values(["repeat", "fold"])["mae"].to_numpy())
            idx = summary.index[summary.target == t]
            ps = []
            for i in idx:
                s = summary.at[i, "feature_set"]
                mine = (ft[ft.feature_set == s]
                        .sort_values(["repeat", "fold"])["mae"].to_numpy())
                d = mine - other
                summary.at[i, f"diff_vs_{ref}"] = d.mean()
                p = corrected_ttest(d, args.folds) if s != ref else np.nan
                summary.at[i, f"p_vs_{ref}"] = p
                summary.at[i, f"pct_folds_better_than_{ref}"] = \
                    100 * np.mean(d < 0)
                ps.append(p)
            summary.loc[idx, f"p_holm_vs_{ref}"] = holm(ps)
    summary = summary.sort_values(["target", "mae_mean"])
    summary.to_csv(out_dir / f"feature_fusion_cv_summary{tag}.csv",
                   index=False)

    for t in TARGETS:
        st = summary[summary.target == t].set_index("feature_set")
        print(f"\n=== {t} ({UNITS[t]}), mean over {args.repeats} repeats, "
              "sorted by MAE ===")
        show = st[["mae_mean", "mae_sd_over_repeats", "rmse_mean", "r2_mean",
                   "gain_vs_dummy_pct", "fit_predict_s_per_fold"]].copy()
        show.columns = ["MAE", "MAE sd", "RMSE", "R2", "Gain vs Dummy %",
                        "Fit+predict s/fold"]
        print(show.round(3).to_string())
        for ref in args.references:
            d = st[f"diff_vs_{ref}"].round(2).astype(str)
            star = np.where(st[f"p_holm_vs_{ref}"] < 0.05, "*", " ")
            col = (d + star).rename(f"vs {ref}")
            st[f"vs {ref}"] = col
        cols = [f"vs {ref}" for ref in args.references]
        print("Difference in MAE (negative = better than the reference; "
              "* = corrected t-test, Holm-adjusted p < 0.05):")
        print(st[cols].to_string())

    stat_sets = args.stat_sets or DEFAULT_STAT_SETS
    stats_figures(per_rep, stat_sets, out_dir, tag)

    if args.timing:
        print()
        mode_timing(args, splits)

    print(f"\nSaved to {out_dir}: feature_fusion_cv_summary{tag}.csv, "
          f"feature_fusion_cv_folds{tag}.csv, "
          f"feature_fusion_cv_predictions{tag}.csv")


# --------------------------------------------------------------------------
# Mode: train -> val, or train+val -> test
# --------------------------------------------------------------------------

def result_from_predictions(y_true, y_pred, split, params, seed=42):
    """Same result format and bootstrap as evaluate_features()."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=(500, len(y_true)), replace=True)
    boot = np.abs(y_true - y_pred)[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"parameters": params,
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "lower_bound_mae": float(lo), "upper_bound_mae": float(hi),
            "y_test": y_true.tolist(), "y_pred": y_pred.tolist(),
            "evaluation_split": split}


def mode_holdout(args, splits, sets):
    split = args.mode
    train_splits = ["train"] if split == "val" else ["train", "val"]
    need_raw = any(s["kind"] == "aeon" for s in sets.values())
    ids, data, targets, _ = load_blocks(args, splits, train_splits + [split],
                                        args.embeddings, need_raw)
    n_tr = sum(len(splits[s]) for s in train_splits)
    tr, ev = np.arange(n_tr), np.arange(n_tr, len(ids))
    print(f"\nTraining on {' + '.join(train_splits)} ({n_tr} participants), "
          f"evaluating on {split} ({len(ev)} participants)")
    procedure = (f"feature_fusion_paper_grid_{'+'.join(train_splits)}_to_"
                 f"{split}_drop{'-'.join(map(str, args.drop_ids)) or 'none'}"
                 "_seed42")
    warnings.filterwarnings("ignore")
    summary, pred_rows = [], []
    for t in TARGETS:
        y_tr, y_ev = targets[t][tr], targets[t][ev]
        results = {"training_mean": result_from_predictions(
            y_ev, np.full(len(ev), y_tr.mean()), split,
            {"constant": float(y_tr.mean())})}
        times = {"training_mean": 0.0}
        for name, spec in sets.items():
            t0 = time.perf_counter()
            if spec["kind"] == "ridge":
                X_tr, X_ev = build_matrices(spec, data["blocks"], tr, ev, y_tr)
                results[name] = evaluate_features(
                    X_train=X_tr, y_train=y_tr, X_eval=X_ev, y_eval=y_ev,
                    evaluation_split=split, bootstrap_seed=42)
            else:
                pred, _ = fit_predict(spec, data, tr, ev, y_tr)
                results[name] = result_from_predictions(
                    y_ev, pred, split, {"aeon_defaults": spec["method"]})
            times[name] = time.perf_counter() - t0
        dummy_mae = results["training_mean"]["mae"]
        for name, res in results.items():
            record_result(result=res, model_name=name, target=t, n_train=n_tr,
                          procedure=procedure, csv_path=args.results_csv)
            yt, yp = np.array(res["y_test"]), np.array(res["y_pred"])
            summary.append({
                "feature_set": name, "target": t, "mae": res["mae"],
                "ci_lower": res["lower_bound_mae"],
                "ci_upper": res["upper_bound_mae"],
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "r2": float(1 - np.sum((yt - yp) ** 2) /
                         np.sum((yt - yt.mean()) ** 2)),
                "gain_vs_dummy_pct": 100 * (1 - res["mae"] / dummy_mae),
                "fit_predict_s": times[name]})
            pred_rows.append(pd.DataFrame({
                "target": t, "feature_set": name,
                "subject_ID": np.asarray(ids)[ev], "y_true": yt,
                "y_pred": yp}))
    out_dir = Path(args.results_csv).parent / "improvements"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(summary)
    table.to_csv(out_dir / f"feature_fusion_summary_{split}.csv", index=False)
    pd.concat(pred_rows).to_csv(
        out_dir / f"feature_fusion_predictions_{split}.csv", index=False)
    for t in TARGETS:
        show = (table[table.target == t].set_index("feature_set")
                .sort_values("mae")
                [["mae", "ci_lower", "ci_upper", "rmse", "r2",
                  "gain_vs_dummy_pct", "fit_predict_s"]])
        print(f"\n=== {t} ({UNITS[t]}), {split}, n = {len(ev)} ===")
        print(show.round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data_PPG")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="cv",
                    choices=["cv", "val", "test", "figures", "timing"],
                    help="figures: redraw CD/MCM from a saved cv run; "
                         "timing: only measure feature extraction cost")
    ap.add_argument("--embeddings", nargs="+", default=EMBEDDINGS,
                    help="papagei_s, papagei_p, papagei_s_svri; add "
                         "@<variant> for another experiment-1 input "
                         "representation, e.g. papagei_s@tile_cycles")
    ap.add_argument("--aeon", nargs="*", default=AEON_METHODS,
                    choices=AEON_METHODS,
                    help="course TSER baselines to include "
                         "(pass --aeon with nothing after it to skip them)")
    ap.add_argument("--sets", nargs="+", default=None,
                    help="only run these feature sets (default: all)")
    ap.add_argument("--drop-ids", nargs="*", type=int, default=[24],
                    help="participants to leave out (default: 24, duplicate "
                         "of a recording of 23); pass --drop-ids with "
                         "nothing after it to keep everyone")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=-1,
                    help="parallel folds (-1 = all CPU cores)")
    ap.add_argument("--references", nargs="+",
                    default=["papagei_s", "handcrafted"],
                    help="feature sets to compare everything against (cv)")
    ap.add_argument("--stat-sets", nargs="+", default=None,
                    help="methods in the CD diagram / MCM (cv)")
    ap.add_argument("--no-timing", dest="timing", action="store_false",
                    help="skip measuring feature extraction cost (cv)")
    ap.add_argument("--out-tag", default="",
                    help="suffix for the cv output files, so runs with "
                         "different options don't overwrite each other")
    ap.add_argument("--results-csv", default="results/improvement_results.csv")
    args = ap.parse_args()

    for e in args.embeddings:
        parse_embedding(e)
    if args.mode == "figures":
        mode_figures(args)
        return
    if args.aeon and args.mode != "timing":
        try:
            import aeon  # noqa: F401
        except ImportError:
            raise SystemExit("The ROCKET-family baselines need aeon: run "
                             "`pip install aeon`, or pass --aeon with nothing "
                             "after it to skip them.")
    splits = apply_drop(pc.split_metadata(pc.load_metadata(args.data_dir)),
                        args.drop_ids)
    sets = feature_sets(args.embeddings, args.aeon)
    if args.sets:
        unknown = [x for x in args.sets if x not in sets]
        if unknown:
            raise SystemExit(f"--sets: unknown {unknown}; choose from "
                             f"{list(sets)}")
        keep = set(args.sets) | set(args.references) | set(
            args.stat_sets or [])
        sets = {k: v for k, v in sets.items() if k in keep}
    for ref in args.references:
        if ref not in sets and ref != "training_mean":
            raise SystemExit(f"--references: unknown feature set {ref}")
    if args.mode == "timing":
        mode_timing(args, splits)
    elif args.mode == "cv":
        mode_cv(args, splits, sets)
    else:
        mode_holdout(args, splits, sets)


if __name__ == "__main__":
    main()
