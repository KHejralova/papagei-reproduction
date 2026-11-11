"""
Improvement experiment 2: what does PaPaGei add beyond simple information
about the person, how should the two be combined, and how does it compare
with the standard time series regression methods from the course?

Methods (defined in estimators.feature_sets):

  demo                      age, sex, height, weight, BMI
  morph                     sVRI and SQI (IPA left out, see features.py)
  demo+morph                the paper's "Demo + PPG" baseline (Table 15)
  peak_hr                   heart rate from the interval between pulse peaks
  handcrafted               demo + morph + peak_hr ("simple features")
  <emb>                     PaPaGei embedding alone (as in the paper)
  <emb>+demo_concat         embedding + demographics in one ridge
                            (early fusion, the paper's Table 15)
  <emb>+demo_stacked        PPG score from the embedding + demographics
  <emb>+handcrafted_stacked PPG score + all simple features (late fusion)
  rocket, minirocket,       course baselines (Lab 8) on the raw PPG; each
  hydra, quant              recording is one series, predictions averaged
                            over a participant's 3 recordings

Participant 24 is left out by default (--drop-ids): one of its recordings
is identical to one of participant 23's (leakage through duplicates, Week 7).

Modes (--mode):

  cv       (default, for choosing) repeated 5-fold CV over the train + val
           participants (10 repeats = 50 folds). Test set untouched.
  val      train on training participants, evaluate on validation.
  test     retrain on train + val, evaluate ONCE on test (Week 7). Run only
           after the choices are frozen.
  figures  redraw the CD diagrams and MCMs from a saved cv run (fast).
  timing   only measure feature extraction cost per participant.

Example (from the repo root):
    python improvements/02_feature_fusion.py --data-dir data_PPG
"""

import argparse
import platform
import time
import warnings
from pathlib import Path

import data as D
import features as F
import numpy as np
import pandas as pd
from embeddings import cached_embeddings, parse_embedding
from estimators import AEON_METHODS, build_matrices, feature_sets, fit_predict
from evaluation import (
    measure_extraction_cost,
    per_repeat_metrics,
    repeated_cv,
    result_from_predictions,
    summarise,
    training_mean_result,
)
from plots import DEFAULT_STAT_SETS, stats_figures

from evaluation_utils import evaluate_features
from results_utils import record_result

EMBEDDINGS = ["papagei_s", "papagei_p"]


def load_inputs(args, splits, split_names, sets):
    """Everything the methods need, for the participants of the given
    splits (in that order): feature blocks, raw series and targets."""
    meta = pd.concat([splits[s] for s in split_names], ignore_index=True)
    ids = meta["subject_ID"].astype(int).tolist()
    cache_dir = Path(args.features_dir or
                     Path(args.data_dir) / "features_improvements")
    cache_dir.mkdir(parents=True, exist_ok=True)

    recordings = None

    def get_recordings():  # preprocess only if something isn't cached
        nonlocal recordings
        if recordings is None:
            print(f"Preprocessing {len(ids) * 3} recordings ...")
            recordings = D.load_preprocessed_recordings(args.data_dir, ids)
        return recordings

    blocks = {"demo": F.demographic_features(meta)}
    blocks.update(F.handcrafted_features(ids, get_recordings, cache_dir,
                                         D.FS_TARGET))
    for e in args.embeddings:
        model_name, variant = parse_embedding(e)
        by_split = cached_embeddings(
            model_name, variant,
            {s: splits[s]["subject_ID"].astype(int).tolist()
             for s in split_names},
            get_recordings, cache_dir, args.weights_dir, args.device)
        blocks[e] = np.vstack([by_split[s] for s in split_names])

    need_raw = any(s["kind"] == "aeon" for s in sets.values())
    raw = D.raw_matrix(get_recordings(), ids) if need_raw else None
    targets = {t: meta[t].to_numpy(float) for t in D.TARGETS}
    return ids, {"blocks": blocks, "raw": raw}, targets


def out_paths(args):
    out_dir = Path(args.results_csv).parent / "improvements"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    return out_dir, tag


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def mode_cv(args, splits, sets):
    ids, data, targets = load_inputs(args, splits, ["train", "val"], sets)
    print(f"\nRepeated CV: {args.repeats} x {args.folds}-fold on {len(ids)} "
          f"participants, {len(sets) + 1} feature sets, {len(D.TARGETS)} "
          "targets ...")
    t0 = time.perf_counter()
    folds, preds = repeated_cv(data, targets, sets, ids, args.folds,
                               args.repeats, args.jobs)
    print(f"Done in {time.perf_counter() - t0:.0f} s")

    out_dir, tag = out_paths(args)
    folds.to_csv(out_dir / f"feature_fusion_cv_folds{tag}.csv", index=False)
    preds.to_csv(out_dir / f"feature_fusion_cv_predictions{tag}.csv",
                 index=False)
    per_rep, summary = summarise(folds, preds, args.references, args.folds)
    summary.to_csv(out_dir / f"feature_fusion_cv_summary{tag}.csv",
                   index=False)

    for t in D.TARGETS:
        st = summary[summary.target == t].set_index("feature_set")
        print(f"\n=== {t} ({D.UNITS[t]}), mean over {args.repeats} repeats, "
              "sorted by MAE ===")
        show = st[["mae_mean", "mae_sd_over_repeats", "rmse_mean", "r2_mean",
                   "gain_vs_dummy_pct", "fit_predict_s_per_fold"]].copy()
        show.columns = ["MAE", "MAE sd", "RMSE", "R2", "Gain vs Dummy %",
                        "Fit+predict s/fold"]
        print(show.round(3).to_string())
        for ref in args.references:
            star = np.where(st[f"p_holm_vs_{ref}"] < 0.05, "*", " ")
            st[f"vs {ref}"] = st[f"diff_vs_{ref}"].round(2).astype(str) + star
        print("Difference in MAE (negative = better than the reference; "
              "* = corrected t-test, Holm-adjusted p < 0.05):")
        print(st[[f"vs {ref}" for ref in args.references]].to_string())

    stats_figures(per_rep, args.stat_sets or DEFAULT_STAT_SETS, out_dir, tag)
    if args.timing:
        print()
        mode_timing(args, splits)
    print(f"\nSaved to {out_dir}: feature_fusion_cv_summary{tag}.csv, "
          f"feature_fusion_cv_folds{tag}.csv, "
          f"feature_fusion_cv_predictions{tag}.csv")


def mode_holdout(args, splits, sets):
    split = args.mode
    train_splits = ["train"] if split == "val" else ["train", "val"]
    ids, data, targets = load_inputs(args, splits, train_splits + [split],
                                     sets)
    n_tr = sum(len(splits[s]) for s in train_splits)
    tr, ev = np.arange(n_tr), np.arange(n_tr, len(ids))
    print(f"\nTraining on {' + '.join(train_splits)} ({n_tr} participants), "
          f"evaluating on {split} ({len(ev)} participants)")
    procedure = (f"feature_fusion_paper_grid_{'+'.join(train_splits)}_to_"
                 f"{split}_drop{'-'.join(map(str, args.drop_ids)) or 'none'}"
                 "_seed42")
    warnings.filterwarnings("ignore")
    summary, pred_rows = [], []
    for t in D.TARGETS:
        y_tr, y_ev = targets[t][tr], targets[t][ev]
        results = {"training_mean": training_mean_result(y_tr, y_ev, split)}
        times = {"training_mean": 0.0}
        for name, spec in sets.items():
            t0 = time.perf_counter()
            if spec["kind"] == "ridge":  # the shared evaluator
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

    out_dir, _ = out_paths(args)
    table = pd.DataFrame(summary)
    table.to_csv(out_dir / f"feature_fusion_summary_{split}.csv", index=False)
    pd.concat(pred_rows).to_csv(
        out_dir / f"feature_fusion_predictions_{split}.csv", index=False)
    for t in D.TARGETS:
        show = (table[table.target == t].set_index("feature_set")
                .sort_values("mae")
                [["mae", "ci_lower", "ci_upper", "rmse", "r2",
                  "gain_vs_dummy_pct", "fit_predict_s"]])
        print(f"\n=== {t} ({D.UNITS[t]}), {split}, n = {len(ev)} ===")
        print(show.round(3).to_string())


def mode_figures(args):
    """Redraw the CD diagrams / MCMs from a saved cv run (no refitting)."""
    out_dir, tag = out_paths(args)
    path = out_dir / f"feature_fusion_cv_predictions{tag}.csv"
    if not path.exists():
        raise SystemExit(f"No saved cv predictions at {path}: run --mode cv "
                         "first (with the same --out-tag).")
    per_rep = per_repeat_metrics(pd.read_csv(path))
    stats_figures(per_rep, args.stat_sets or DEFAULT_STAT_SETS, out_dir, tag)


def mode_timing(args, splits):
    out_dir, tag = out_paths(args)
    print("Measuring feature extraction cost (CPU: "
          f"{platform.processor() or platform.machine()}, "
          f"{platform.system()}) ...")
    ids = splits["train"]["subject_ID"].astype(int).tolist()[:30]
    cost = measure_extraction_cost(args.data_dir, ids, args.embeddings,
                                   args.weights_dir, args.device)
    cost.to_csv(out_dir / f"feature_extraction_cost{tag}.csv", index=False)
    print(cost[["step", "ms_per_participant"]].round(2).to_string(
        index=False))


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data_PPG")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="cv",
                    choices=["cv", "val", "test", "figures", "timing"])
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
        return mode_figures(args)
    if args.aeon and args.mode != "timing":
        try:
            import aeon  # noqa: F401
        except ImportError:
            raise SystemExit("The ROCKET-family baselines need aeon: run "
                             "`pip install aeon`, or pass --aeon with nothing "
                             "after it to skip them.")
    splits = D.load_splits(args.data_dir, args.drop_ids)
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
