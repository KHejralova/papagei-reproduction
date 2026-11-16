"""
Improvement experiment 2: 
    what does PaPaGei add beyond simple information about the person, 
    how should the two be combined 
    and how does it compare with the standard time series regression methods?

Methods (defined in estimators.feature_sets):

  demo                      age, sex, height, weight, BMI
  morph                     sVRI and SQI (IPA left out, see features.py)
  demo+morph                the paper's "Demo + PPG" baseline (Table 15)
  peak_hr                   heart rate from the interval between pulse peaks
  handcrafted               demo + morph + peak_hr ("simple features")
  <emb>                     PaPaGei embedding alone (as in the paper)
  <emb>+demo_concat         embedding + demographics in one ridge (early fusion)
  <emb>+demo_stacked        PPG score from the embedding + demographics
  <emb>+handcrafted_stacked PPG score + all simple features (late fusion)
  rocket, minirocket,       baselines on the raw PPG
  hydra, quant              

Participant 24 is left out by default: one of its recordings is identical to one of participant 23's.

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
import platform  # to print which CPU the timing was done on
import time
import warnings
from pathlib import Path

import data as D  # our own files in improvements/
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

# code from reproduction so the ridge eval and results CSV are identical
from evaluation_utils import evaluate_features
from results_utils import record_result

# the embeddings used by default (S = the main Papagei model, P = the other one)
EMBEDDINGS = ["papagei_s", "papagei_p"]


def load_inputs(args, splits, split_names, sets):
    """Everything the methods need, for the participants of the given splits (in that order): feature blocks, raw series and targets."""
    # one table with all the participants we need
    meta = pd.concat([splits[s] for s in split_names], ignore_index=True)
    ids = meta["subject_ID"].astype(int).tolist()
    # folder for commputed features and embeddings (so they are computed only once)
    cache_dir = Path(args.features_dir or
                     Path(args.data_dir) / "features_improvements")
    cache_dir.mkdir(parents=True, exist_ok=True)

    recordings = None

    def get_recordings():  # preprocess only if something isn't cached
        # preprocessing is slow -> happens only first time it is needed -> then same recordings are reused
        nonlocal recordings
        if recordings is None:
            print(f"Preprocessing {len(ids) * 3} recordings ...")
            recordings = D.load_preprocessed_recordings(args.data_dir, ids)
        return recordings

    blocks = {"demo": F.demographic_features(meta)} # the groups of features
    # add the morph (sVRI, SQI) and peak_hr blocks
    blocks.update(F.handcrafted_features(ids, get_recordings, cache_dir,
                                         D.FS_TARGET))
    for e in args.embeddings:
        model_name, variant = parse_embedding(e)
        # embeddings come back per split
        by_split = cached_embeddings(
            model_name, variant,
            {s: splits[s]["subject_ID"].astype(int).tolist()
             for s in split_names},
            get_recordings, cache_dir, args.weights_dir, args.device)
        # stacking them in the same order as the splits, so the rows match the meta table
        blocks[e] = np.vstack([by_split[s] for s in split_names])

    # the raw signals are needed for the aeon baselines, but not for the ridge models
    need_raw = any(s["kind"] == "aeon" for s in sets.values())
    raw = D.raw_matrix(get_recordings(), ids) if need_raw else None
    # the three things we predict (SBP, DBP, HR) as arrays
    targets = {t: meta[t].to_numpy(float) for t in D.TARGETS}
    return ids, {"blocks": blocks, "raw": raw}, targets


def out_paths(args):
    # results go into results/improvements
    # tag is added to file names -> different runs do not overwrite each other
    out_dir = Path(args.results_csv).parent / "improvements"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    return out_dir, tag


# MODELS
def mode_cv(args, splits, sets):
    # CV only uses train + val
    ids, data, targets = load_inputs(args, splits, ["train", "val"], sets)
    # + 1 because the Dummy (training mean) is always added as well
    print(f"\nRepeated CV: {args.repeats} x {args.folds}-fold on {len(ids)} "
          f"participants, {len(sets) + 1} feature sets, {len(D.TARGETS)} "
          "targets ...")
    t0 = time.perf_counter()
    # the actual work - every method on every fold of every repeat
    # folds = one row per fold (MAE etc.), preds = every single prediction
    folds, preds = repeated_cv(data, targets, sets, ids, args.folds,
                               args.repeats, args.jobs)
    print(f"Done in {time.perf_counter() - t0:.0f} s")

    # save the raw results
    out_dir, tag = out_paths(args)
    folds.to_csv(out_dir / f"feature_fusion_cv_folds{tag}.csv", index=False)
    preds.to_csv(out_dir / f"feature_fusion_cv_predictions{tag}.csv",
                 index=False)
    # summary = averages over the repeats + the corrected t-tests against the reference methods
    # per_rep = the MAE of each repeat (for the figures)
    per_rep, summary = summarise(folds, preds, args.references, args.folds)
    summary.to_csv(out_dir / f"feature_fusion_cv_summary{tag}.csv",
                   index=False)

    # print one table per target
    for t in D.TARGETS:
        st = summary[summary.target == t].set_index("feature_set")
        print(f"\n=== {t} ({D.UNITS[t]}), mean over {args.repeats} repeats, "
              "sorted by MAE ===")
        show = st[["mae_mean", "mae_sd_over_repeats", "rmse_mean", "r2_mean",
                   "gain_vs_dummy_pct", "fit_predict_s_per_fold"]].copy()
        show.columns = ["MAE", "MAE sd", "RMSE", "R2", "Gain vs Dummy %",
                        "Fit+predict s/fold"]
        print(show.round(3).to_string())
        # second table - how much better/worse each method is than each reference,
        # * = the difference is significant
        for ref in args.references:
            star = np.where(st[f"p_holm_vs_{ref}"] < 0.05, "*", " ")
            st[f"vs {ref}"] = st[f"diff_vs_{ref}"].round(2).astype(str) + star
        print("Difference in MAE (negative = better than the reference; "
              "* = corrected t-test, Holm-adjusted p < 0.05):")
        print(st[[f"vs {ref}" for ref in args.references]].to_string())

    # CD diagrams and MCMs (only for the methods in --stat-sets)
    stats_figures(per_rep, args.stat_sets or DEFAULT_STAT_SETS, out_dir, tag)
    # also measure the feature extraction cost, unless no timing
    if args.timing:
        print()
        mode_timing(args, splits)
    print(f"\nSaved to {out_dir}: feature_fusion_cv_summary{tag}.csv, "
          f"feature_fusion_cv_folds{tag}.csv, "
          f"feature_fusion_cv_predictions{tag}.csv")


def mode_holdout(args, splits, sets):
    # used for both --mode val and --mode test
    split = args.mode # the split we evaluate on "val" or "test"
    # val - train on train only, test - retrain on train + val
    train_splits = ["train"] if split == "val" else ["train", "val"]
    ids, data, targets = load_inputs(args, splits, train_splits + [split],
                                     sets)
    # training participants come first, then evaluation ones
    # so the first n_tr rows are training and the rest are evaluation
    n_tr = sum(len(splits[s]) for s in train_splits)
    tr, ev = np.arange(n_tr), np.arange(n_tr, len(ids))
    print(f"\nTraining on {' + '.join(train_splits)} ({n_tr} participants), "
          f"evaluating on {split} ({len(ev)} participants)")
    # name describing exactly how this was run, saved in the results CSV,
    # e.g. feature_fusion_paper_grid_train+val_to_test_drop24_seed42
    procedure = (f"feature_fusion_paper_grid_{'+'.join(train_splits)}_to_"
                 f"{split}_drop{'-'.join(map(str, args.drop_ids)) or 'none'}"
                 "_seed42")
    warnings.filterwarnings("ignore") # hide the convergence warnings
    summary, pred_rows = [], []
    for t in D.TARGETS:
        y_tr, y_ev = targets[t][tr], targets[t][ev]
        # the Dummy - always predicts the training mean
        results = {"training_mean": training_mean_result(y_tr, y_ev, split)}
        times = {"training_mean": 0.0}
        for name, spec in sets.items():
            t0 = time.perf_counter()
            if spec["kind"] == "ridge":  # the shared evaluator
                # build the feature matrix for this method (for stacking this includes the out-of-fold PPG score, learned on training only)
                X_tr, X_ev = build_matrices(spec, data["blocks"], tr, ev, y_tr)
                # same grid search, metrics and bootstrap CI (seed 42) as in the reproduction
                results[name] = evaluate_features(
                    X_train=X_tr, y_train=y_tr, X_eval=X_ev, y_eval=y_ev,
                    evaluation_split=split, bootstrap_seed=42)
            else:
                # the aeon methods (ROCKET etc.) don't fit into evaluate_features
                # so we predict ourselves and then compute the same metrics in the same format    
                pred, _ = fit_predict(spec, data, tr, ev, y_tr)
                results[name] = result_from_predictions(
                    y_ev, pred, split, {"aeon_defaults": spec["method"]})
            times[name] = time.perf_counter() - t0
        dummy_mae = results["training_mean"]["mae"]
        for name, res in results.items():
            # add a row to the shared results CSV (same as the reproduction)
            record_result(result=res, model_name=name, target=t, n_train=n_tr,
                          procedure=procedure, csv_path=args.results_csv)
            yt, yp = np.array(res["y_test"]), np.array(res["y_pred"])
            # our own summary row - MAE + its 95% bootstrap CI, RMSE, R2, how much better than the Dummy (%) and how long it took
            summary.append({
                "feature_set": name, "target": t, "mae": res["mae"],
                "ci_lower": res["lower_bound_mae"],
                "ci_upper": res["upper_bound_mae"],
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
                "r2": float(1 - np.sum((yt - yp) ** 2) /
                         np.sum((yt - yt.mean()) ** 2)),
                "gain_vs_dummy_pct": 100 * (1 - res["mae"] / dummy_mae),
                "fit_predict_s": times[name]})
            # keep every prediction too (used for the predicted vs true figure)
            pred_rows.append(pd.DataFrame({
                "target": t, "feature_set": name,
                "subject_ID": np.asarray(ids)[ev], "y_true": yt,
                "y_pred": yp}))

    # save and print one table per target
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
    """Redraw the CD diagrams / MCMs from a saved cv run (no refitting) -> no need for a new CV run."""
    out_dir, tag = out_paths(args)
    path = out_dir / f"feature_fusion_cv_predictions{tag}.csv"
    if not path.exists():
        raise SystemExit(f"No saved cv predictions at {path}: run --mode cv "
                         "first (with the same --out-tag).")
    per_rep = per_repeat_metrics(pd.read_csv(path))
    stats_figures(per_rep, args.stat_sets or DEFAULT_STAT_SETS, out_dir, tag)


def mode_timing(args, splits):
    """how many milliseconds it takes to compute each kind of feature for one participant (for the cost figure); 
    the times depend on the computer, so we print which CPU was used"""
    out_dir, tag = out_paths(args)
    print("Measuring feature extraction cost (CPU: "
          f"{platform.processor() or platform.machine()}, "
          f"{platform.system()}) ...")
    # 30 training participants are enough to get an average
    ids = splits["train"]["subject_ID"].astype(int).tolist()[:30]
    cost = measure_extraction_cost(args.data_dir, ids, args.embeddings,
                                   args.weights_dir, args.device)
    cost.to_csv(out_dir / f"feature_extraction_cost{tag}.csv", index=False)
    print(cost[["step", "ms_per_participant"]].round(2).to_string(
        index=False))


# Command line
def main():
    """all the options you can pass in the terminal,
    the docstring at the top is shown as the help text (python 02_feature_fusion.py --help)"""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data_PPG")
    ap.add_argument("--weights-dir", default="weights") # PaPaGei model files
    ap.add_argument("--features-dir", default=None) # where the cache goes
    ap.add_argument("--device", default="cpu") # or "cuda"/"mps" for a GPU
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
    # store_false - timing is on by default, --no-timing turns it off
    ap.add_argument("--no-timing", dest="timing", action="store_false",
                    help="skip measuring feature extraction cost (cv)")
    ap.add_argument("--out-tag", default="",
                    help="suffix for the cv output files, so runs with "
                         "different options don't overwrite each other")
    ap.add_argument("--results-csv", default="results/improvement_results.csv")
    args = ap.parse_args()

    # check the embedding names straight away
    # so a typo fails immediately instead of after a long run
    for e in args.embeddings:
        parse_embedding(e)
    # figures mode needs nothing else (no data, no aeon), so it goes first
    if args.mode == "figures":
        return mode_figures(args)
    # if ROCKET etc. are requested, check aeon is installed before starting
    if args.aeon and args.mode != "timing":
        try:
            import aeon  # noqa: F401
        except ImportError:
            raise SystemExit("The ROCKET-family baselines need aeon: run "
                             "`pip install aeon`, or pass --aeon with nothing "
                             "after it to skip them.")
    # load the train/val/test participants (without the dropped ones)
    splits = D.load_splits(args.data_dir, args.drop_ids)
    # all the methods
    sets = feature_sets(args.embeddings, args.aeon)
    if args.sets:
        # only keep the requested methods, stop if one doesn't exist
        unknown = [x for x in args.sets if x not in sets]
        if unknown:
            raise SystemExit(f"--sets: unknown {unknown}; choose from "
                             f"{list(sets)}")
        # keep references and the figure methods as well (or it would break)
        keep = set(args.sets) | set(args.references) | set(
            args.stat_sets or [])
        sets = {k: v for k, v in sets.items() if k in keep}
    # the references have to be methods that actually run
    for ref in args.references:
        if ref not in sets and ref != "training_mean":
            raise SystemExit(f"--references: unknown feature set {ref}")

    # run the chosen mode (val and test both go to mode_holdout)
    if args.mode == "timing":
        mode_timing(args, splits)
    elif args.mode == "cv":
        mode_cv(args, splits, sets)
    else:
        mode_holdout(args, splits, sets)


if __name__ == "__main__":
    main()
