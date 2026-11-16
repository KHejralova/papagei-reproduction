"""
Improvement experiment 1: how should a 2.1 s PPG-BP recording be fed to a
model pre-trained on 10 s segments?

The paper (Appendix B) and the released notebook zero-pad each recording
(~263 samples at 125 Hz) to 1250 samples, so ~79% of every input is zeros.
This script keeps every other step identical and compares the input
representations in representations.py. For each one the participant's
recording embeddings are averaged, then evaluated with the shared
evaluate_features() pipeline (StandardScaler + Ridge, paper alpha grid,
4-fold CV on train, participant-level MAE with bootstrap CI).

Use --split val while choosing (the repeated-CV comparison of the padding
choices is done with 02_feature_fusion.py, see README).

Example (from the repo root):
    python improvements/01_input_representation.py --data-dir data_PPG
"""

import argparse
import json
from pathlib import Path

import data as D
import pandas as pd
from embeddings import MODEL_FILES, cached_embeddings
from evaluation import training_mean_result
from representations import ALL_VARIANTS

from evaluation_utils import evaluate_features
from results_utils import record_result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data_PPG",
                    help="folder containing the unzipped 'Data File' folder")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--models", nargs="+", default=["papagei_s", "papagei_p"],
                    choices=list(MODEL_FILES))
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                    choices=ALL_VARIANTS)
    ap.add_argument("--targets", nargs="+", default=D.TARGETS,
                    choices=D.TARGETS)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="evaluation split (use val until choices are frozen)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--features-dir", default=None,
                    help="where to cache embeddings "
                         "(default: <data-dir>/features_improvements)")
    ap.add_argument("--results-csv", default="results/improvement_results.csv")
    ap.add_argument("--overwrite-features", action="store_true")
    ap.add_argument("--no-weights", action="store_true",
                    help="random weights; only for testing the pipeline")
    args = ap.parse_args()

    features_dir = Path(args.features_dir or
                        Path(args.data_dir) / "features_improvements")
    features_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.results_csv).parent / "improvements"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data: the supplied split, and the preprocessed recordings.
    print("Loading metadata and split ...")
    splits = D.load_splits(args.data_dir)
    ids = {k: v["subject_ID"].astype(int).tolist() for k, v in splits.items()}
    needed = ids["train"] + ids[args.split]
    print(f"Preprocessing {len(needed) * 3} recordings ...")
    recordings = D.load_preprocessed_recordings(args.data_dir, needed)
    lengths = sorted({len(r) for recs in recordings.values() for r in recs})
    print(f"Recording lengths at 125 Hz: {lengths} samples (zero padding "
          f"would make {1 - lengths[0] / D.MODEL_INPUT_LENGTH:.0%} of the "
          "input zeros)")

    procedure = f"input_representation_paper_grid_{args.split}_seed42"
    if args.no_weights:
        procedure = "PIPELINE_TEST_random_weights_" + procedure
    summary = []

    def log(result, model, variant, target):
        """Append to the shared results CSV and to this run's summary."""
        name = model if variant == "-" else f"{model}_{variant}"
        record_result(result=result, model_name=name, target=target,
                      n_train=len(ids["train"]), procedure=procedure,
                      csv_path=args.results_csv)
        summary.append({"model": model, "variant": variant, "target": target,
                        "mae": result["mae"],
                        "ci_lower": result["lower_bound_mae"],
                        "ci_upper": result["upper_bound_mae"],
                        "params": json.dumps(result["parameters"])})

    # 2. Reference: predict the training mean (same split, same bootstrap).
    for target in args.targets:
        y_tr = splits["train"][target].to_numpy(float)
        y_ev = splits[args.split][target].to_numpy(float)
        log(training_mean_result(y_tr, y_ev, args.split), "training_mean",
            "-", target)

    # 3. Every model x input representation: embed, then the shared ridge.
    for model_name in args.models:
        for variant in args.variants:
            X = cached_embeddings(
                model_name, variant,
                {"train": ids["train"], args.split: ids[args.split]},
                lambda: recordings, features_dir, args.weights_dir,
                args.device, load_weights=not args.no_weights,
                overwrite=args.overwrite_features)
            for target in args.targets:
                y_tr = splits["train"][target].to_numpy(float)
                y_ev = splits[args.split][target].to_numpy(float)
                res = evaluate_features(
                    X_train=X["train"], y_train=y_tr,
                    X_eval=X[args.split], y_eval=y_ev,
                    evaluation_split=args.split, bootstrap_seed=42)
                log(res, model_name, variant, target)
                pd.DataFrame({
                    "subject_ID": ids[args.split],
                    "y_true": res["y_test"], "y_pred": res["y_pred"],
                }).to_csv(out_dir / f"pred_{model_name}_{variant}_{target}"
                          f"_{args.split}.csv", index=False)

    # 4. Summary table.
    table = pd.DataFrame(summary)
    out = out_dir / f"input_representation_summary_{args.split}.csv"
    table.to_csv(out, index=False)
    print("\nMAE by model / input representation "
          f"({args.split}, n={len(ids[args.split])}):")
    print(table.pivot_table(index=["model", "variant"], columns="target",
                            values="mae").round(2).to_string())
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
