"""
Improvement experiment 1: how should a 2.1 s PPG-BP recording be fed to a
model pre-trained on 10 s segments?

The paper (Appendix B) and the released notebook zero-pad each 2.1 s
recording (~262 samples at 125 Hz) to 1250 samples, so ~79% of every input
is zeros. Pre-training never saw inputs like that. This script keeps every
other step identical and compares input representations:

  zero_pad      paper / notebook baseline: centred zero padding to 1250
  native        no padding at all (the encoder averages over time, so any
                length works)
  tile          repeat the recording end-to-end until it is 1250 long
  reflect       centred mirror padding (signal is reflected at its edges)
  concat3_tile  join the participant's 3 recordings (~6.3 s), then tile to
                1250 -> one embedding per participant instead of three

For each representation the 3 recording embeddings are averaged per
participant (as in the notebook), then evaluated with the shared
evaluate_features() pipeline (StandardScaler + Ridge, paper alpha grid,
4-fold CV on train, participant-level MAE with bootstrap CI).

Use --split val while choosing. Run --split test only once, after the choice
of representation is frozen.

Example (from the repo root):
    python improvements/01_input_representation.py --data-dir data_PPG
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppgbp_common as pc  # noqa: E402

from evaluation_utils import evaluate_features  # noqa: E402
from results_utils import record_result  # noqa: E402


L = pc.MODEL_INPUT_LENGTH


# --------------------------------------------------------------------------
# Input representations
# --------------------------------------------------------------------------

def zero_pad(x):
    need = L - len(x)
    left = need // 2
    return np.pad(x, (left, need - left))  # identical to the notebook


def native(x):
    return x


def tile(x):
    return np.resize(x, L)  # np.resize repeats the array cyclically


def reflect(x):
    need = L - len(x)
    left = need // 2
    return np.pad(x, (left, need - left), mode="symmetric")


PER_RECORDING = {
    "zero_pad": zero_pad,
    "native": native,
    "tile": tile,
    "reflect": reflect,
}
PER_PARTICIPANT = {
    "concat3_tile": lambda recs: tile(np.concatenate(recs)),
}
ALL_VARIANTS = list(PER_RECORDING) + list(PER_PARTICIPANT)


def build_inputs(variant, recordings, subject_ids):
    """Return (list of model inputs, list of owning subject ids)."""
    inputs, owners = [], []
    for sid in subject_ids:
        recs = recordings[int(sid)]
        if variant in PER_RECORDING:
            for r in recs:
                x = PER_RECORDING[variant](r)
                assert variant == "native" or len(x) == L
                inputs.append(x.astype(np.float32))
                owners.append(int(sid))
        else:
            inputs.append(PER_PARTICIPANT[variant](recs).astype(np.float32))
            owners.append(int(sid))
    return inputs, owners


def participant_features(model, variant, recordings, subject_ids, device):
    """One 512-d row per participant, in the order of subject_ids."""
    inputs, owners = build_inputs(variant, recordings, subject_ids)
    emb = pc.embed_signals(model, inputs, device=device)
    owners = np.asarray(owners)
    return np.stack([emb[owners == int(s)].mean(axis=0) for s in subject_ids])


# --------------------------------------------------------------------------
# Constant baseline in the same result format as evaluate_features()
# --------------------------------------------------------------------------

def training_mean_result(y_train, y_eval, split, seed=42):
    y_true = np.asarray(y_eval, dtype=float)
    y_pred = np.full_like(y_true, float(np.mean(y_train)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=(500, len(y_true)), replace=True)
    boot = np.abs(y_true - y_pred)[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "parameters": {"constant": float(np.mean(y_train))},
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "lower_bound_mae": float(lo),
        "upper_bound_mae": float(hi),
        "y_test": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "evaluation_split": split,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data_PPG",
                    help="folder containing the unzipped 'Data File' folder")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--models", nargs="+", default=["papagei_s", "papagei_p"],
                    choices=list(pc.MODEL_FILES))
    ap.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                    choices=ALL_VARIANTS)
    ap.add_argument("--targets", nargs="+", default=["sysbp", "diasbp", "hr"],
                    choices=["sysbp", "diasbp", "hr"])
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
    pred_dir = Path(args.results_csv).parent / "improvements"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metadata and split ...")
    splits = pc.split_metadata(pc.load_metadata(args.data_dir))
    ids = {k: v["subject_ID"].astype(int).tolist() for k, v in splits.items()}
    needed = ids["train"] + ids[args.split]

    print(f"Preprocessing {len(needed) * 3} recordings ...")
    recordings = pc.load_preprocessed_recordings(args.data_dir, needed)
    lengths = sorted({len(r) for recs in recordings.values() for r in recs})
    print(f"Recording lengths at 125 Hz: {lengths} samples "
          f"(zero padding would make {1 - lengths[0] / L:.0%} of the input "
          "zeros)")

    procedure = f"input_representation_paper_grid_{args.split}_seed42"
    if args.no_weights:
        procedure = "PIPELINE_TEST_random_weights_" + procedure
    summary = []

    # Constant baseline for reference (same split, same bootstrap seed).
    for target in args.targets:
        y_tr = splits["train"][target].to_numpy(float)
        y_ev = splits[args.split][target].to_numpy(float)
        res = training_mean_result(y_tr, y_ev, args.split)
        record_result(result=res, model_name="training_mean", target=target,
                      n_train=len(y_tr), procedure=procedure,
                      csv_path=args.results_csv)
        summary.append({"model": "training_mean", "variant": "-",
                        "target": target, "mae": res["mae"],
                        "ci_lower": res["lower_bound_mae"],
                        "ci_upper": res["upper_bound_mae"]})

    for model_name in args.models:
        model = pc.build_model(model_name, args.weights_dir, args.device,
                               load_weights=not args.no_weights)
        for variant in args.variants:
            tag = "random" if args.no_weights else "pretrained"
            cache = features_dir / f"{model_name}_{tag}_{variant}.npz"
            X_by_split = {}
            if cache.exists() and not args.overwrite_features:
                data = np.load(cache)
                for s in ("train", "val", "test"):
                    if f"X_{s}" in data and \
                            data[f"ids_{s}"].tolist() == ids[s]:
                        X_by_split[s] = data[f"X_{s}"]
            for s in ("train", args.split):
                if s not in X_by_split:
                    print(f"Embedding {model_name} / {variant} / {s} ...")
                    X_by_split[s] = participant_features(
                        model, variant, recordings, ids[s], args.device)
            np.savez(cache, **{f"X_{s}": X for s, X in X_by_split.items()},
                     **{f"ids_{s}": np.array(ids[s]) for s in X_by_split})

            for target in args.targets:
                y_tr = splits["train"][target].to_numpy(float)
                y_ev = splits[args.split][target].to_numpy(float)
                res = evaluate_features(
                    X_train=X_by_split["train"], y_train=y_tr,
                    X_eval=X_by_split[args.split], y_eval=y_ev,
                    evaluation_split=args.split, bootstrap_seed=42)
                record_result(result=res,
                              model_name=f"{model_name}_{variant}",
                              target=target, n_train=len(y_tr),
                              procedure=procedure,
                              csv_path=args.results_csv)
                pd.DataFrame({
                    "subject_ID": ids[args.split],
                    "y_true": res["y_test"], "y_pred": res["y_pred"],
                }).to_csv(pred_dir / f"pred_{model_name}_{variant}_{target}"
                          f"_{args.split}.csv", index=False)
                summary.append({"model": model_name, "variant": variant,
                                "target": target, "mae": res["mae"],
                                "ci_lower": res["lower_bound_mae"],
                                "ci_upper": res["upper_bound_mae"],
                                "params": json.dumps(res["parameters"])})

    table = pd.DataFrame(summary)
    out = pred_dir / f"input_representation_summary_{args.split}.csv"
    table.to_csv(out, index=False)
    print("\nMAE by model / input representation "
          f"({args.split}, n={len(ids[args.split])}):")
    print(table.pivot_table(index=["model", "variant"], columns="target",
                            values="mae").round(2).to_string())
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
