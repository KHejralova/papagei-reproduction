"""
Figures and the results table for the report, built from the saved outputs
of experiments 1 and 2 (nothing is refitted).

  fig_cv_mae          CV MAE (mean +/- sd over repeats) per method, one
                      panel per target, Dummy baseline as a dashed line
  fig_padding         experiment 1: validation MAE per input representation
  fig_test_pred       test set: predicted vs true systolic BP for PaPaGei-S,
                      the main method and MiniROCKET
  fig_cost            accuracy vs fit+predict time (CV, systolic BP)
  results_table       CV MAE and test MAE [95% CI] for the key methods
                      (.csv, .md for the README, .tex for the report)

Saved to results/improvements/report/ as .png (300 dpi) and .pdf (vector,
for LaTeX). The CD diagrams and MCMs are made by 02_feature_fusion.py.

Example (from the repo root, after 02 --mode cv and --mode test):
    python improvements/03_report_figures.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import data as D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plots import DISPLAY

# Methods shown in the figures and the table, in this order.
KEY_METHODS = ["training_mean", "demo", "handcrafted", "papagei_s",
               "papagei_p", "papagei_s+demo_concat",
               "papagei_s+handcrafted_stacked", "minirocket", "rocket",
               "hydra", "quant"]
MAIN = "papagei_s+handcrafted_stacked"

# Colour = method family (identity), from the validated reference palette;
# three hues only, so every pair stays distinguishable for colour-blind
# readers. The main method gets a bigger, filled marker.
FAMILY_COLOR = {"No deep learning": "#1baf7a",   # aqua
                "PaPaGei": "#2a78d6",            # blue
                "Course TSER (Lab 8)": "#eb6834"}  # orange
BASELINE_GREY = "#8c8c8c"
INK, MUTED = "#0b0b0b", "#52514e"
TARGET_TITLE = {"sysbp": "Systolic BP", "diasbp": "Diastolic BP",
                "hr": "Heart rate"}


def family(method):
    if method.startswith("papagei"):
        return "PaPaGei"
    if method in ("rocket", "minirocket", "hydra", "quant"):
        return "Course TSER (Lab 8)"
    return "No deep learning"


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def family_legend(fig, extra=()):
    handles = [plt.Line2D([], [], ls="", marker="o", ms=6, color=c, label=f)
               for f, c in FAMILY_COLOR.items()]
    handles += list(extra)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.06))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_cv_mae(summary, out_dir, n_repeats):
    methods = [m for m in KEY_METHODS[1:] if m in set(summary.feature_set)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
    y = np.arange(len(methods))[::-1]  # first method at the top
    for ax, t in zip(axes, D.TARGETS):
        st = summary[summary.target == t].set_index("feature_set")
        for yi, m in zip(y, methods):
            main = m == MAIN
            ax.errorbar(st.at[m, "mae_mean"], yi,
                        xerr=st.at[m, "mae_sd_over_repeats"],
                        fmt="o", ms=7 if main else 5,
                        color=FAMILY_COLOR[family(m)],
                        mfc=FAMILY_COLOR[family(m)] if main else "white",
                        mew=1.6, elinewidth=1.2, capsize=0, zorder=3)
        dummy = st.at["training_mean", "mae_mean"]
        ax.axvline(dummy, color=BASELINE_GREY, ls="--", lw=1.2, zorder=1)
        ax.set_title(f"{TARGET_TITLE[t]} ({D.UNITS[t]})")
        ax.set_xlabel("MAE (lower is better)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [DISPLAY.get(m, m) for m in methods])
    for lbl in axes[0].get_yticklabels():
        if lbl.get_text() == DISPLAY[MAIN]:
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    dummy_handle = plt.Line2D([], [], color=BASELINE_GREY, ls="--",
                              label="Dummy (predict the mean)")
    family_legend(fig, [dummy_handle])
    fig.suptitle(f"Repeated 5-fold CV on train + validation "
                 f"({n_repeats} repeats); error bars = SD over repeats",
                 fontsize=9, color=MUTED, y=1.0)
    fig.tight_layout()
    save(fig, out_dir, "fig_cv_mae")


def fig_padding(log, out_dir):
    rows = log[log.procedure == "input_representation_paper_grid_val_seed42"]
    if rows.empty:
        print("  (skipped fig_padding: no experiment-1 validation results)")
        return
    rows = rows.drop_duplicates(["model", "target"], keep="last").copy()
    rows = rows[rows.model.str.match(r"papagei_[sp]_") |
                (rows.model == "training_mean")]
    rows["base"] = rows.model.str[:9]
    rows["variant"] = rows.model.str[10:]
    order = ["zero_pad", "native", "tile", "tile_cycles", "reflect",
             "concat3_tile"]
    variants = [v for v in order if v in set(rows.variant)]
    # both are PaPaGei models: same family colour, told apart by marker shape
    marker = {"papagei_s": {"marker": "o", "mfc": FAMILY_COLOR["PaPaGei"]},
              "papagei_p": {"marker": "s", "mfc": "white"}}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    y = np.arange(len(variants))[::-1]
    for ax, t in zip(axes, D.TARGETS):
        rt = rows[rows.target == t]
        for base, dy in (("papagei_s", 0.12), ("papagei_p", -0.12)):
            r = rt[rt.base == base].set_index("variant")
            ys = [yi + dy for yi, v in zip(y, variants) if v in r.index]
            xs = [r.at[v, "mae"] for v in variants if v in r.index]
            ax.plot(xs, ys, ls="", ms=5, color=FAMILY_COLOR["PaPaGei"],
                    mew=1.6, label=DISPLAY[base], zorder=3, **marker[base])
        dummy = rt[rt.model == "training_mean"]["mae"]
        if len(dummy):
            ax.axvline(dummy.iloc[0], color=BASELINE_GREY, ls="--", lw=1.2)
        ax.set_title(f"{TARGET_TITLE[t]} ({D.UNITS[t]})")
        ax.set_xlabel("Validation MAE (lower is better)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, variants)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=BASELINE_GREY, ls="--"))
    labels.append("Dummy (predict the mean)")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Experiment 1: input representation, validation set "
                 "(n = 41)", fontsize=9, color=MUTED, y=1.0)
    fig.tight_layout()
    save(fig, out_dir, "fig_padding")


def fig_test_pred(preds, test_summary, out_dir, target="sysbp"):
    methods = [m for m in ("papagei_s", MAIN, "minirocket")
               if m in set(preds.feature_set)]
    p = preds[preds.target == target]
    lo = min(p.y_true.min(), p.y_pred.min()) - 5
    hi = max(p.y_true.max(), p.y_pred.max()) + 5
    dummy = p[p.feature_set == "training_mean"].y_pred.iloc[0]
    fig, axes = plt.subplots(1, len(methods), figsize=(7.2, 2.7),
                             sharex=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), methods):
        pm = p[p.feature_set == m]
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, label="Perfect (y = x)")
        ax.axhline(dummy, color=BASELINE_GREY, ls="--", lw=1.2,
                   label="Dummy (predict the mean)")
        ax.plot(pm.y_true, pm.y_pred, "o", ms=4.5,
                color=FAMILY_COLOR[family(m)], mfc="white", mew=1.3)
        mae = test_summary.set_index(["feature_set", "target"]).at[
            (m, target), "mae"]
        ax.set_title(f"{DISPLAY.get(m, m)}\nMAE {mae:.1f} {D.UNITS[target]}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"True {TARGET_TITLE[target]} ({D.UNITS[target]})")
    np.atleast_1d(axes)[0].set_ylabel("Predicted")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.17))
    fig.suptitle(f"Test set (n = {p.subject_ID.nunique()}), trained on "
                 "train + validation", fontsize=9, color=MUTED, y=1.02)
    fig.tight_layout()
    save(fig, out_dir, f"fig_test_pred_{target}")


# The five methods the accuracy-vs-cost argument is about (the full
# comparison is in fig_cv_mae and the MCMs). Label offsets in points.
COST_METHODS = {"handcrafted": (-3, 0, "right"),
                "papagei_s": (6, 0, "left"),
                MAIN: (7, -2, "left"),
                "minirocket": (6, 8, "center"),
                "rocket": (0, 8, "center")}


def fig_cost(summary, out_dir, target="sysbp"):
    st = summary[summary.target == target].set_index("feature_set")
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for m, (dx, dy, ha) in COST_METHODS.items():
        if m not in st.index:
            continue
        main = m == MAIN
        x, y = st.at[m, "fit_predict_s_per_fold"], st.at[m, "mae_mean"]
        ax.plot(x, y, "o", ms=7 if main else 5,
                color=FAMILY_COLOR[family(m)],
                mfc=FAMILY_COLOR[family(m)] if main else "white", mew=1.6,
                zorder=3)
        ax.annotate(DISPLAY.get(m, m), (x, y), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=7, color=INK,
                    fontweight="bold" if main else "normal")
    ax.axhline(st.at["training_mean", "mae_mean"], color=BASELINE_GREY,
               ls="--", lw=1.2)
    ax.set_xscale("log")
    xs = [st.at[m, "fit_predict_s_per_fold"] for m in COST_METHODS
          if m in st.index]
    ax.set_xlim(min(xs) / 12, max(xs) * 3)  # room for the left-hand label
    ax.set_xlabel("Fit + predict time per fold (s, log scale)")
    ax.set_ylabel(f"CV MAE, {TARGET_TITLE[target]} ({D.UNITS[target]})")
    ax.set_title("Accuracy vs cost (lower left is better)")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=c,
                          mfc="white", mew=1.6, label=f)
               for f, c in FAMILY_COLOR.items()]
    handles.append(plt.Line2D([], [], color=BASELINE_GREY, ls="--",
                              label="Dummy"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out_dir, f"fig_cost_{target}")


# --------------------------------------------------------------------------
# Results table
# --------------------------------------------------------------------------

def results_table(summary, test_summary, out_dir):
    rows = []
    for m in [m for m in KEY_METHODS if m in set(summary.feature_set)]:
        row = {"Method": DISPLAY.get(m, m), "feature_set": m}
        for t in D.TARGETS:
            s = summary[(summary.feature_set == m) &
                        (summary.target == t)].iloc[0]
            row[f"{t}_cv"] = (f"{s.mae_mean:.2f} ± "
                              f"{s.mae_sd_over_repeats:.2f}")
            if test_summary is not None:
                ts = test_summary[(test_summary.feature_set == m) &
                                  (test_summary.target == t)]
                if len(ts):
                    ts = ts.iloc[0]
                    row[f"{t}_test"] = (f"{ts.mae:.2f} "
                                        f"[{ts.ci_lower:.1f}, "
                                        f"{ts.ci_upper:.1f}]")
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "results_table.csv", index=False)

    cols = [c for c in table.columns if c not in ("Method", "feature_set")]
    head = {f"{t}_{k}": f"{TARGET_TITLE[t]} {'CV' if k == 'cv' else 'test'}"
            for t in D.TARGETS for k in ("cv", "test")}
    md = ["| Method | " + " | ".join(head[c] for c in cols) + " |",
          "| --- | " + " | ".join("---" for _ in cols) + " |"]
    for _, r in table.iterrows():
        name = f"**{r.Method}**" if r.feature_set == MAIN else r.Method
        md.append(f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |")
    (out_dir / "results_table.md").write_text("\n".join(md) + "\n")

    tex = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
           "Method & " + " & ".join(head[c] for c in cols) + r" \\",
           r"\midrule"]
    for _, r in table.iterrows():
        name = (rf"\textbf{{{r.Method}}}" if r.feature_set == MAIN
                else r.Method)
        cells = [str(r[c]).replace("±", r"$\pm$") for c in cols]
        tex.append(f"{name} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "results_table.tex").write_text("\n".join(tex) + "\n")
    print("  results_table.csv / .md / .tex")
    print("\n" + "\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-tag", default="",
                    help="same --out-tag as the cv run, if one was used")
    args = ap.parse_args()

    res = Path(args.results_dir)
    imp = res / "improvements"
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_dir = imp / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    style()

    summary = pd.read_csv(imp / f"feature_fusion_cv_summary{tag}.csv")
    n_repeats = pd.read_csv(imp / f"feature_fusion_cv_folds{tag}.csv")[
        "repeat"].nunique()
    test_path = imp / "feature_fusion_summary_test.csv"
    test_summary = pd.read_csv(test_path) if test_path.exists() else None
    print(f"Writing to {out_dir}:")
    fig_cv_mae(summary, out_dir, n_repeats)
    fig_cost(summary, out_dir)
    log_path = res / "improvement_results.csv"
    if log_path.exists():
        fig_padding(pd.read_csv(log_path), out_dir)
    if test_summary is not None:
        fig_test_pred(pd.read_csv(imp / "feature_fusion_predictions_test.csv"),
                      test_summary, out_dir)
    else:
        print("  (skipped fig_test_pred: run 02 --mode test first)")
    results_table(summary, test_summary, out_dir)


if __name__ == "__main__":
    main()
