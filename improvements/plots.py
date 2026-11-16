"""
Statistical comparison figures from Labs 7-8: a Friedman test, a critical
difference (CD) diagram and a multiple comparison matrix (MCM) per target.

The paired scores are the MAE of each CV repeat (10 per method). 
Repeats of one dataset are not independent, so these are exploratory.
"""

import data as D
from scipy import stats

# Short names for the figures
DISPLAY = {"training_mean": "Dummy (mean)", "demo": "Demographics",
           "handcrafted": "Simple features", "papagei_s": "PaPaGei-S",
           "papagei_p": "PaPaGei-P", "papagei_s+demo_concat": "S + demo concat",
           "papagei_s+demo_stacked": "S + demo stacked",
           "papagei_s+handcrafted_stacked": "S + simple stacked",
           "papagei_p+handcrafted_stacked": "P + simple stacked",
           "rocket": "ROCKET", "minirocket": "MiniROCKET", "hydra": "Hydra",
           "quant": "QUANT", "peak_hr": "Peak HR", "morph": "sVRI + SQI"}

# Methods in the figures 
DEFAULT_STAT_SETS = ["training_mean", "demo", "handcrafted", "papagei_s",
                     "papagei_s+demo_concat", "papagei_s+handcrafted_stacked",
                     "rocket", "minirocket", "hydra", "quant"]


def _mcm(scores, title, png_path):
    """Draw and save a multiple comparison matrix with whichever aeon version is installed."""
    import inspect

    import matplotlib.pyplot as plt
    from aeon.visualisation import create_multi_comparison_matrix as mcm
    # look at which arguments the installed version accepts
    params = inspect.signature(mcm).parameters
    # settings that are the same in every version
    common = {"pvalue_test": "wilcoxon",
                  "pvalue_test_params": {"alternative": "two-sided",
                                      "zero_method": "wilcox"},
                  "pvalue_correction": "Holm", "pvalue_threshold": 0.05,
                  "order_stats": "average-statistic", "fig_size": (22, 13),
                  "font_size": 11}
    # lower MAE is better, so this is set in each version's own way
    if "statistic_name" in params:  # aeon >= 1.4
        fig = mcm(scores, statistic_name="MAE", higher_stat_better=False,
                  order_stats_increasing=True, formats=None, **common)
    elif "higher_stat_better" in params:  # aeon 1.3
        fig = mcm(scores, used_statistic="MAE", higher_stat_better=False,
                  order_stats_increasing=True, formats=None, **common)
    else:  # aeon < 1.3 - saves the file itself
        mcm(scores, output_dir=str(png_path.parent) + "/",
            png_savename=png_path.stem, used_statistic="MAE",
            order_win_tie_loss="lower", order_better="increasing", **common)
        plt.close("all")
        return
    fig.suptitle(title, y=1.0)
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stats_figures(per_rep, stat_sets, out_dir, tag):
    """Friedman test, CD diagram and MCM per target on the per-repeat MAE"""
    # skip the figures (but keep the results) if aeon/matplotlib are missing
    try:
        import matplotlib
        matplotlib.use("Agg") # draw to files, no window needed
        import matplotlib.pyplot as plt
        from aeon.visualisation import plot_critical_difference
    except ImportError:
        print("aeon/matplotlib not installed: skipping CD diagrams and MCM")
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # only the methods that were actually run
    present = [s for s in stat_sets if s in set(per_rep["feature_set"])]
    n_rep = per_rep["repeat"].nunique()
    print(f"\nFriedman test + CD diagram + MCM on per-repeat MAE "
          f"({n_rep} paired scores per method, exploratory), methods: "
          f"{present}")
    for t in D.TARGETS:
        # table: one row per repeat, one column per method, values = MAE
        scores = (per_rep[per_rep.target == t]
                  .pivot(index="repeat", columns="feature_set", values="mae")
                  [present])
        # Friedman test 
        chi2, p = stats.friedmanchisquare(*[scores[c] for c in present])
        print(f"  {t}: Friedman chi2 = {chi2:.1f}, p = {p:.2g}")
        # nicer names for the figures
        scores = scores.rename(columns=lambda c: DISPLAY.get(c, c))
        labels = list(scores.columns)
        # CD diagram - methods ordered by average rank
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
        # MCM - every pair of methods compared with each other
        try:
            _mcm(scores, f"{t}: MCM on MAE over {n_rep} CV repeats "
                 "(two-sided Wilcoxon, Holm, alpha = 0.05)",
                 fig_dir / f"mcm_{t}{tag}.png")
        except Exception as exc:  # noqa: BLE001
            print(f"  MCM for {t} failed: {exc!r}")
    print(f"  Saved CD diagrams and MCMs to {fig_dir}")
