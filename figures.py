"""Figures for the report, from results/robustness.csv (run robustness.py first).
All points are ten-seed means with 95% CIs.

  results/pareto.png  accuracy vs signed race FPR gap: lambda sweeps and other methods (test)
  results/shift.png         race FPR gap and accuracy per method, test vs later period
  results/shift_charge.png  the same for felony test vs misdemeanors
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from eval import FPR_CAP, GAP_TOLERANCE, RESULTS_CSV  # noqa: E402

OUT = RESULTS_CSV.parent
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BASELINE = "lr_no_race"
LABELS = {
    "lr_no_race": "LR, no race (baseline)", "compas_decile": "COMPAS (decile ≥ 5)",
    "compas_capped": "COMPAS (FPR cap)", "lr_with_race": "LR, with race",
    "reweighting_race": "Reweighting", "ensemble_soft": "Soft-vote ensemble",
    "group_thresholds": "Per-race thresholds", "fair_lr": "Fair LR (λ penalty)",
    "group_dro": "Group DRO", "dro_fair": "Group DRO + λ penalty",
    "fairlearn_expgrad_fpr": "Fairlearn (low FPR)",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _wide(d: pd.DataFrame, index) -> pd.DataFrame:
    """Columns (stat, metric) for the mean and CI ends."""
    return d.pivot(index=index, columns="metric", values=["mean", "lo", "hi"])


def _errs(w: pd.DataFrame, metric: str):
    # a percentile CI can miss the mean of a biased statistic; keep whiskers >= 0
    return [(w["mean"][metric] - w["lo"][metric]).clip(lower=0), (w["hi"][metric] - w["mean"][metric]).clip(lower=0)]


def pareto(df: pd.DataFrame, split: str = "test") -> None:
    d = df[df["split"] == split]
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=SURFACE)
    _style(ax)
    ax.axvline(0, color=INK2, lw=1)
    cand = d[d["kind"] == "candidate"]
    for method, color, eta in (("fair_lr", BLUE, None), ("dro_fair", ORANGE, 0.01)):
        s = cand[(cand["method"] == method) & (cand["eta"].isna() if eta is None else cand["eta"] == eta)]
        w = _wide(s, "lam").sort_index()
        x, y = w["mean"]["race_fpr_diff"], w["mean"]["accuracy"]
        ax.errorbar(x, y, xerr=_errs(w, "race_fpr_diff"), yerr=_errs(w, "accuracy"), fmt="none",
                    ecolor=color, elinewidth=0.8, alpha=0.25, zorder=1)
        ax.plot(x, y, "-o", color=color, lw=2, ms=6, mec=SURFACE, mew=1.2, zorder=3,
                label=LABELS[method] + (f", η={eta}" if eta else "") + f": λ {w.index.min():g}–{w.index.max():g}")
    w = _wide(d[(d["kind"] == "chosen") & (d["cap"] == FPR_CAP)], "method")
    offsets = {"lr_no_race": (6, -64), "compas_capped": (6, 42), "group_thresholds": (6, 8), "lr_with_race": (6, 8)}
    for m, off in offsets.items():
        r = w.loc[[m]]
        ax.errorbar(r["mean"]["race_fpr_diff"].to_numpy(), r["mean"]["accuracy"].to_numpy(), xerr=_errs(r, "race_fpr_diff"),
                    yerr=_errs(r, "accuracy"), fmt="D", ms=6, color=INK2, elinewidth=1.2, capsize=2, zorder=4)
        ax.annotate(LABELS[m], (r["mean"]["race_fpr_diff"].iloc[0], r["mean"]["accuracy"].iloc[0]),
                    xytext=off, textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("Race FPR gap, Black − white (0 = equal FPR)", color=INK, fontsize=10)
    ax.set_ylabel("Accuracy", color=INK, fontsize=10)
    ax.set_title(f"Accuracy vs. FPR gap at a {FPR_CAP:.0%} FPR cap ({split}, 10-seed mean, 95% CI)",
                 color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "pareto.png", dpi=200)
    plt.close(fig)


def shift(df: pd.DataFrame, title: str, labels: tuple[str, str], out: str) -> None:
    d = df[(df["kind"] == "chosen") & (df["cap"] == FPR_CAP)]
    order = [m for m in LABELS if m in set(d["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6), sharey=True, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    for ax, metric, panel in ((axes[0], "race_fpr_diff", "FPR gap, Black − white (0 = equal)"),
                              (axes[1], "accuracy", "Accuracy")):
        _style(ax)
        ax.grid(axis="y", visible=False)
        for off, split, color, label in ((-0.16, "test", BLUE, labels[0]),
                                         (0.16, "shift", ORANGE, labels[1])):
            w = _wide(d[d["split"] == split], "method").loc[order]
            ax.errorbar(w["mean"][metric], [i + off for i in range(len(order))], xerr=_errs(w, metric), fmt="o",
                        ms=7, color=color, mec=SURFACE, mew=1, elinewidth=2, capsize=0, label=label)
        ax.set_title(panel, color=INK, fontsize=10, loc="left")
        ax.set_xlim(*((-0.22, 0.34) if metric == "race_fpr_diff" else (0.60, 0.76)))   # same axes in both figures
    axes[0].axvline(0, color=INK2, lw=1)
    axes[0].set_yticks(range(len(order)), [LABELS[m] for m in order], color=INK, fontsize=9)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK, loc="lower center", ncol=2)
    fig.suptitle(f"{title} (10-seed mean, 95% CI)", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(OUT / "robustness.csv")
    df = df[df["tol"] == GAP_TOLERANCE]
    temporal, charge = df[df["scheme"] == "temporal"], df[df["scheme"] == "charge"]
    pareto(temporal)
    shift(temporal, "Does equal FPR survive a later period?", ("Test (same period)", "Shift (later period)"), "shift.png")
    shift(charge, "Does equal FPR survive a change of charge type?",
          ("Test (felonies)", "Shift (misdemeanors, held out of training)"), "shift_charge.png")
    print("wrote pareto.png, shift.png and shift_charge.png to", OUT)
