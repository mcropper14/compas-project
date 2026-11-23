"""Run every method on the shared splits, rebuild results/results.csv, and
write results/summary.md (test + shift comparison and the trade-off table).

    python run_all.py            # everything (~3-5 min on CPU)
    python run_all.py --skip-ensembles
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

import methods  # noqa: F401
from data import RACE_A, RACE_B, describe, load_splits
from eval import RESULTS_CSV, FPR_CAP, clear_results, summary
from methods import baseline, dro, ensembles, fair_lr, group_thresholds, reweighting

OUT = Path(__file__).resolve().parent / "results"
MAIN = ["compas_decile", "compas_capped", "lr_no_race", "lr_with_race", "reweighting_race", "ensemble_soft",
        "fair_lr", "group_thresholds", "fairlearn_expgrad_fpr", "group_dro", "dro_fair"]


def tradeoff_table(df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Per-race rates: what equalizing FPR costs each group in FNR and PPV."""
    d = df[(df["split"] == split) & df["selected"] & df["method"].isin(MAIN)].set_index("method")
    cols = {}
    for k in ("base_rate", "pos_rate", "fpr", "fnr", "ppv"):
        for g in (RACE_A, RACE_B):
            cols[f"{k} {'AA' if g == RACE_A else 'White'}"] = d[f"race_{k}_{g}"]
    return pd.DataFrame(cols).reindex([m for m in MAIN if m in d.index]).round(3)


def main(skip_ensembles: bool = False) -> None:
    torch.set_num_threads(4)
    clear_results()
    splits = load_splits()
    print(describe(splits).round(3).to_string(index=False), "\n")
    steps = [("baseline", baseline.run), ("reweighting", reweighting.run),
             ("group_thresholds", group_thresholds.run),
             ("fair_lr", fair_lr.run), ("dro", dro.run)]
    if not skip_ensembles:
        steps.insert(2, ("ensembles", ensembles.run))
    for name, fn in steps:
        print(f"running {name} ...", flush=True)
        fn(splits)

    df = pd.read_csv(RESULTS_CSV)
    test = summary(split="test").reindex([m for m in MAIN if m in set(df.method)])
    shift = summary(split="shift").reindex(test.index)
    dcols = ["accuracy", "fpr", "race_fpr_diff", "race_fnr_diff", "race_ppv_diff"]
    delta = (shift[dcols] - test[dcols]).round(3)
    params = (df[df.selected & (df.split == "test")].set_index("method")["params"]
              .reindex(test.index))
    lines = [
        "# Results summary (single seed, no CIs: see robustness.md for reported results)", "",
        f"Criterion: equal FPR by race, with overall FPR capped at {FPR_CAP:.0%} on validation "
        "(compas_decile keeps decile >= 5).",
        "Diffs are signed: African-American minus Caucasian (race), Male minus Female (sex). "
        "race_fpr_diff is the primary metric; the rest are reported costs.", "",
        "## Selected configurations", "", params.to_frame().to_markdown(), "",
        "## Test split (same period as training)", "", test.to_markdown(), "",
        "## Shift split (latest 20% by screening date)", "", shift.to_markdown(), "",
        "## Change under shift (shift - test)", "", delta.to_markdown(), "",
        "## Trade-offs (test): per-race rates", "",
        tradeoff_table(df).to_markdown(), "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ensembles", action="store_true")
    main(**vars(ap.parse_args()))
