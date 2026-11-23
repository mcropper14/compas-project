"""Myra: group distributionally robust optimization + fairness penalty.

Group DRO (Sagawa et al., 2020) minimizes the worst group's loss over the six
race x sex groups instead of the average loss. We optimize the standard online
form: group weights q are updated by exponentiated gradient,
    q_g <- q_g * exp(eta * loss_g),  normalized,
and the model minimizes  sum_g q_g * loss_g + lambda * FPR gap.

lambda = 0 gives plain group DRO; eta controls how fast q concentrates on the
worst group. Both are chosen on validation with eval.select(). The point of the
method is the shift experiment (run_all.py / figures.py): does robustness keep
the FPR gap small on the later-in-time shift split?
"""
from __future__ import annotations

import methods  # noqa: F401
from data import GROUP_NAMES, load_splits
from methods.fair_lr import LAMBDAS, sweep

ETAS = [0.01, 0.1]


def run(splits=None) -> dict:
    splits = splits or load_splits()
    out = {"dro_only": sweep(splits, "group_dro", [{"lam": 0.0, "dro_eta": e} for e in ETAS])}
    out["dro_fair"] = sweep(splits, "dro_fair",
                            [{"lam": lam, "dro_eta": e} for lam in LAMBDAS if lam > 0 for e in ETAS])
    return out


if __name__ == "__main__":
    for name, res in run().items():
        t, s = res["metrics"]["test"], res["metrics"]["shift"]
        print(f"{name}: {res['config']}")
        for split, m in (("test", t), ("shift", s)):
            print(f"  {split:5s} acc={m['accuracy']:.3f} worst-group acc={m['worst_group_acc']:.3f} "
                  f"({m['worst_group']}) race FPR gap={m['race_fpr_gap']:.3f} FNR gap={m['race_fnr_gap']:.3f}")
    print("groups:", GROUP_NAMES)
