"""Race-specific thresholds on the race-blind logistic regression.

A simple post-processing comparator (in the spirit of Hardt et al., 2016): one
threshold per race, each set on validation so that group's FPR is <= the cap.
Unlike every other fair method here, it uses race explicitly at decision time.
"""
from __future__ import annotations

import numpy as np

import methods  # noqa: F401
from data import load_splits
from eval import capped_threshold, evaluate_all, get_scores
from methods.baseline import train_lr


def race_thresholds(model, val) -> dict[str, float]:
    s = get_scores(model, val)
    return {str(r): capped_threshold(s[val.race == r], val.y[val.race == r]) for r in np.unique(val.race)}


def run(splits=None) -> dict:
    splits = splits or load_splits()
    model = train_lr(splits["train"])
    t = race_thresholds(model, splits["val"])
    return {"model": model, "threshold": t,
            "metrics": evaluate_all(model, splits, "group_thresholds", {"per": "race"}, threshold=t)}


if __name__ == "__main__":
    for name, m in run()["metrics"].items():
        print(f"{name:5s} acc={m['accuracy']:.3f} FPR diff={m['race_fpr_diff']:+.3f} "
              f"FNR diff={m['race_fnr_diff']:+.3f} PPV diff={m['race_ppv_diff']:+.3f}")
