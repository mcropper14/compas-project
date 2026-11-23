"""Justin: Kamiran & Calders (2012) reweighing.

Each training sample in (group g, label y) gets weight
    w = P(g) * P(y) / P(g, y)
so that group and label are statistically independent in the weighted data.
"""
from __future__ import annotations

import numpy as np

import methods  # noqa: F401
from data import load_splits
from eval import evaluate_all
from methods.baseline import train_lr


def kamiran_calders_weights(group: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = np.empty(len(y), dtype=float)
    for g in np.unique(group):
        for label in (0, 1):
            mask = (group == g) & (y == label)
            if mask.any():
                w[mask] = (group == g).mean() * (y == label).mean() / mask.mean()
    return w


def weights_for(split, by: str = "race") -> np.ndarray:
    group = {"race": split.race, "race_sex": split.group}[by]
    return kamiran_calders_weights(group, split.y)


def run(splits=None) -> dict:
    splits = splits or load_splits()
    results = {}
    for by in ("race", "race_sex"):
        model = train_lr(splits["train"], sample_weight=weights_for(splits["train"], by))
        results[by] = evaluate_all(model, splits, f"reweighting_{by}", {"groups": by})
    return results


if __name__ == "__main__":
    for by, res in run().items():
        t = res["test"]
        print(f"reweighting[{by}] acc={t['accuracy']:.3f} race FPR gap={t['race_fpr_gap']:.3f} "
              f"FNR gap={t['race_fnr_gap']:.3f}")
