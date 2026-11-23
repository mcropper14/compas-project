"""Shared evaluation harness. Every method reports through `evaluate()` and
`record()`, so all numbers land in one comparable table (results/results.csv).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import GROUP_NAMES, RACE_A, RACE_B, SEX_A, SEX_B, Split, train_base_rate

RESULTS_CSV = Path(__file__).resolve().parent / "results" / "results.csv"

# Fairness criterion: equal false positive rates across race (predictive equality),
# measured between African-American and Caucasian defendants, subject to a cap.
#
# Operating point: each model's threshold is the lowest one whose overall FPR on
# validation is <= FPR_CAP. All models therefore have about the same false
# positive budget, and none can look fair by flagging nobody or everybody.
# The cap is enforced on validation; test/shift FPR may differ and is reported.
# (The unmodified COMPAS rule, decile >= 5, is the one exception.)
FPR_CAP = 0.25

# Base rate on train; used only inside ensembles.py, where each voter needs a
# cut-off before the votes are aggregated.
THRESHOLD = train_base_rate()

# Model selection, validation only: among candidates with race FPR gap <=
# GAP_TOLERANCE take the highest accuracy; if none qualify, the smallest gap.
# Candidates that don't beat majority-class accuracy are skipped if possible.
GAP_TOLERANCE = 0.05

RECORD = True   # robustness.py turns this off so it doesn't touch results.csv


def capped_threshold(scores: np.ndarray, y: np.ndarray, cap: float | None = None) -> float:
    """Lowest threshold t such that FPR of (score >= t) is <= cap."""
    cap = FPR_CAP if cap is None else cap
    neg = np.sort(scores[y == 0])
    candidates = np.append(np.unique(scores), np.inf)
    fpr = 1 - np.searchsorted(neg, candidates, side="left") / len(neg)
    return float(candidates[np.argmax(fpr <= cap)])


def predict(scores: np.ndarray, split: Split, threshold) -> np.ndarray:
    """threshold: one float, or {race: float} for race-specific thresholds."""
    if isinstance(threshold, dict):
        threshold = np.array([threshold[r] for r in split.race])
    return (scores >= threshold).astype(int)


def _rates(y: np.ndarray, pred: np.ndarray) -> dict:
    pos, neg = y == 1, y == 0
    flagged = pred == 1
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else np.nan,
        "pos_rate": float(flagged.mean()) if len(y) else np.nan,
        "fpr": float(flagged[neg].mean()) if neg.any() else np.nan,
        "fnr": float((~flagged[pos]).mean()) if pos.any() else np.nan,
        "ppv": float(y[flagged].mean()) if flagged.any() else np.nan,   # calibration-style
        "acc": float((pred == y).mean()) if len(y) else np.nan,
    }


def get_scores(model, split: Split) -> np.ndarray:
    p = np.asarray(model.predict_proba(split.X))
    return p[:, 1] if p.ndim == 2 else p


def evaluate_scores(scores: np.ndarray, split: Split, threshold) -> dict:
    """Metrics from P(recid) scores. *_gap is |group A - group B|; *_diff is A - B."""
    y = split.y
    pred = predict(scores, split, threshold)
    overall = _rates(y, pred)
    m = {"accuracy": float((pred == y).mean()), "auc": float(roc_auc_score(y, scores)),
         "pos_rate": float(pred.mean()), "majority_acc": float(max(y.mean(), 1 - y.mean())),
         "fpr": overall["fpr"], "fnr": overall["fnr"], "ppv": overall["ppv"], "n": len(y),
         "threshold": json.dumps(threshold) if isinstance(threshold, dict) else threshold}

    for attr, values, a, b in (("race", split.race, RACE_A, RACE_B), ("sex", split.sex, SEX_A, SEX_B)):
        ra, rb = (_rates(y[values == g], pred[values == g]) for g in (a, b))
        for k in ("fpr", "fnr", "ppv", "pos_rate", "base_rate"):
            m[f"{attr}_{k}_{a}"], m[f"{attr}_{k}_{b}"] = ra[k], rb[k]
        for k in ("fpr", "fnr", "ppv", "pos_rate"):
            m[f"{attr}_{k}_diff"] = ra[k] - rb[k]
            m[f"{attr}_{k}_gap"] = abs(ra[k] - rb[k])
        m[f"{attr}_eo_gap"] = m[f"{attr}_fpr_gap"] + m[f"{attr}_fnr_gap"]

    group_acc = {GROUP_NAMES[g]: _rates(y[split.group == g], pred[split.group == g])["acc"]
                 for g in np.unique(split.group)}
    m["worst_group_acc"] = float(min(group_acc.values()))
    m["worst_group"] = min(group_acc, key=group_acc.get)
    return m


def evaluate(model, split: Split, threshold) -> dict:
    """evaluate(model, split): model needs predict_proba(X) (sklearn style or 1-D)."""
    return evaluate_scores(get_scores(model, split), split, threshold)


def fit_threshold(model, val: Split) -> float:
    return capped_threshold(get_scores(model, val), val.y)


def select(candidates: list[tuple[dict, object]], tol: float = GAP_TOLERANCE):
    """Apply the selection rule above. candidates = [(val_metrics, obj)]."""
    viable = [c for c in candidates if c[0]["accuracy"] > c[0]["majority_acc"]] or candidates
    ok = [c for c in viable if c[0]["race_fpr_gap"] <= tol]
    if ok:
        return max(ok, key=lambda c: c[0]["accuracy"])
    return min(viable, key=lambda c: c[0]["race_fpr_gap"])


def record(method: str, split_name: str, metrics: dict, params: dict | None = None,
           selected: bool = True, path: Path = RESULTS_CSV) -> None:
    """Add (or replace) one row in the shared results table (columns are unioned)."""
    if not RECORD:
        return
    row = {"method": method, "split": split_name, "selected": selected,
           "params": json.dumps(params or {}, sort_keys=True), **metrics}
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        # rerunning a method replaces its rows instead of duplicating them
        same = ((old["method"] == method) & (old["split"] == split_name)
                & (old["selected"] == selected))
        if not selected:
            same &= old["params"] == row["params"]
        new = pd.concat([old[~same], new], ignore_index=True)
    new.to_csv(path, index=False)


def clear_results(method_prefix: str | None = None, path: Path = RESULTS_CSV) -> None:
    """Remove all rows (or rows whose method starts with a prefix) before a rerun."""
    if not path.exists():
        return
    if method_prefix is None:
        path.unlink()
        return
    df = pd.read_csv(path)
    df[~df["method"].str.startswith(method_prefix)].to_csv(path, index=False)


def evaluate_all(model, splits: dict[str, Split], method: str, params: dict | None = None,
                 selected: bool = True, names=("val", "test", "shift"),
                 threshold=None) -> dict[str, dict]:
    """Evaluate on val/test/shift and record each row. Returns {split: metrics}.
    threshold=None -> set on validation to meet FPR_CAP."""
    if threshold is None:
        threshold = fit_threshold(model, splits["val"])
    out = {}
    for name in names:
        out[name] = evaluate(model, splits[name], threshold)
        record(method, name, out[name], params, selected)
    return out


SUMMARY_COLS = ["accuracy", "auc", "fpr", "fnr", "race_fpr_diff", "race_fnr_diff",
                "race_ppv_diff", "race_pos_rate_diff", "sex_fpr_diff"]


def summary(path: Path = RESULTS_CSV, split: str = "test") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["split"] == split) & df["selected"]]
    return df.set_index("method")[SUMMARY_COLS].round(3)
