"""Jihyun: baseline logistic regression (with / without race), COMPAS decile
comparison, and the proxy check for the data-bias section."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import methods  # noqa: F401  (path setup)
from data import RACE_A, RACE_B, load_splits
from eval import evaluate_all, evaluate_scores, record


def train_lr(split, sample_weight=None) -> LogisticRegression:
    return LogisticRegression(max_iter=2000).fit(split.X, split.y, sample_weight=sample_weight)


def compas_decile(splits) -> dict:
    """COMPAS itself: decile >= 5 (Medium/High) counts as 'high risk', as in ProPublica."""
    out = {}
    for name in ("val", "test", "shift"):
        s = splits[name]
        out[name] = evaluate_scores(s.decile / 10.0, s, threshold=0.5)
        record("compas_decile", name, out[name], {"rule": "decile>=5"})
    return out


class CompasScore:
    """Adapter so COMPAS deciles can be evaluated like a model."""
    def __init__(self, splits):
        self.by_id = {id(s.X): s.decile / 10.0 for s in splits.values()}

    def predict_proba(self, X):
        return self.by_id[id(X)]


def proxy_check(splits) -> pd.DataFrame:
    """How well do the race-blind features predict race (African-American vs Caucasian)?"""
    rows = []
    tr, te = splits["train"], splits["test"]
    mtr, mte = np.isin(tr.race, [RACE_A, RACE_B]), np.isin(te.race, [RACE_A, RACE_B])
    clf = LogisticRegression(max_iter=2000).fit(tr.X[mtr], tr.race[mtr] == RACE_A)
    auc = roc_auc_score(te.race[mte] == RACE_A, clf.predict_proba(te.X[mte])[:, 1])
    rows.append({"feature": "ALL (logistic regression)", "auc_predicting_race": auc})
    for col in tr.X.columns:
        a = roc_auc_score(te.race[mte] == RACE_A, te.X[col][mte])
        rows.append({"feature": col, "auc_predicting_race": max(a, 1 - a)})
    return pd.DataFrame(rows).round(3)


def run(splits=None) -> dict:
    splits = splits or load_splits()
    results = {"compas_decile": compas_decile(splits),
               # same deciles, cut at the shared FPR cap instead of at 5
               "compas_capped": evaluate_all(CompasScore(splits), splits, "compas_capped",
                                             {"rule": "decile, FPR cap"})}
    results["lr_no_race"] = evaluate_all(train_lr(splits["train"]), splits, "lr_no_race")
    with_prot = load_splits(include_protected=True)
    results["lr_with_race"] = evaluate_all(train_lr(with_prot["train"]), with_prot,
                                           "lr_with_race", {"features": "+race,+sex"})
    return results


if __name__ == "__main__":
    splits = load_splits()
    for name, res in run(splits).items():
        t = res["test"]
        print(f"{name:14s} acc={t['accuracy']:.3f} auc={t['auc']:.3f} "
              f"FPR {RACE_A}={t['race_fpr_African-American']:.3f} {RACE_B}={t['race_fpr_Caucasian']:.3f} "
              f"FNR {RACE_A}={t['race_fnr_African-American']:.3f} {RACE_B}={t['race_fnr_Caucasian']:.3f} "
              f"PPV gap={t['race_ppv_gap']:.3f}")
    print("\nProxy check (test AUC of predicting African-American vs Caucasian):")
    print(proxy_check(splits).to_string(index=False))
