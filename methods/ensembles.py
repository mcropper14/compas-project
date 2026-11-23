"""Justin: ensemble ablations from the proposal's Project Plan.

1. Train several classifier families.
2. Aggregate them: hard majority vote, soft (probability) average, and a
   weighted vote with weights from validation accuracy.
3. Repeat everything with Kamiran-Calders reweighing to see whether ensembling
   changes fairness gaps or only accuracy.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import methods  # noqa: F401
from data import SEED, load_splits
from eval import THRESHOLD, evaluate, evaluate_all
from methods.reweighting import weights_for


def base_models(seed: int = SEED):
    return {
        "logreg": LogisticRegression(max_iter=2000),
        "tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=seed),
        "forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, random_state=seed),
        "gboost": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed),
        "naive_bayes": GaussianNB(),
    }


class VotingEnsemble:
    """Aggregates fitted models. mode: 'majority' | 'soft' | 'weighted'."""

    def __init__(self, models: dict, mode: str, weights: dict | None = None):
        self.models, self.mode = models, mode
        self.weights = weights or {k: 1.0 for k in models}

    def predict_proba(self, X) -> np.ndarray:
        names = list(self.models)
        P = np.column_stack([np.asarray(self.models[n].predict_proba(X))[:, 1] for n in names])
        w = np.array([self.weights[n] for n in names])
        w = w / w.sum()
        if self.mode == "majority":
            # Fraction of votes for "high risk". Rescaled so a strict majority
            # (> half the models) lands at or above THRESHOLD.
            return self._rescale((P >= THRESHOLD).mean(axis=1))
        if self.mode == "soft":
            return P.mean(axis=1)
        if self.mode == "weighted":
            return self._rescale((P >= THRESHOLD).astype(float) @ w)
        raise ValueError(self.mode)

    @staticmethod
    def _rescale(vote_share: np.ndarray) -> np.ndarray:
        # Map vote share 0.5 -> THRESHOLD so eval's threshold means "majority".
        return np.where(vote_share > 0.5, THRESHOLD + (1 - THRESHOLD) * (vote_share - 0.5) * 2,
                        THRESHOLD * vote_share * 2 * 0.999)


def run(splits=None) -> dict:
    splits = splits or load_splits()
    tr = splits["train"]
    results = {}
    for rw in (False, True):
        tag = "rw_" if rw else ""
        sw = weights_for(tr, "race") if rw else None
        fitted = {}
        for name, model in base_models().items():
            fitted[name] = model.fit(tr.X, tr.y, sample_weight=sw)
            results[f"{tag}{name}"] = evaluate_all(fitted[name], splits, f"single_{tag}{name}",
                                                   {"reweighted": rw})
        val_acc = {n: evaluate(m, splits["val"], THRESHOLD)["accuracy"] for n, m in fitted.items()}
        for mode in ("majority", "soft", "weighted"):
            ens = VotingEnsemble(fitted, mode, val_acc if mode == "weighted" else None)
            results[f"{tag}ens_{mode}"] = evaluate_all(ens, splits, f"ensemble_{tag}{mode}",
                                                       {"reweighted": rw, "mode": mode})
    return results


if __name__ == "__main__":
    for name, res in run().items():
        t = res["test"]
        print(f"{name:20s} acc={t['accuracy']:.3f} auc={t['auc']:.3f} race FPR gap={t['race_fpr_gap']:.3f} "
              f"FNR gap={t['race_fnr_gap']:.3f} EO gap={t['race_eo_gap']:.3f}")
