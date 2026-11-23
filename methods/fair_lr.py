"""William: fairness-constrained logistic regression.

Objective:  BCE(w) + lambda * |FPR_A - FPR_B|
with the differentiable surrogate FPR_g ~ mean p over (g, y=0), where
A/B = African-American / Caucasian. The decision threshold is set afterwards on
validation (eval.FPR_CAP). lambda is swept and chosen on validation with
eval.select(); every candidate is recorded for the trade-off plot. Race is used
only in training; the model never sees it at prediction time.

Fairlearn's ExponentiatedGradient + FalsePositiveRateParity is the reference.

`train_penalized_lr` also implements group DRO (dro_eta != None); see dro.py.
"""
from __future__ import annotations

import numpy as np
import torch

import methods  # noqa: F401
from sklearn.linear_model import LogisticRegression

from data import RACE_A, RACE_B, SEED, load_splits
from eval import evaluate, evaluate_all, fit_threshold, select

LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 5.0]
OVERSAMPLE = [1.0, 1.25, 1.5, 1.75, 2.0]
STEPS, LR, WEIGHT_DECAY = 1500, 0.05, 1e-3


class TorchLogReg:
    def __init__(self, linear: torch.nn.Linear):
        self.linear = linear

    def predict_proba(self, X) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(X, dtype=np.float32))
            return torch.sigmoid(self.linear(x)).squeeze(1).numpy()


def fpr_penalty(p, y, race) -> torch.Tensor:
    a, b = torch.as_tensor(race == RACE_A), torch.as_tensor(race == RACE_B)
    neg = y == 0
    return (p[a & neg].mean() - p[b & neg].mean()).abs()


def train_penalized_lr(split, lam: float, dro_eta: float | None = None,
                       steps: int = STEPS, seed: int = SEED) -> TorchLogReg:
    """Full-batch training. dro_eta=None -> average loss (ERM);
    otherwise group DRO over split.group with exponentiated-gradient weights q."""
    torch.manual_seed(seed)
    X = torch.as_tensor(split.X.to_numpy(np.float32))
    y = torch.as_tensor(split.y.astype(np.float32))
    groups = torch.as_tensor(split.group)
    gids = torch.unique(groups)
    q = torch.ones(len(gids)) / len(gids)

    linear = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(linear.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(steps):
        logits = linear(X).squeeze(1)
        losses = bce(logits, y)
        if dro_eta is None:
            base = losses.mean()
        else:
            group_loss = torch.stack([losses[groups == g].mean() for g in gids])
            q = q * torch.exp(dro_eta * group_loss.detach())
            q = q / q.sum()
            base = (q * group_loss).sum()
        loss = base + lam * fpr_penalty(torch.sigmoid(logits), y, split.race)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model = TorchLogReg(linear)
    model.q = dict(zip(gids.tolist(), q.tolist()))
    return model


def candidates(splits, grid: list[dict], seed: int = SEED) -> list[tuple[dict, object]]:
    """[(config, trained model)] for every config in grid."""
    return [(cfg, train_penalized_lr(splits["train"], seed=seed, **cfg)) for cfg in grid]


def choose(cands, val) -> tuple[object, object, float]:
    """(config, model, threshold) picked on validation at the current eval.FPR_CAP."""
    scored = []
    for cfg, model in cands:
        t = fit_threshold(model, val)
        scored.append((evaluate(model, val, t), (cfg, model, t)))
    return select(scored)[1]


def sweep(splits, method: str, grid: list[dict]) -> dict:
    """Train every config in grid, record all candidates (selected=False),
    then re-record the one chosen on validation (selected=True)."""
    cands = candidates(splits, grid)
    for cfg, model in cands:
        evaluate_all(model, splits, method, cfg, selected=False)
    cfg, model, t = choose(cands, splits["val"])
    return {"config": cfg, "model": model, "threshold": t, "metrics": evaluate_all(model, splits, method, cfg, selected=True)}


def fairlearn_candidates(splits, seed: int = SEED) -> list[tuple[float, object]]:
    """ExpGrad decides at 0.5 and takes no sample weights, so its operating point
    is moved by oversampling positives; the factor is chosen on validation."""
    try:
        from fairlearn.reductions import ExponentiatedGradient, FalsePositiveRateParity
    except ImportError:
        print("fairlearn not installed; skipping reference")
        return []
    tr = splits["train"]
    pos = np.flatnonzero(tr.y == 1)

    class Wrapper:  # share of the randomized mixture voting "high risk"
        def __init__(self, mitigator):
            self.mitigator = mitigator

        def predict_proba(self, X):
            return np.asarray(self.mitigator._pmf_predict(X))[:, 1]

    cands = []
    for factor in OVERSAMPLE:
        rng = np.random.default_rng(seed)
        idx = np.concatenate([np.arange(len(tr.y)),
                              rng.choice(pos, int(round((factor - 1) * len(pos))), replace=False)])
        mit = ExponentiatedGradient(LogisticRegression(max_iter=2000), FalsePositiveRateParity(), eps=0.02)
        mit.fit(tr.X.iloc[idx], tr.y[idx], sensitive_features=tr.race[idx])
        cands.append((factor, Wrapper(mit)))
    return cands


def fairlearn_reference(splits, seed: int = SEED) -> dict | None:
    cands = fairlearn_candidates(splits, seed)
    if not cands:
        return None
    factor, model, t = choose(cands, splits["val"])
    params = {"eps": 0.02, "positive_oversample": factor}
    return {"model": model, "threshold": t, "config": params,
            "metrics": evaluate_all(model, splits, "fairlearn_expgrad_fpr", params, threshold=t)}


def run(splits=None) -> dict:
    splits = splits or load_splits()
    out = sweep(splits, "fair_lr", [{"lam": lam} for lam in LAMBDAS])
    out["fairlearn"] = fairlearn_reference(splits)
    return out


if __name__ == "__main__":
    res = run()
    t = res["metrics"]["test"]
    print(f"selected {res['config']}: test acc={t['accuracy']:.3f} race FPR gap={t['race_fpr_gap']:.3f} "
          f"FNR gap={t['race_fnr_gap']:.3f}")
    if res["fairlearn"]:
        f = res["fairlearn"]["metrics"]["test"]
        print(f"fairlearn reference: test acc={f['accuracy']:.3f} race FPR gap={f['race_fpr_gap']:.3f} "
              f"FNR gap={f['race_fnr_gap']:.3f}")
