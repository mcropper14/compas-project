"""Multi-seed comparison with confidence intervals. The baseline is plain
logistic regression without race (lr_no_race).

    python robustness.py              # results/robustness.md and robustness.csv
    python robustness.py --fit 0 2 4  # only fit and save these seeds (for parallel runs)

Ten seeds. Each seed draws a new train/val/test split, seeds torch and the
ensembles, retrains every method and re-chooses thresholds and hyperparameters
on that seed's validation set. Two shift schemes (data.assign_splits):
  temporal  shift = latest 20% by screening date
  charge    train/val/test are felony charges only, shift = misdemeanors
The shift people are the same in every seed.

Estimate = mean over seeds. 95% CI = bootstrap of that mean: every draw
resamples people AND seeds. A person gets the same resampling weight in every
seed, so people who appear in several test sets (and the fixed shift set) are
not counted as new evidence. Differences from the baseline use the same draw
for both methods (paired). Intervals cover test-sample and seed/split
variation; they don't cover collecting a new training population.

Each seed's scores are kept in results/robustness_runs/<code hash>/, so
interrupted runs resume and thresholds, caps, tolerances and reports can be
rebuilt without retraining. Changing the data, methods or fitting code changes
the hash and forces new runs.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import eval as ev
import methods  # noqa: F401
from data import RACE_A, RACE_B, SEX_A, SEX_B, load_splits
from methods import dro, ensembles, fair_lr
from methods.baseline import train_lr
from methods.reweighting import weights_for

SEEDS = range(10)
SCHEMES = ["temporal", "charge"]
CAPS = [0.15, 0.20, 0.25, 0.30, 0.35]
TOLS = [0.03, 0.05, 0.08]
N_BOOT = 2000
BASELINE = "lr_no_race"
OUT = ev.RESULTS_CSV.parent
MAIN = ["lr_no_race", "compas_decile", "compas_capped", "lr_with_race", "reweighting_race", "ensemble_soft",
        "group_thresholds", "fair_lr", "group_dro", "dro_fair", "fairlearn_expgrad_fpr"]
GAPS = [f"{a}_{k}_diff" for a in ("race", "sex") for k in ("fpr", "fnr", "ppv")]
PER_RACE = [f"{k}_{g}" for k in ("fpr", "fnr", "ppv") for g in ("black", "white")]
METRICS = ["accuracy", "auc", "fpr", "abs_race_fpr_gap", *GAPS, *PER_RACE]


# ---------------------------------------------------------------- fitting
def fit_scores(seed: int, scheme: str) -> dict[str, np.ndarray]:
    """Scores of every candidate model on val/test/shift: {'split|method|config': scores}."""
    splits = load_splits(seed=seed, scheme=scheme)
    with_race = load_splits(include_protected=True, seed=seed, scheme=scheme)
    tr = splits["train"]
    out = {}

    def add(method, model, cfg="", sp=splits):
        for n in ("val", "test", "shift"):
            out[f"{n}|{method}|{cfg}"] = np.asarray(ev.get_scores(model, sp[n]), dtype=np.float64)

    add("lr_no_race", train_lr(tr))
    add("lr_with_race", train_lr(with_race["train"]), sp=with_race)
    add("reweighting_race", train_lr(tr, sample_weight=weights_for(tr, "race")))
    for tag, sw in (("", None), ("rw_", weights_for(tr, "race"))):
        fitted = {n: m.fit(tr.X, tr.y, sample_weight=sw) for n, m in ensembles.base_models(seed).items()}
        val_acc = {n: ev.evaluate(m, splits["val"], ev.THRESHOLD)["accuracy"] for n, m in fitted.items()}
        for n, m in fitted.items():
            add(f"single_{tag}{n}", m)
        for mode in ("majority", "soft", "weighted"):
            add(f"ensemble_{tag}{mode}", ensembles.VotingEnsemble(fitted, mode, val_acc if mode == "weighted" else None))
    grids = {"fair_lr": [{"lam": x} for x in fair_lr.LAMBDAS],
             "group_dro": [{"lam": 0.0, "dro_eta": e} for e in dro.ETAS],
             "dro_fair": [{"lam": x, "dro_eta": e} for x in fair_lr.LAMBDAS if x > 0 for e in dro.ETAS]}
    for method, grid in grids.items():
        for cfg, model in fair_lr.candidates(splits, grid, seed):
            add(method, model, f"{cfg['lam']},{cfg.get('dro_eta', '')}")
    for factor, model in fair_lr.fairlearn_candidates(splits, seed):
        add("fairlearn_expgrad_fpr", model, str(factor))
    return out


def code_hash() -> str:
    root = Path(__file__).resolve().parent
    files = [root / "data.py", root / "eval.py", *sorted((root / "methods").glob("*.py")),
             root / "data" / "compas-scores-two-years.csv"]
    blob = b"".join(f.read_bytes() for f in files) + inspect.getsource(fit_scores).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def get_scores(seed: int, scheme: str) -> dict[str, np.ndarray]:
    folder = OUT / "robustness_runs" / code_hash()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{scheme}_seed{seed}.npz"
    if not path.exists():
        tmp = folder / f"tmp_{path.name}"
        np.savez_compressed(tmp, **fit_scores(seed, scheme))
        tmp.replace(path)
        print(scheme, "seed", seed, "done", flush=True)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- decisions
def decide(scores: dict, splits: dict) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """{split: (0/1 predictions, scores)}, columns 'method@cap@tol' for chosen
    models and 'cand|method|lam|eta' for lambda candidates at the default cap."""
    val, names = splits["val"], ("test", "shift")
    P, Z = {n: {} for n in names}, {n: {} for n in names}
    models: dict[str, list[str]] = {}
    for key in scores:
        n, method, cfg = key.split("|")
        if n == "val":
            models.setdefault(method, []).append(cfg)

    def add(col, s, t):
        for n in names:
            P[n][col], Z[n][col] = ev.predict(s[n], splits[n], t), s[n]

    get = lambda m, c: {n: scores[f"{n}|{m}|{c}"] for n in ("val", *names)}  # noqa: E731
    compas = {n: splits[n].decile / 10.0 for n in ("val", *names)}
    settings = [(c, ev.GAP_TOLERANCE) for c in CAPS] + [(ev.FPR_CAP, t) for t in TOLS if t != ev.GAP_TOLERANCE]
    add(f"compas_decile@{ev.FPR_CAP}@{ev.GAP_TOLERANCE}", compas, 0.5)
    for cap, tol in settings:
        tag = f"@{cap}@{tol}"
        for method, cfgs in models.items():
            scored = []
            for c in cfgs:
                s = get(method, c)
                t = ev.capped_threshold(s["val"], val.y, cap)
                scored.append((ev.evaluate_scores(s["val"], val, t) if len(cfgs) > 1 else None, (s, t)))
            s, t = ev.select(scored, tol)[1] if len(cfgs) > 1 else scored[0][1]
            add(method + tag, s, t)
        add("compas_capped" + tag, compas, ev.capped_threshold(compas["val"], val.y, cap))
        s = get("lr_no_race", "")
        add("group_thresholds" + tag, s, {str(g): ev.capped_threshold(s["val"][val.race == g], val.y[val.race == g], cap)
                                          for g in np.unique(val.race)})
    for k in ("fair_lr", "dro_fair"):
        for c in models[k]:
            s = get(k, c)
            add(f"cand|{k}|" + c.replace(",", "|"), s, ev.capped_threshold(s["val"], val.y))
    base = {n: pd.DataFrame({"id": splits[n].meta["id"], "race": splits[n].race, "sex": splits[n].sex, "y": splits[n].y}) for n in names}
    return {n: (pd.concat([base[n], pd.DataFrame(P[n])], axis=1), pd.DataFrame(Z[n])) for n in names}


# ---------------------------------------------------------------- statistics
def weighted_auc(z: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """AUC per weight draw, ties counted half."""
    o = np.argsort(z, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(z[o]) != 0])
    pos = np.add.reduceat((W[:, o] * y[o]).astype(float), starts, axis=1)
    neg = np.add.reduceat((W[:, o] * (1 - y[o])).astype(float), starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + neg / 2)).sum(1) / (pos.sum(1) * neg.sum(1))


def weighted_metrics(df: pd.DataFrame, scores: pd.DataFrame, cols: list[str], W: np.ndarray) -> np.ndarray:
    """(draws, methods, metrics) for one seed; W = (draws, people) weights."""
    P, y = df[cols].to_numpy(np.float32), df["y"].to_numpy()
    race, sex = df["race"].to_numpy(), df["sex"].to_numpy()

    def rates(mask):
        m0, m1 = mask & (y == 0), mask & (y == 1)
        n0, n1 = W[:, m0].sum(1)[:, None], W[:, m1].sum(1)[:, None]
        fp, tp = W[:, m0] @ P[m0], W[:, m1] @ P[m1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"acc": (n0 - fp + tp) / (n0 + n1), "fpr": fp / n0, "fnr": 1 - tp / n1, "ppv": tp / (tp + fp)}

    al = rates(np.ones(len(y), bool))
    b, w = rates(race == RACE_A), rates(race == RACE_B)
    m, f = rates(sex == SEX_A), rates(sex == SEX_B)
    auc = np.full((len(W), len(cols)), np.nan)
    default = f"@{ev.FPR_CAP}@{ev.GAP_TOLERANCE}"
    for j, c in enumerate(cols):
        if c.endswith(default):              # AUC doesn't depend on the cap
            auc[:, j] = weighted_auc(scores[c].to_numpy(), y, W)
    v = {"accuracy": al["acc"], "auc": auc, "fpr": al["fpr"], "abs_race_fpr_gap": np.abs(b["fpr"] - w["fpr"])}
    for k in ("fpr", "fnr", "ppv"):
        v[f"race_{k}_diff"], v[f"sex_{k}_diff"] = b[k] - w[k], m[k] - f[k]
        v[f"{k}_black"], v[f"{k}_white"] = b[k], w[k]
    return np.stack([v[k] for k in METRICS], axis=2)


def analyze(runs: list[tuple[pd.DataFrame, pd.DataFrame]], n_boot: int = N_BOOT) -> pd.DataFrame:
    """Tidy table: one row per (column, metric)."""
    cols = [c for c in runs[0][0].columns if c not in ("id", "race", "sex", "y")]
    people = pd.Index(sorted(set().union(*(d["id"] for d, _ in runs))))
    rng = np.random.default_rng(0)
    W = rng.multinomial(len(people), np.full(len(people), 1 / len(people)), size=n_boot).astype(np.float32)
    SW = rng.multinomial(len(runs), np.full(len(runs), 1 / len(runs)), size=n_boot).astype(float)
    point, total = [], np.zeros((n_boot, len(cols), len(METRICS)))
    for s, (df, z) in enumerate(runs):
        assert df["id"].is_unique
        point.append(weighted_metrics(df, z, cols, np.ones((1, len(df)), np.float32))[0])
        total += SW[:, s, None, None] * weighted_metrics(df, z, cols, W[:, people.get_indexer(df["id"])])
    point, boot = np.array(point), total / len(runs)

    rows = []
    for j, c in enumerate(cols):
        cap = c.split("@")[1] if "@" in c else str(ev.FPR_CAP)
        b = cols.index(f"{BASELINE}@{cap}@{ev.GAP_TOLERANCE}")
        for k, m in enumerate(METRICS):
            if np.isnan(point[:, j, k]).all():
                continue
            lo, hi = np.nanpercentile(boot[:, j, k], [2.5, 97.5])
            d = boot[:, j, k] - boot[:, b, k]
            dlo, dhi = np.nanpercentile(d, [2.5, 97.5]) if not np.isnan(d).all() else (np.nan, np.nan)
            rows.append({"column": c, "metric": m, "mean": np.nanmean(point[:, j, k]),
                         "seed_sd": np.nanstd(point[:, j, k], ddof=1), "lo": lo, "hi": hi,
                         "vs_baseline": np.mean(point[:, j, k] - point[:, b, k]), "vs_lo": dlo, "vs_hi": dhi,
                         "undefined_draws": int(np.isnan(boot[:, j, k]).sum())})
    return pd.DataFrame(rows)


def shift_change(test: list, shift: list, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Shift minus test for each method, and how that change differs from the
    baseline's. Test and shift people are disjoint, so they are resampled
    independently; the seed resampling is shared."""
    suffix = f"@{ev.FPR_CAP}@{ev.GAP_TOLERANCE}"
    cols = [f"{m}{suffix}" for m in MAIN]
    SW = np.random.default_rng(1).multinomial(len(SEEDS), np.full(len(SEEDS), 1 / len(SEEDS)), size=n_boot)
    point, boot = [], []
    for stream, runs in ((2, test), (3, shift)):
        people = pd.Index(sorted(set().union(*(d["id"] for d, _ in runs))))
        W = np.random.default_rng(stream).multinomial(
            len(people), np.full(len(people), 1 / len(people)), size=n_boot).astype(np.float32)
        point.append(np.mean([weighted_metrics(d, z, cols, np.ones((1, len(d)), np.float32))[0] for d, z in runs], axis=0))
        boot.append(sum(SW[:, s, None, None] * weighted_metrics(d, z, cols, W[:, people.get_indexer(d["id"])])
                        for s, (d, z) in enumerate(runs)) / len(runs))
    p, b, base = point[1] - point[0], boot[1] - boot[0], MAIN.index(BASELINE)
    rows = []
    for j, m in enumerate(MAIN):
        for k, metric in enumerate(METRICS):
            dd = b[:, j, k] - b[:, base, k]
            lo, hi = np.nanpercentile(b[:, j, k], [2.5, 97.5])
            dlo, dhi = np.nanpercentile(dd, [2.5, 97.5])
            rows.append({"method": m, "metric": metric, "mean": p[j, k], "lo": lo, "hi": hi,
                         "vs_baseline": p[j, k] - p[base, k], "vs_lo": dlo, "vs_hi": dhi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report
def cells(t: pd.DataFrame, diff: bool = False) -> pd.Series:
    v, lo, hi = ("vs_baseline", "vs_lo", "vs_hi") if diff else ("mean", "lo", "hi")
    sign = "+" if diff else ""
    return pd.Series([f"{a:{sign}.3f} [{b:{sign}.3f}, {c:{sign}.3f}]" for a, b, c in zip(t[v], t[lo], t[hi])], index=t.index)


def table(t: pd.DataFrame, metrics, methods_, diff: bool = False, columns: str = "metric") -> str:
    t = t[t["metric"].isin(metrics) & t["method"].isin(methods_)]
    p = t.assign(cell=cells(t, diff)).pivot(index="method", columns=columns, values="cell")
    p = p.reindex([m for m in methods_ if m in p.index])
    return (p[[m for m in metrics if m in p]] if columns == "metric" else p).to_markdown()


def main() -> None:
    torch.set_num_threads(4)
    parts, changes = [], []
    for scheme in SCHEMES:
        decided = [decide(get_scores(s, scheme), load_splits(seed=s, scheme=scheme)) for s in SEEDS]
        for n in ("test", "shift"):
            parts.append(analyze([d[n] for d in decided]).assign(scheme=scheme, split=n))
        changes.append(shift_change([d["test"] for d in decided], [d["shift"] for d in decided]).assign(scheme=scheme))
    tab = pd.concat(parts, ignore_index=True)
    change = pd.concat(changes, ignore_index=True)
    change.to_csv(OUT / "shift_change.csv", index=False)
    is_cand = tab["column"].str.startswith("cand|")
    tab["kind"] = np.where(is_cand, "candidate", "chosen")
    tab["method"] = [c.split("|")[1] if k else c.split("@")[0] for c, k in zip(tab["column"], is_cand)]
    tab["cap"] = [ev.FPR_CAP if k else float(c.split("@")[1]) for c, k in zip(tab["column"], is_cand)]
    tab["tol"] = [ev.GAP_TOLERANCE if k else float(c.split("@")[2]) for c, k in zip(tab["column"], is_cand)]
    tab["lam"] = [c.split("|")[2] if k else "" for c, k in zip(tab["column"], is_cand)]
    tab["eta"] = [c.split("|")[3] if k else "" for c, k in zip(tab["column"], is_cand)]
    tab.to_csv(OUT / "robustness.csv", index=False)

    ch = tab[tab["kind"] == "chosen"]
    d = ch[(ch["cap"] == ev.FPR_CAP) & (ch["tol"] == ev.GAP_TOLERANCE)]
    extra = [m for m in d["method"].drop_duplicates() if m.startswith(("single_", "ensemble_"))]
    tuned = ["fair_lr", "dro_fair", "fairlearn_expgrad_fpr", "group_dro"]
    out = ["# Multi-seed results", "",
           f"Baseline: `{BASELINE}`. {len(SEEDS)} seeds; cells are mean [95% CI] from a bootstrap over people and "
           "seeds (see robustness.py). Race gaps are African-American minus Caucasian, sex gaps male minus female. "
           "abs_race_fpr_gap is the mean absolute gap, which sampling noise pushes upward, so prefer the signed gap "
           "near zero. Intervals are not adjusted for multiple comparisons.", ""]
    titles = {("temporal", "test"): "Temporal scheme, test", ("temporal", "shift"): "Temporal scheme, shift (later period)",
              ("charge", "test"): "Charge scheme, test (felonies)", ("charge", "shift"): "Charge scheme, shift (misdemeanors)"}
    for (scheme, n), title in titles.items():
        t = d[(d["scheme"] == scheme) & (d["split"] == n)]
        head = ["accuracy", "auc", "fpr", "race_fpr_diff", "race_fnr_diff", "race_ppv_diff", "abs_race_fpr_gap"]
        rest = [m for m in MAIN if m != BASELINE]
        out += [f"# {title}", "", "## Estimates", "", table(t, head, MAIN), "",
                f"## Difference from {BASELINE} (paired)", "", table(t, head, rest, True), "",
                "## Who pays: rates by race", "", table(t, PER_RACE, MAIN), "",
                f"## Who pays: change from {BASELINE} (paired)", "", table(t, PER_RACE, rest, True), "",
                "## Sex gaps (male minus female)", "", table(t, GAPS[3:], MAIN), "",
                f"## Sex gaps: change from {BASELINE} (paired)", "", table(t, GAPS[3:], rest, True), ""]
    for scheme, label in (("temporal", "later period"), ("charge", "misdemeanors")):
        t = change[change["scheme"] == scheme]
        ms = ["accuracy", "fpr", "race_fpr_diff", "race_fnr_diff", "abs_race_fpr_gap"]
        out += [f"# Change from test to shift ({label}): shift minus test", "",
                "## Change", "", table(t, ms, MAIN), "",
                f"## Change minus {BASELINE}'s change (paired)", "",
                table(t, ms, [m for m in MAIN if m != BASELINE], True), ""]
    out += ["# Classifier and voting ablations (temporal scheme)", ""]
    for n in ("test", "shift"):
        t = d[(d["scheme"] == "temporal") & (d["split"] == n)]
        ms = [BASELINE] + extra
        out += [f"## {n}: estimates", "", table(t, ["accuracy", "auc", "race_fpr_diff"], ms), "",
                f"## {n}: difference from {BASELINE} (paired)", "", table(t, ["accuracy", "auc", "race_fpr_diff"], extra, True), ""]
    for scheme, n in titles:
        t = ch[(ch["scheme"] == scheme) & (ch["split"] == n) & (ch["cap"] == ev.FPR_CAP)]
        out += [f"# Selection tolerance: {titles[scheme, n]}", ""]
        for m in ("accuracy", "race_fpr_diff"):
            out += [f"## {m}", "", table(t, [m], tuned, columns="tol"), ""]
    for scheme, n in titles:
        t = ch[(ch["scheme"] == scheme) & (ch["split"] == n) & (ch["tol"] == ev.GAP_TOLERANCE) & (ch["method"] != "compas_decile")]
        out += [f"# Race FPR gap by FPR cap: {titles[scheme, n]}", "",
                table(t, ["race_fpr_diff"], MAIN, columns="cap"), ""]
    (OUT / "robustness.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=int, nargs="+")
    args = ap.parse_args()
    if args.fit is None:
        main()
    else:
        torch.set_num_threads(4)
        for sch in SCHEMES:
            for sd in args.fit:
                get_scores(sd, sch)
