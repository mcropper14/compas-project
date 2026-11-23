"""Shared data pipeline for the COMPAS fairness project.

Every method imports `load_splits()` from here. Do not write your own
preprocessing: identical rows, features, scaling and splits are what make the
results comparable across methods.

Rows: ProPublica's filters, then screenings before 2014-04-01 only (later
rows are label-truncated, see TRUNCATION_CUTOFF).

Splits (fixed, seeded):
  * shift : the latest 20% of defendants by compas_screening_date (temporal
            dataset shift; never used for training or model selection)
  * train / val / test : a stratified random 60/20/20 split of the remaining
            earlier 80%, stratified on race group x label.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "compas-scores-two-years.csv"
RAW_URL = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
           "master/compas-scores-two-years.csv")

SEED = 42
# From 2014-04 on, the two-year file keeps (almost) only defendants who
# recidivated: non-recidivists without a full two-year follow-up were dropped
# (see Barenstein 2019, "ProPublica's COMPAS Data Revisited"). Base rate jumps
# from ~0.37 to ~1.0, so we drop those screenings before splitting.
TRUNCATION_CUTOFF = pd.Timestamp("2014-04-01")
SHIFT_FRACTION = 0.20          # latest 20% by screening date
VAL_FRACTION = 0.20            # of the non-shift data
TEST_FRACTION = 0.20           # of the non-shift data

# The two groups ProPublica compared; race gaps are reported between these.
RACE_A, RACE_B = "African-American", "Caucasian"
SEX_A, SEX_B = "Male", "Female"

NUMERIC = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
BINARY = ["charge_felony"]                    # c_charge_degree == "F"
PROTECTED = ["race_african_american", "race_other", "sex_male"]


@dataclass
class Split:
    name: str
    X: pd.DataFrame           # model features (scaled), protected attrs excluded by default
    y: np.ndarray             # two_year_recid (0/1)
    race: np.ndarray          # "African-American" / "Caucasian" / "Other"
    sex: np.ndarray           # "Male" / "Female"
    group: np.ndarray         # int id of race x sex (6 groups), used by group DRO
    decile: np.ndarray        # COMPAS decile_score (1-10), for comparison only
    meta: pd.DataFrame        # id, screening date, etc. (never a model input)

    def __len__(self) -> int:
        return len(self.y)


GROUP_NAMES = [f"{r}|{s}" for r in (RACE_A, RACE_B, "Other") for s in (SEX_A, SEX_B)]


def download(path: Path = RAW_CSV) -> Path:
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, path)
    return path


def load_clean(path: Path = RAW_CSV, drop_truncated: bool = True) -> pd.DataFrame:
    """Load compas-scores-two-years.csv with ProPublica's standard filters.

    drop_truncated removes screenings on/after TRUNCATION_CUTOFF (see above).
    """
    df = pd.read_csv(download(path))
    df = df[
        df["days_b_screening_arrest"].between(-30, 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()
    df["compas_screening_date"] = pd.to_datetime(df["compas_screening_date"])
    if drop_truncated:
        df = df[df["compas_screening_date"] < TRUNCATION_CUTOFF].copy()
    df["race_group"] = np.where(df["race"].isin([RACE_A, RACE_B]), df["race"], "Other")
    df["charge_felony"] = (df["c_charge_degree"] == "F").astype(float)
    df["race_african_american"] = (df["race_group"] == RACE_A).astype(float)
    df["race_other"] = (df["race_group"] == "Other").astype(float)
    df["sex_male"] = (df["sex"] == SEX_A).astype(float)
    df["group_id"] = [GROUP_NAMES.index(f"{r}|{s}") for r, s in zip(df["race_group"], df["sex"])]
    return df.reset_index(drop=True)


def assign_splits(df: pd.DataFrame, seed: int = SEED, scheme: str = "temporal") -> pd.Series:
    """Return a Series of split names ("train"/"val"/"test"/"shift") aligned with df."""
    split = pd.Series("", index=df.index, dtype=object)
    order = df.sort_values(["compas_screening_date", "id"]).index
    n_shift = int(round(SHIFT_FRACTION * len(df)))
    split[order[-n_shift:]] = "shift"

    if scheme == "charge":
        # second shift set: same period, but misdemeanor charges held out entirely;
        # later-period rows are not used
        split[split == "shift"] = "unused"
        split[(split == "") & (df["charge_felony"] == 0)] = "shift"
    elif scheme != "temporal":
        raise ValueError(scheme)
    rest = df.index[split == ""]
    strat = df.loc[rest, "race_group"] + "_" + df.loc[rest, "two_year_recid"].astype(str)
    trval, test = train_test_split(rest, test_size=TEST_FRACTION, stratify=strat, random_state=seed)
    val_share = VAL_FRACTION / (1 - TEST_FRACTION)
    train, val = train_test_split(trval, test_size=val_share, stratify=strat[trval], random_state=seed)
    split[train], split[val], split[test] = "train", "val", "test"
    return split


def load_splits(include_protected: bool = False, seed: int = SEED,
                scheme: str = "temporal") -> dict[str, Split]:
    """Main entry point. Returns {"train", "val", "test", "shift"} -> Split.

    include_protected=True adds race and sex indicator columns to X (used only
    for the "with race" baseline). Numeric features are standardized with
    statistics from the training split. scheme="charge" trains on felony charges
    and uses misdemeanors as the shift split (the charge feature is then constant
    in training, so it is dropped).
    """
    df = load_clean()
    df["split"] = assign_splits(df, seed, scheme)
    cols = NUMERIC + (BINARY if scheme == "temporal" else []) + (PROTECTED if include_protected else [])

    train = df[df["split"] == "train"]
    mu, sd = train[NUMERIC].mean(), train[NUMERIC].std().replace(0, 1)

    out = {}
    for name in ("train", "val", "test", "shift"):
        part = df[df["split"] == name].reset_index(drop=True)
        X = part[cols].astype(float).copy()
        X[NUMERIC] = (X[NUMERIC] - mu) / sd
        out[name] = Split(
            name=name,
            X=X,
            y=part["two_year_recid"].to_numpy(int),
            race=part["race_group"].to_numpy(),
            sex=part["sex"].to_numpy(),
            group=part["group_id"].to_numpy(int),
            decile=part["decile_score"].to_numpy(int),
            meta=part[["id", "compas_screening_date", "race", "sex", "age", "priors_count"]],
        )
    return out


@lru_cache(maxsize=None)
def train_base_rate(seed: int = SEED) -> float:
    """P(two_year_recid=1) on the training split; used as the decision threshold."""
    df = load_clean()
    return float(df.loc[assign_splits(df, seed) == "train", "two_year_recid"].mean())


def describe(splits: dict[str, Split]) -> pd.DataFrame:
    rows = []
    for name, s in splits.items():
        dates = s.meta["compas_screening_date"]
        rows.append({
            "split": name, "n": len(s), "base_rate": s.y.mean(),
            f"share_{RACE_A}": (s.race == RACE_A).mean(),
            f"base_rate_{RACE_A}": s.y[s.race == RACE_A].mean(),
            f"base_rate_{RACE_B}": s.y[s.race == RACE_B].mean(),
            "date_min": dates.min().date(), "date_max": dates.max().date(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(describe(load_splits()).to_string(index=False))
