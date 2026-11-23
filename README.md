# Investigating Fairness Optimization in Bias Risk Assessment


We train COMPAS-like recidivism models on ProPublica's Broward County data and
ask what it costs to satisfy one fairness criterion, equal false positive rates
by race under an FPR cap, and who bears that cost. We compare reweighting, an
FPR-gap penalty, group DRO with that penalty, and per-race thresholds. Every
method uses the same data, splits and metrics, and we test whether the result
holds under two shifts (a later period, and misdemeanors held out of training).

## Quick start

```bash
python -m venv --system-site-packages .venv   # or a plain venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py      
.venv/bin/python robustness.py  
.venv/bin/python figures.py      
```

Each method also runs on its own, for example `.venv/bin/python -m methods.dro`.
Note that run_all.py clears results.csv first.

## Layout

| File | Owner | What it does |
|---|---|---|
| `data.py` | The only preprocessing. Applies ProPublica's filters, fixes the splits and builds the features. `load_splits()` |
| `eval.py` | `evaluate(model, split)`, the λ-selection rule `select()` and the shared results table `record()` |
| `methods/baseline.py` | LR with and without race, the COMPAS decile comparison and the proxy check |
| `methods/reweighting.py` | Kamiran & Calders reweighing, `w = P(g)P(y)/P(g,y)` |
| `methods/ensembles.py` | 5 classifier families, majority/soft/weighted voting, each with and without reweighting |
| `methods/fair_lr.py` | PyTorch LR with BCE + λ·\|ΔFPR\|, the λ sweep and the Fairlearn ExpGrad reference |
| `methods/dro.py` | Group DRO over race × sex groups + λ penalty, with η and λ chosen on validation |
| `methods/group_thresholds.py` | | Plain LR with one threshold per race (uses race at decision time) |
| `robustness.py` | | Ten seeds, 95% CIs, paired differences from the baseline; rates by race, sex gaps, ablations, cap and tolerance sensitivity |
| `run_all.py`, `figures.py` | Run everything, write the summary tables and draw the figures |

All methods record into `results/results.csv`, one row per (method, split, config).
`selected=True` marks the configuration chosen on validation. The figures use the
multi-seed `results/robustness.csv`; test-to-shift changes are in `results/shift_change.csv`.

## Decisions (apply to every method)

- **Data.** We use `compas-scores-two-years.csv` with ProPublica's filters
  (`days_b_screening_arrest` within ±30, `is_recid != -1`, charge degree not "O",
  valid score). That leaves 6,172 rows.
- **Label truncation.** From April 2014 onward, the file keeps almost only
  defendants who reoffended. The base rate jumps from about 0.37 to 1.0, which
  matches Barenstein (2019). We drop those screenings, leaving 5,297 rows. A
  shift split built on the raw data would have been ~100% positive, which makes
  FPR meaningless there. This belongs in the data-bias section of the report.
- **Splits (seed 42).**
  - The **shift** split is the latest 20% of the data by screening date
    (2013-12-23 to 2014-03-31, n = 1,059).
  - The remaining earlier data is split 60/20/20 into
    **train/val/test** (2,542/848/848), stratified by race group × label.
  - **Second shift (`scheme="charge"`).** Felony charges only, split 60/20/20
    (1,626/542/542); all 1,528 misdemeanors from the same period are the shift set
    (base rate 0.30, against 0.40 for felonies). The charge feature is dropped.
    Resampling by race or label wouldn't test the FPR gap: FPR is already
    conditional on both.
- **Features.** Age, priors count, the three juvenile counts and whether the
  charge was a felony, standardized on train. Race and sex are excluded except
  in the `lr_with_race` baseline.
- **Groups.** Race gaps compare African-American and Caucasian defendants, as
  ProPublica did. Sex gaps compare male and female defendants. Group DRO uses
  6 race × sex groups, with "Other" race as its own group.
- **Fairness criterion: equal FPR by race, under a cap.** If a high-risk label can
  contribute to detention, a false positive costs liberty to someone who was not
  rearrested. This is a value judgment. FNR and PPV gaps are reported as costs.
- **Decision threshold.** Each model uses the lowest threshold with overall FPR
  ≤ 25% on validation, so none can look fair by flagging no one. Realized FPR on
  test and shift can differ. COMPAS is shown with its own decile ≥ 5 rule and
  with the cap.
- **Choosing λ (and η) on validation only:**
  1. Drop candidates that don't beat majority-class accuracy, if possible.
  2. Among the rest, take the highest accuracy with race FPR gap ≤ 0.05.
  3. If none qualifies, take the smallest race FPR gap.
- **Fairlearn reference.** ExpGrad with false positive rate parity; the positive
  oversampling factor is chosen on validation.

## Results

The **baseline is plain logistic regression without race**. Ten seeds; cells are
mean [95% CI]; gaps are Black minus white. Each seed redraws train/val/test, seeds
torch and the ensembles, retrains, and re-chooses thresholds and hyperparameters.
CIs bootstrap people and seeds together, with paired differences from the baseline.
They don't cover a new training population and aren't adjusted for multiple
comparisons. The shift people are the same in every seed. All metrics and caps are
in `results/robustness.md`. `results/summary.md` and `results.csv` are single-seed.

| Method | Test accuracy | Test FPR gap | Shift accuracy | Shift FPR gap |
|---|---|---|---|---|
| **LR, no race (baseline)** | 0.688 [0.666, 0.708] | 0.166 [0.118, 0.216] | 0.705 [0.678, 0.731] | 0.104 [0.032, 0.173] |
| COMPAS, decile ≥ 5 | 0.666 [0.645, 0.687] | 0.195 [0.138, 0.248] | 0.679 [0.652, 0.708] | 0.159 [0.089, 0.228] |
| COMPAS, FPR cap | 0.690 [0.670, 0.710] | 0.177 [0.127, 0.222] | 0.699 [0.672, 0.727] | 0.105 [0.047, 0.163] |
| LR with race | 0.689 [0.669, 0.709] | 0.260 [0.211, 0.312] | 0.698 [0.671, 0.724] | 0.177 [0.109, 0.246] |
| Reweighting | 0.688 [0.666, 0.708] | 0.163 [0.115, 0.213] | 0.705 [0.679, 0.730] | 0.102 [0.029, 0.171] |
| Soft-vote ensemble | 0.690 [0.669, 0.709] | 0.164 [0.116, 0.214] | 0.692 [0.666, 0.717] | 0.119 [0.048, 0.192] |
| Per-race thresholds | 0.677 [0.656, 0.698] | -0.032 [-0.088, 0.023] | 0.707 [0.681, 0.733] | -0.097 [-0.167, -0.024] |
| Fair LR (λ penalty) | 0.648 [0.627, 0.668] | 0.048 [-0.009, 0.106] | 0.650 [0.623, 0.679] | 0.088 [0.017, 0.153] |
| Group DRO | 0.687 [0.666, 0.707] | 0.172 [0.122, 0.220] | 0.703 [0.678, 0.729] | 0.106 [0.037, 0.174] |
| Group DRO + λ | 0.649 [0.626, 0.670] | 0.042 [-0.013, 0.097] | 0.650 [0.622, 0.679] | 0.091 [0.024, 0.155] |
| Fairlearn ExpGrad* | 0.674 [0.651, 0.696] | 0.055 [0.016, 0.097] | 0.672 [0.645, 0.700] | 0.035 [-0.009, 0.080] |

Change from the baseline (paired), methods that target FPR:

| Method | Test accuracy | Test FPR gap | Shift accuracy | Shift FPR gap |
|---|---|---|---|---|
| Per-race thresholds | -0.011 [-0.023, +0.002] | -0.198 [-0.244, -0.157] | +0.002 [-0.015, +0.019] | -0.201 [-0.252, -0.153] |
| Fair LR | -0.040 [-0.063, -0.019] | -0.118 [-0.180, -0.059] | -0.055 [-0.082, -0.027] | -0.016 [-0.086, +0.050] |
| Group DRO + λ | -0.039 [-0.061, -0.018] | -0.124 [-0.180, -0.067] | -0.055 [-0.082, -0.028] | -0.014 [-0.082, +0.051] |
| Fairlearn ExpGrad* | -0.013 [-0.034, +0.006] | -0.111 [-0.166, -0.058] | -0.033 [-0.060, -0.007] | -0.070 [-0.125, -0.019] |

*Not directly comparable: its overall FPR is about 11-12% in the temporal scheme, so it flags far fewer people.

These changes are in the signed gap: per-race thresholds' change on shift overshoots
zero. Their mean absolute gap there changes by -0.008 [-0.127, +0.126].

- **On test, per-race thresholds and both penalties reduce the FPR gap**, with CIs
  excluding zero. Thresholds show no clear accuracy loss; the penalties cost about
  0.04. Thresholds use race at decision time. They also raise the PPV gap by 0.097
  [0.054, 0.143]; the penalties' PPV change is not distinguishable from zero.
- **Under the temporal shift, neither keeps its benefit.** Thresholds' gap is
  reversed (-0.097 [-0.167, -0.024]). The penalties no longer detectably improve on
  the baseline and lose about 0.055 accuracy.
- **Under the charge shift, the penalties keep their benefit and thresholds keep
  about half** (tables below). Whether equal FPR survives depends on the shift.
- **No detectable FPR-gap change from reweighting, the soft-vote ensemble or group DRO alone.**
  DRO + λ is similar to fair LR.
- **Using race and sex as features raises the FPR gap** by 0.094 [0.061, 0.127] on test.
  Dropping it isn't enough: race-blind features still predict race (AUC 0.70, seed 42).
- **Cap sensitivity (temporal test).** Thresholds' gap stays near zero for caps of
  15-35%. The penalties' gaps are about 0.04-0.06 up to 30% and rise to 0.109-0.135
  at 35%.
- **Tolerance sensitivity.** Selecting at 0.03, 0.05 or 0.08 moves the penalties'
  test gap from about 0.04 to 0.06-0.07 and accuracy by under 0.01. Conclusions hold.

### Who pays (temporal test, change from baseline)

| Method | Black FPR | White FPR | Black FNR | White FNR |
|---|---|---|---|---|
| Per-race thresholds | -0.112 [-0.146, -0.079] | +0.087 [+0.062, +0.112] | +0.121 [+0.090, +0.152] | -0.090 [-0.128, -0.055] |
| Fair LR | -0.050 [-0.092, -0.009] | +0.069 [+0.026, +0.114] | +0.157 [+0.109, +0.207] | +0.017 [-0.052, +0.086] |
| Group DRO + λ | -0.051 [-0.095, -0.008] | +0.073 [+0.033, +0.114] | +0.156 [+0.107, +0.203] | +0.019 [-0.051, +0.086] |

Equal FPR lowers false positives for Black defendants and raises them for white
defendants. Thresholds trade the other way in FNR. The penalties are worse on this
accounting: they miss more Black defendants who were rearrested with no FNR gain
for white defendants, and AUC falls from 0.727 to about 0.63.

### Second shift: train on felonies, test on misdemeanors (change from baseline)

The baseline FPR gap is 0.144 [0.083, 0.205] on felony test and
0.123 [0.074, 0.169] on misdemeanors. Compare methods within this scheme only: the
training set is smaller than in the temporal scheme.

| Method | Felony FPR gap change | Misdemeanor FPR gap change | Misdemeanor FPR gap | Misdemeanor accuracy change |
|---|---|---|---|---|
| Per-race thresholds | -0.128 [-0.181, -0.079] | -0.068 [-0.095, -0.043] | 0.055 [0.004, 0.106] | -0.002 [-0.010, +0.006] |
| Fair LR | -0.093 [-0.170, -0.025] | -0.100 [-0.154, -0.052] | 0.023 [-0.035, 0.079] | -0.043 [-0.067, -0.022] |
| Group DRO | -0.004 [-0.033, +0.022] | -0.005 [-0.017, +0.008] | 0.118 [0.073, 0.164] | +0.001 [-0.004, +0.006] |
| Group DRO + λ | -0.082 [-0.160, -0.014] | -0.095 [-0.144, -0.047] | 0.028 [-0.029, 0.085] | -0.038 [-0.061, -0.018] |

### Change from test to shift

Shift minus test, with test and shift people resampled independently. "Relative to
baseline" is the method's change minus the baseline's change. A positive value means
the method's signed gap moved toward positive relative to the baseline's; for methods
whose gap is below the baseline's, that is a smaller improvement.

| Method | Temporal: gap change | Temporal: relative to baseline | Charge: gap change | Charge: relative to baseline |
|---|---|---|---|---|
| LR, no race (baseline) | -0.062 [-0.144, +0.024] | | -0.021 [-0.100, +0.055] | |
| Per-race thresholds | -0.064 [-0.146, +0.018] | -0.003 [-0.057, +0.053] | +0.039 [-0.046, +0.119] | +0.060 [+0.022, +0.105] |
| Fair LR | +0.040 [-0.044, +0.126] | +0.102 [+0.009, +0.190] | -0.028 [-0.114, +0.052] | -0.007 [-0.088, +0.067] |
| Group DRO | -0.065 [-0.145, +0.020] | -0.004 [-0.025, +0.018] | -0.021 [-0.103, +0.054] | -0.000 [-0.031, +0.030] |
| Group DRO + λ | +0.049 [-0.034, +0.135] | +0.110 [+0.021, +0.197] | -0.034 [-0.120, +0.043] | -0.013 [-0.094, +0.058] |

- **No listed method's own gap change is detectable,** so don't say a gap "significantly
  grew" or "shrank" from test to shift.
- **The improvement over the baseline does shrink detectably** for the penalties in
  the later period (by about 0.10-0.11) and for thresholds on misdemeanors (0.060).
- **Thresholds in the later period** move with the baseline: the same shift applied to
  a gap that started near zero ends negative.

### Secondary checks

- **Sex.** The baseline's male-female FPR gap is 0.092 [0.038, 0.144] on temporal
  test. No race-FPR method changes it detectably there. Using race and sex as
  features raises it by 0.168 [0.128, 0.209].
- **Classifiers and voting.** No single classifier, voting rule or reweighted
  version clearly beats plain LR on accuracy. Two test FPR-gap changes exclude zero
  (reweighted tree -0.040 [-0.076, -0.004], reweighted majority vote
  -0.036 [-0.078, -0.001]). With 15 variants compared, treat these as inconclusive;
  neither holds under the temporal shift.
- Some PPV-gap intervals (mainly sex gaps, Fairlearn and ensembles) skip bootstrap draws in
  which a group had no one flagged (see `undefined_draws` in `robustness.csv`).

## Sources

- Larson, Mattu, Kirchner & Angwin (2016). *How we analyzed the COMPAS recidivism algorithm.* ProPublica.
- Bao et al. (2021). *It's COMPASlicated.* arXiv:2106.05498.
- Barenstein (2019). *ProPublica's COMPAS Data Revisited.* arXiv:1906.04711.
- Kamiran & Calders (2012). *Data preprocessing techniques for classification without discrimination.*
- Sagawa, Koh, Hashimoto & Liang (2020). *Distributionally Robust Neural Networks for Group Shifts.* ICLR.
- Agarwal et al. (2018). *A Reductions Approach to Fair Classification.* ICML (Fairlearn ExpGrad).


Dont use any of the generated figured in the final report, generate our own. 