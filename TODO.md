# TODO

Status as of 2026-11-23: the project now targets equal FPR by race under a 25%
FPR cap, following the proposal feedback. `run_all.py`, `figures.py` and
`robustness.py` have been run; results are in README.md.

## Everyone
- [ ] Review your module in `methods/` and the decisions in README.md
      (criterion, FPR cap, selection rule, shift split). Raise objections at the next check-in.
- [ ] Don't add preprocessing outside `data.py`.

## Robustness of results
- [x] Ten seeds (splits, torch, ensembles) for every reported method.
- [x] 95% CIs for accuracy, FPR and the FPR/FNR/PPV gaps; paired against plain LR.
- [x] FPR-cap sensitivity, 15-35%.
- [x] Error bars on both figures.
- [x] λ selection at tolerance 0.03, 0.05, 0.08.
- [x] Rates by race, sex gaps and AUC with CIs.
- [x] Classifier and voting ablations over ten seeds.
- [x] Second shift set (felony to misdemeanor).
- [x] Direct test-to-shift changes with CIs; figures for both shifts.

## Jihyun: data, baseline, bias analysis
- [ ] Data-bias section:
  - rearrest as a proxy label
  - priors_count inheriting policing bias (it is the strongest single race
    proxy, AUC 0.62)
  - inconsistent race labels (Bao et al.)
  - the label truncation after 2014-04 (Barenstein 2019)
- [ ] Decide whether to add `c_charge_desc` or `age_cat` features. Changing
      features changes everyone's numbers, so decide early.

## Justin: reweighting and ensembles
- [ ] Reweighting by race barely moves the FPR gap (-0.003 [-0.013, +0.006]).
      Explain why; one candidate is that the race-blind features proxy race.
- [ ] Voting ablations show no clear benefit over ten seeds: decide whether to report or drop.
- [ ] Optional: reweight on race × label × sex, or tune the base models on validation.

## William: fairness-constrained LR
- [ ] The penalty costs about 0.04 accuracy on test and its FPR gap rises
      sharply at a 35% cap. Investigate, e.g. a squared penalty or a surrogate nearer the threshold.
- [ ] Main figure: `results/pareto.png`.

## Myra: DRO and shift
- [ ] Explain why neither DRO + λ nor per-race thresholds keep equal FPR under
      the temporal shift (gaps 0.091 and -0.097), and why DRO alone matches plain LR.
- [ ] Explain why the penalties keep their gain on misdemeanors but not in the later
      period.
- [ ] Try η values beyond {0.01, 0.1} and groups by race only.
- [ ] Lead the results and discussion sections; `results/shift.png` and `shift_charge.png` are the key figures.

## Week 4: report
- [ ] Each person writes their method section. Jihyun leads intro and data;
      Myra leads results and discussion.
- [ ] Fix the proposal wording: method 3 is "distributionally robust
      optimization," not "distributed."
