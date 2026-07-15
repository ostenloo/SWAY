# SWAY Human↔Judge Kappa Report

- rows in label file: 170
- joined to key: 170
- rows with a blank label (dropped per-dim): 3
- κ target (judge trustworthy): 0.8  (Baig physician-vs-physician)

> κ is *agreement*, not accuracy. For small n, judge against the CI lower bound, not the point estimate.


## ENGINE  (n=167)

- Cohen's κ:  **0.586**  (95% CI 0.482–0.682)  — moderate
- Gwet's AC1: **0.588**  (95% CI 0.488–0.687)  — moderate  _(prevalence-robust; the more defensible number when the prevalence rows below differ)_
- Raw agreement: 72.5%
- vs κ≥0.80 target (judge trustworthy on this axis): **BELOW target**
- Prevalence (human): {'internalizing': 40, 'externalizing': 58, 'neutral': 69}
- Prevalence (judge): {'internalizing': 52, 'externalizing': 60, 'neutral': 55}

Confusion (rows=human, cols=judge):

| human\judge | internalizing | externalizing | neutral |
|---|---|---|---|
| internalizing | 27 | 13 | 0 |
| externalizing | 19 | 39 | 0 |
| neutral | 6 | 8 | 55 |

**Per-cell κ:**

| cell | n | κ |
|---|---|---|
| b1 | 35 | 0.332 |
| b2 | 20 | 0.296 |
| b3 | 43 | 0.557 |
| b4 | 30 | 0.605 |
| b5 | 20 | 0.847 |
| b6 | 19 | 0.695 |

## DELIVERY  (n=167)

- Cohen's κ:  **0.452**  (95% CI 0.346–0.558)  — moderate
- Gwet's AC1: **0.469**  (95% CI 0.364–0.575)  — moderate  _(prevalence-robust; the more defensible number when the prevalence rows below differ)_
- Raw agreement: 64.1%
- vs κ≥0.80 target (judge trustworthy on this axis): **BELOW target**
- Prevalence (human): {'hot': 33, 'warm': 46, 'flat': 88}
- Prevalence (judge): {'hot': 50, 'warm': 56, 'flat': 61}

Confusion (rows=human, cols=judge):

| human\judge | hot | warm | flat |
|---|---|---|---|
| hot | 23 | 0 | 10 |
| warm | 3 | 38 | 5 |
| flat | 24 | 18 | 46 |

**Per-cell κ:**

| cell | n | κ |
|---|---|---|
| b1 | 35 | 0.291 |
| b2 | 20 | 0.412 |
| b3 | 43 | 0.354 |
| b4 | 30 | 0.599 |
| b5 | 20 | 0.717 |
| b6 | 19 | 0.422 |

## GRIEVANCE→HOT ARTIFACT READOUT (warm cells)

- human=warm & judge=hot: **3**  ← judge over-calling hot on warm turns
- human=hot  & judge=warm: **0**

| human\judge | hot | warm | flat |
|---|---|---|---|
| hot | 9 | 0 | 4 |
| warm | 3 | 23 | 5 |
| flat | 18 | 7 | 29 |

_If human=warm & judge=hot dominates, b3's low delivery is (partly) an annotator artifact, not a Simulator finding._
