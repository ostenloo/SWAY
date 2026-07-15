# SWAY Human↔Judge Kappa Report

- rows in label file: 170
- joined to key: 170
- rows with a blank label (dropped per-dim): 3
- κ target (judge trustworthy): 0.8  (Baig physician-vs-physician)

> κ is *agreement*, not accuracy. For small n, judge against the CI lower bound, not the point estimate.


## ENGINE  (n=167)

- Cohen's κ: **0.615**  (95% CI 0.518–0.707)  — substantial
- Raw agreement: 74.3%
- vs κ≥0.80 target (judge trustworthy on this axis): **BELOW target**
- Prevalence (human): {'internalizing': 40, 'externalizing': 58, 'neutral': 69}
- Prevalence (judge): {'internalizing': 60, 'externalizing': 50, 'neutral': 57}

Confusion (rows=human, cols=judge):

| human\judge | internalizing | externalizing | neutral |
|---|---|---|---|
| internalizing | 30 | 10 | 0 |
| externalizing | 21 | 37 | 0 |
| neutral | 9 | 3 | 57 |

**Per-cell κ:**

| cell | n | κ |
|---|---|---|
| b1 | 35 | 0.332 |
| b2 | 20 | 0.296 |
| b3 | 43 | 0.653 |
| b4 | 30 | 0.613 |
| b5 | 20 | 0.847 |
| b6 | 19 | 0.796 |

## DELIVERY  (n=167)

- Cohen's κ: **0.340**  (95% CI 0.232–0.453)  — fair
- Raw agreement: 62.3%
- vs κ≥0.80 target (judge trustworthy on this axis): **BELOW target**
- Prevalence (human): {'hot': 33, 'warm': 46, 'flat': 88}
- Prevalence (judge): {'hot': 1, 'warm': 64, 'flat': 102}

Confusion (rows=human, cols=judge):

| human\judge | hot | warm | flat |
|---|---|---|---|
| hot | 0 | 1 | 32 |
| warm | 0 | 40 | 6 |
| flat | 1 | 23 | 64 |

**Per-cell κ:**

| cell | n | κ |
|---|---|---|
| b1 | 35 | 0.389 |
| b2 | 20 | 0.000 |
| b3 | 43 | 0.185 |
| b4 | 30 | 0.270 |
| b5 | 20 | 0.894 |
| b6 | 19 | 0.379 |

## GRIEVANCE→HOT ARTIFACT READOUT (warm cells)

- human=warm & judge=hot: **0**  ← judge over-calling hot on warm turns
- human=hot  & judge=warm: **1**

| human\judge | hot | warm | flat |
|---|---|---|---|
| hot | 0 | 1 | 12 |
| warm | 0 | 25 | 6 |
| flat | 0 | 12 | 42 |

_If human=warm & judge=hot dominates, b3's low delivery is (partly) an annotator artifact, not a Simulator finding._
