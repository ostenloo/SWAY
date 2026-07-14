# SWAY Human↔Judge Kappa Report

- rows in label file: 170
- joined to key: 170
- rows with a blank label (dropped per-dim): 3
- κ target (judge trustworthy): 0.8  (Baig physician-vs-physician)

> κ is *agreement*, not accuracy. For small n, judge against the CI lower bound, not the point estimate.


## ENGINE  (n=167)

- Cohen's κ: **0.592**  (95% CI 0.490–0.685)  — moderate
- Raw agreement: 73.1%
- vs κ≥0.80 target (judge trustworthy on this axis): **BELOW target**
- Prevalence (human): {'internalizing': 40, 'externalizing': 58, 'neutral': 69}
- Prevalence (judge): {'internalizing': 45, 'externalizing': 68, 'neutral': 54}

Confusion (rows=human, cols=judge):

| human\judge | internalizing | externalizing | neutral |
|---|---|---|---|
| internalizing | 24 | 16 | 0 |
| externalizing | 14 | 44 | 0 |
| neutral | 7 | 8 | 54 |

**Per-cell κ:**

| cell | n | κ |
|---|---|---|
| b1 | 35 | 0.290 |
| b2 | 20 | 0.424 |
| b3 | 43 | 0.628 |
| b4 | 30 | 0.548 |
| b5 | 20 | 0.845 |
| b6 | 19 | 0.606 |

## DELIVERY  (n=167)

- Cohen's κ: **0.262**  (95% CI 0.157–0.370)  — fair
- Raw agreement: 49.7%
- vs κ≥0.80 target (judge trustworthy on this axis): **BELOW target**
- Prevalence (human): {'hot': 33, 'warm': 46, 'flat': 88}
- Prevalence (judge): {'hot': 67, 'warm': 51, 'flat': 49}

Confusion (rows=human, cols=judge):

| human\judge | hot | warm | flat |
|---|---|---|---|
| hot | 19 | 1 | 13 |
| warm | 8 | 33 | 5 |
| flat | 40 | 17 | 31 |

**Per-cell κ:**

| cell | n | κ |
|---|---|---|
| b1 | 35 | 0.224 |
| b2 | 20 | 0.361 |
| b3 | 43 | 0.209 |
| b4 | 30 | 0.359 |
| b5 | 20 | 0.177 |
| b6 | 19 | 0.422 |

## GRIEVANCE→HOT ARTIFACT READOUT (warm cells)

- human=warm & judge=hot: **8**  ← judge over-calling hot on warm turns
- human=hot  & judge=warm: **0**

| human\judge | hot | warm | flat |
|---|---|---|---|
| hot | 9 | 0 | 4 |
| warm | 8 | 18 | 5 |
| flat | 32 | 7 | 15 |

_If human=warm & judge=hot dominates, b3's low delivery is (partly) an annotator artifact, not a Simulator finding._
