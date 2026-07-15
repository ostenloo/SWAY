# SWAY Human↔Judge Kappa Report

- rows in label file: 170
- joined to key: 170
- rows with a blank label (dropped per-dim): 3
- κ target (judge trustworthy): 0.8  (Baig physician-vs-physician)

> κ is *agreement*, not accuracy. For small n, judge against the CI lower bound, not the point estimate.


## ENGINE  (n=167)

- Cohen's κ:  **0.601**  (95% CI 0.501–0.694)  — substantial
- Gwet's AC1: **0.620**  (95% CI 0.519–0.713)  — substantial  _(prevalence-robust; the more defensible number when the prevalence rows below differ)_
- Raw agreement: 74.3%
- vs κ≥0.80 target (judge trustworthy on this axis): **BELOW target**
- Prevalence (human): {'internalizing': 40, 'externalizing': 58, 'neutral': 69}
- Prevalence (judge): {'internalizing': 30, 'externalizing': 71, 'neutral': 66}

Confusion (rows=human, cols=judge):

| human\judge | internalizing | externalizing | neutral |
|---|---|---|---|
| internalizing | 18 | 16 | 6 |
| externalizing | 8 | 48 | 2 |
| neutral | 4 | 7 | 58 |

**Per-cell κ:**

| cell | n | κ |
|---|---|---|
| b1 | 35 | 0.383 |
| b2 | 20 | 0.545 |
| b3 | 43 | 0.499 |
| b4 | 30 | 0.647 |
| b5 | 20 | 0.764 |
| b6 | 19 | 0.695 |

## DELIVERY  (n=167)

- Cohen's κ:  **0.333**  (95% CI 0.213–0.448)  — fair
- Gwet's AC1: **0.466**  (95% CI 0.362–0.575)  — moderate  _(prevalence-robust; the more defensible number when the prevalence rows below differ)_
- Raw agreement: 61.7%
- vs κ≥0.80 target (judge trustworthy on this axis): **BELOW target**
- Prevalence (human): {'hot': 33, 'warm': 46, 'flat': 88}
- Prevalence (judge): {'hot': 5, 'warm': 61, 'flat': 101}

Confusion (rows=human, cols=judge):

| human\judge | hot | warm | flat |
|---|---|---|---|
| hot | 3 | 1 | 29 |
| warm | 0 | 37 | 9 |
| flat | 2 | 23 | 63 |

**Per-cell κ:**

| cell | n | κ |
|---|---|---|
| b1 | 35 | 0.222 |
| b2 | 20 | 0.361 |
| b3 | 43 | 0.102 |
| b4 | 30 | 0.322 |
| b5 | 20 | 0.792 |
| b6 | 19 | 0.248 |

## GRIEVANCE→HOT ARTIFACT READOUT (warm cells)

- human=warm & judge=hot: **0**  ← judge over-calling hot on warm turns
- human=hot  & judge=warm: **0**

| human\judge | hot | warm | flat |
|---|---|---|---|
| hot | 0 | 0 | 13 |
| warm | 0 | 24 | 7 |
| flat | 1 | 15 | 38 |

_If human=warm & judge=hot dominates, b3's low delivery is (partly) an annotator artifact, not a Simulator finding._
