# AnnoMI band calibration — disclosure report

grader_version: `v1`  
generated: 2026-08-28T05:55:08+00:00  
C9 sha256: `2b92b483d6cef87a78cca1e6b4cfb16ae433d9d628c48741f7a629714aae37f3`  
backends: `{'engine': 'local:command-r7b:latest', 'delivery': 'local:glm4:9b'}`

corpus: {'n_sessions': 133, 'n_client_turns': 4817, 'n_substantive': 3221, 'n_dropped_backchannel': 1596, 'graded_set': 'substantive'}

## Derived brackets (per-session q, EB-shrunk, percentiles)

| axis | direction | L_design | U | eligible | mu | tau2 | d_anno | informative |
|---|---|---|---|---|---|---|---|---|
| delivery | hot | 0.778 | 0.778 | 1/133 | 0.778 | 0.00000 | 0.032 | **NO** |
| delivery | warm | 0.222 | 0.222 | 1/133 | 0.222 | 0.00000 | 0.032 | **NO** |
| engine | externalizing | 0.739 | 0.795 | 38/133 | 0.614 | 0.04160 | 0.265 | yes |
| engine | internalizing | 0.499 | 0.694 | 38/133 | 0.386 | 0.04160 | 0.265 | yes |

### Uninformative brackets (D2.4 — warn and disclose, do not halt)

- **delivery/hot**: only 1 eligible sessions (< 25); no detectable between-session spread (tau2 <= 0); the percentiles collapse to the pooled mean
- **delivery/warm**: only 1 eligible sessions (< 25); no detectable between-session spread (tau2 <= 0); the percentiles collapse to the pooled mean

§6.5 expects this to fire on **delivery** — its per-session percentiles rest on a small eligible sample (11-33 sessions), where engine is comfortable (63-103).

## Claim

NOT a bound — a chosen position inside a real distribution: the simulated patient should be at least as directional as the P_lo-th percentile real patient of that direction.

## Transfer caveat

AnnoMI is MI counselling, not layoff support. The conditional q is more transferable than the density d; d_floor is operational, not norm-referenced.

## Per-cell entries

| cell | engine mode | engine band | delivery band | delivery d_floor |
|---|---|---|---|---|
| b1 | q_band | internalizing q in [0.4991, 0.6942], d_floor 0.15 | warm q in [0.2022, 0.2422] | 0.05 |
| b2 | q_band | internalizing q in [0.4991, 0.6942], d_floor 0.15 | hot q in [0.7578, 0.7978] | 0.05 |
| b3 | q_band | externalizing q in [0.7387, 0.7954], d_floor 0.15 | warm q in [0.2022, 0.2422] | 0.05 |
| b4 | q_band | externalizing q in [0.7387, 0.7954], d_floor 0.15 | hot q in [0.7578, 0.7978] | 0.05 |
| b5 | density_low | absent engine: d in [0.05, 0.12] (D2.3, d_lo > 0) | warm q in [0.2022, 0.2422] | 0.05 |
| b6 | density_low | absent engine: d in [0.05, 0.12] (D2.3, d_lo > 0) | hot q in [0.7578, 0.7978] | 0.05 |
