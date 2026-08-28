# AnnoMI rate-profile calibration — disclosure report

spec: `SWAY_ENGINE_DELIVERY_RATE_PROFILE_SPEC`  
grader_version: `v1`  
generated: 2026-08-28T19:03:56+00:00  
arc_length_T: **20** (bands are validated against it, §5.2/§11)  
C9 sha256: `59e38cd0ebe88ba9746fd0713378f080d8f21a67694d84378550592b4b4090a4`  
backends: `{'engine': 'local:command-r7b:latest', 'delivery': 'local:glm4:9b'}`

**Source: TRANSCRIBED from RATE §7 (no label cache on this host)**

## Bands

| cell | axis | label | role | band | measured | widened |
|---|---|---|---|---|---|---|
| b1 | engine | internalizing | on_direction | [0.144, 0.290] | yes |  |
| b1 | engine | externalizing | off_direction | [0.000, 0.100] | yes | yes |
| b1 | delivery | warm | on_direction | [0.150, 0.300] | **NO** |  |
| b1 | delivery | hot | off_direction | [0.000, 0.100] | **NO** |  |
| b2 | engine | internalizing | on_direction | [0.144, 0.290] | yes |  |
| b2 | engine | externalizing | off_direction | [0.000, 0.100] | yes | yes |
| b2 | delivery | hot | on_direction | [0.150, 0.300] | **NO** |  |
| b2 | delivery | warm | off_direction | [0.000, 0.100] | **NO** |  |
| b3 | engine | externalizing | on_direction | [0.106, 0.232] | yes |  |
| b3 | engine | internalizing | off_direction | [0.000, 0.100] | yes | yes |
| b3 | delivery | warm | on_direction | [0.150, 0.300] | **NO** |  |
| b3 | delivery | hot | off_direction | [0.000, 0.100] | **NO** |  |
| b4 | engine | externalizing | on_direction | [0.106, 0.232] | yes |  |
| b4 | engine | internalizing | off_direction | [0.000, 0.100] | yes | yes |
| b4 | delivery | hot | on_direction | [0.150, 0.300] | **NO** |  |
| b4 | delivery | warm | off_direction | [0.000, 0.100] | **NO** |  |
| b5 | engine | internalizing | off_direction | [0.000, 0.100] | yes | yes |
| b5 | engine | externalizing | off_direction | [0.000, 0.100] | yes | yes |
| b5 | delivery | warm | on_direction | [0.150, 0.300] | **NO** |  |
| b5 | delivery | hot | off_direction | [0.000, 0.100] | **NO** |  |
| b6 | engine | internalizing | off_direction | [0.000, 0.100] | yes | yes |
| b6 | engine | externalizing | off_direction | [0.000, 0.100] | yes | yes |
| b6 | delivery | hot | on_direction | [0.150, 0.300] | **NO** |  |
| b6 | delivery | warm | off_direction | [0.000, 0.100] | **NO** |  |
| p1 | engine | internalizing | on_direction | [0.144, 0.290] | yes |  |
| p1 | engine | externalizing | off_direction | [0.000, 0.100] | yes | yes |
| p1 | delivery | warm | on_direction | [0.150, 0.300] | **NO** |  |
| p1 | delivery | hot | off_direction | [0.000, 0.100] | **NO** |  |
| p2 | engine | internalizing | off_direction | [0.000, 0.100] | yes | yes |
| p2 | engine | externalizing | off_direction | [0.000, 0.100] | yes | yes |
| p2 | delivery | warm | on_direction | [0.150, 0.300] | **NO** |  |
| p2 | delivery | hot | off_direction | [0.000, 0.100] | **NO** |  |
| p3 | engine | internalizing | off_direction | [0.000, 0.100] | yes | yes |
| p3 | engine | externalizing | off_direction | [0.000, 0.100] | yes | yes |
| p3 | delivery | warm | on_direction | [0.150, 0.300] | **NO** |  |
| p3 | delivery | hot | off_direction | [0.000, 0.100] | **NO** |  |

## Delivery is DECLARED, not measured (§8)

AnnoMI cannot supply delivery targets: 96.8% of its turns are flat, only 102 of 3,221 are marked, and the MEDIAN conversation contains zero warm or hot turns. This is a property of the corpus, not the grader — three independent graders agree (2.5% / 6.5% / 2.2% marked on the same 400 turns), the champion marks 36.0% on simulator output against 37.3% and 38.3% from two human annotators on the same turns, and stratifying by session quality barely moves it (hot 2.4% in the deliberately-poor sessions vs 1.8% in the good ones). Counselling with an intact alliance contains almost no hostility aimed at the counsellor. The declared edges mirror §7's measured ENGINE geometry — a marked stance surfacing on 3 to 6 turns of 20, against an off-pole ceiling near 0.10 — because that is the only empirical anchor for how often a stance surfaces in an arc. DECLARED, NOT MEASURED (§8).

Monitoring reports engine and delivery band-fit **separately** so a declared target is never reported as a measured one. Reverting delivery to per-turn monotone scoring is rejected: it reinstates the failure this redesign exists to remove, on the axis where the hot profiles were already hardest to produce.

## §5.2 widenings

- **engine/externalizing** (off_direction): widened U 0.0890 -> 0.1000 to reach the 2/T = 0.1000 span floor at T = 20 (§5.2). L is pinned at 0 by §5.3, so the widening LOOSENS the off-direction ceiling above what was measured.
- **engine/internalizing** (off_direction): widened U 0.0710 -> 0.1000 to reach the 2/T = 0.1000 span floor at T = 20 (§5.2). L is pinned at 0 by §5.3, so the widening LOOSENS the off-direction ceiling above what was measured.

## What this costs (§10)

EVERY component is scenario-dependent (§10). The conditional ratio's scenario-invariance is given up deliberately: it was an argument, never a measurement, and six measured failures sit against it. AnnoMI is MI counselling, not layoff support; the counselling-to-layoff gap is a standing limitation on BOTH axes.

at a true rate of 0.20 the 95% interval on a single 20-turn arc runs roughly 0.02-0.38. No individual arc's rate is a measurement (§10).

## Percentile window

25th-75th targets the middle half of conversations that lean the intended way. This is a DESIGN DECISION, not a measurement (§6.1); changing it needs its own justification, not a retune.

## Source disclosure

Transcribed from RATE §7, not re-derived: the grader label cache those percentiles were read off is not in this repo. Two disclosures travel with these numbers. (1) §7's group sizes sum to 115 over 100 eligible conversations, so the source run's groups OVERLAP; §6 step 4 says 'partition' and rate_derive.group_conversations implements the disjoint version with low-rate tested first, so a re-derivation will shift these edges. (2) §7 flags the §5.2 widening on EXTERNALIZING p_int in [0, 0.071] but not on the equally sub-span INTERNALIZING p_ext in [0, 0.089] or either NEUTRAL band; it is applied uniformly here and recorded per band.
