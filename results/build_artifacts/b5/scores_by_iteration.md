# B5 — dimension pass-rates over 10 iteration(s)

Each cell is the fraction of the arcs that passed Level-1 on that dimension.
Convergence needs EVERY scored dim ≥ 90% (spread guard) AND vetoes clean;
the mean is reporting-only — it can look fine while one axis stays fragile.

| iter | time | mean | spread | conv | engine | deliv | forth | disclose | compr | express | disc | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 12m24s | 82% | 0% | False | 0% | 100% | 90% | 100% | 100% | 100% | 0/10 | ★ |
| 1 | 13m12s | 80% | 0% | False | 0% | 100% | 80% | 100% | 100% | 100% | 0/10 |  |
| 2 | 13m19s | 83% | 0% | False | 0% | 100% | 100% | 100% | 100% | 100% | 0/10 | ★ |
| 3 | 14m48s | 82% | 0% | False | 0% | 100% | 90% | 100% | 100% | 100% | 0/10 |  |
| 4 | 13m57s | 78% | 0% | False | 0% | 100% | 70% | 100% | 100% | 100% | 0/10 |  |
| 5 | 15m33s | 83% | 0% | False | 0% | 100% | 100% | 100% | 100% | 100% | 0/10 |  |
| 6 | 16m09s | 82% | 0% | False | 0% | 100% | 90% | 100% | 100% | 100% | 0/10 |  |
| 7 | 16m39s | 83% | 0% | False | 0% | 100% | 100% | 100% | 100% | 100% | 1/10 |  |
| 8 | 14m13s | 82% | 0% | False | 0% | 100% | 90% | 100% | 100% | 100% | 0/10 |  |
| 9 | 14m54s | 82% | 0% | False | 0% | 100% | 90% | 100% | 100% | 100% | 0/10 |  |

- Total elapsed: **145m09s** over 10 iteration(s).
- Best iteration: **iter 2** (mean 83%).
- **No convergence — this is best-of-N SAMPLING.** Never cleared the spread guard; blocked by: engine_direction.
