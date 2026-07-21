# P3 — dimension pass-rates over 10 iteration(s)

Each cell is the fraction of the arcs that passed Level-1 on that dimension.
Convergence needs EVERY scored dim ≥ 90% (spread guard) AND vetoes clean;
the mean is reporting-only — it can look fine while one axis stays fragile.

| iter | time | mean | spread | conv | engine | deliv | forth | disclose | compr | express | disc | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 12m24s | 63% | 0% | False | 0% | 100% | 80% | 100% | 0% | 100% | 0/10 | ★ |
| 1 | 13m22s | 62% | 0% | False | 0% | 100% | 70% | 100% | 0% | 100% | 0/10 |  |
| 2 | 13m44s | 60% | 0% | False | 0% | 100% | 60% | 100% | 0% | 100% | 0/10 |  |
| 3 | 12m18s | 67% | 0% | False | 0% | 100% | 100% | 100% | 0% | 100% | 0/10 | ★ |
| 4 | 13m01s | 65% | 0% | False | 0% | 100% | 90% | 100% | 0% | 100% | 0/10 |  |
| 5 | 12m45s | 65% | 0% | False | 0% | 100% | 90% | 100% | 0% | 100% | 0/10 |  |
| 6 | 13m11s | 67% | 0% | False | 0% | 100% | 100% | 100% | 0% | 100% | 0/10 |  |
| 7 | 13m31s | 65% | 0% | False | 0% | 100% | 90% | 100% | 0% | 100% | 0/10 |  |
| 8 | 15m05s | 67% | 0% | False | 0% | 100% | 100% | 100% | 0% | 100% | 0/10 |  |
| 9 | 14m42s | 62% | 0% | False | 0% | 100% | 70% | 100% | 0% | 100% | 0/10 |  |

- Total elapsed: **134m04s** over 10 iteration(s).
- Best iteration: **iter 3** (mean 67%).
- **No convergence — this is best-of-N SAMPLING.** Never cleared the spread guard; blocked by: engine_direction, comprehension.
