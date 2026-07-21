# B6 — dimension pass-rates over 10 iteration(s)

Each cell is the fraction of the arcs that passed Level-1 on that dimension.
Convergence needs EVERY scored dim ≥ 90% (spread guard) AND vetoes clean;
the mean is reporting-only — it can look fine while one axis stays fragile.

| iter | time | mean | spread | conv | engine | deliv | forth | disclose | compr | express | disc | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 13m11s | 80% | 0% | False | 0% | 100% | 90% | 90% | 100% | 100% | 0/10 | ★ |
| 1 | 16m07s | 72% | 0% | False | 0% | 90% | 50% | 90% | 100% | 100% | 0/10 |  |
| 2 | 15m34s | 72% | 0% | False | 0% | 89% | 56% | 89% | 100% | 100% | 1/10 |  |
| 3 | 16m50s | 82% | 0% | False | 0% | 100% | 90% | 100% | 100% | 100% | 0/10 | ★ |
| 4 | 15m52s | 75% | 0% | False | 0% | 90% | 90% | 70% | 100% | 100% | 0/10 |  |
| 5 | 15m42s | 78% | 0% | False | 0% | 100% | 90% | 80% | 100% | 100% | 0/10 |  |
| 6 | 13m40s | 80% | 0% | False | 0% | 100% | 90% | 90% | 100% | 100% | 0/10 |  |
| 7 | 14m43s | 82% | 0% | False | 0% | 100% | 90% | 100% | 100% | 100% | 0/10 |  |
| 8 | 14m58s | 80% | 0% | False | 0% | 100% | 90% | 90% | 100% | 100% | 0/10 |  |
| 9 | 16m55s | 81% | 0% | False | 0% | 100% | 89% | 100% | 100% | 100% | 1/10 |  |

- Total elapsed: **153m32s** over 10 iteration(s).
- Best iteration: **iter 3** (mean 82%).
- **No convergence — this is best-of-N SAMPLING.** Never cleared the spread guard; blocked by: engine_direction.
