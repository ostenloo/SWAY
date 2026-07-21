# B1 — dimension pass-rates over 10 iteration(s)

Each cell is the fraction of the arcs that passed Level-1 on that dimension.
Convergence needs EVERY scored dim ≥ 90% (spread guard) AND vetoes clean;
the mean is reporting-only — it can look fine while one axis stays fragile.

| iter | time | mean | spread | conv | engine | deliv | forth | disclose | compr | express | disc | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7m31s | 97% | 100% | True | 100% | 100% | 80% | 100% | 100% | 100% | 0/10 | ★ |
| 1 | 12m44s | 97% | 80% | True | 80% | 100% | 100% | 100% | 100% | 100% | 0/10 |  |
| 2 | 14m10s | 98% | 90% | True | 90% | 100% | 100% | 100% | 100% | 100% | 0/10 | ★ |
| 3 | 12m40s | 100% | 100% | True | 100% | 100% | 100% | 100% | 100% | 100% | 0/10 | ★ |
| 4 | 12m33s | 98% | 100% | True | 100% | 100% | 90% | 100% | 100% | 100% | 0/10 |  |
| 5 | 12m58s | 95% | 80% | True | 80% | 100% | 90% | 100% | 100% | 100% | 0/10 |  |
| 6 | 13m46s | 95% | 90% | True | 90% | 100% | 80% | 100% | 100% | 100% | 0/10 |  |
| 7 | 17m32s | 98% | 100% | True | 100% | 100% | 90% | 100% | 100% | 100% | 0/10 |  |
| 8 | 12m06s | 98% | 90% | True | 90% | 100% | 100% | 100% | 100% | 100% | 0/10 |  |
| 9 | 13m01s | 97% | 90% | True | 90% | 100% | 90% | 100% | 100% | 100% | 0/10 |  |

- Total elapsed: **129m01s** over 10 iteration(s).
- Best iteration: **iter 3** (mean 100%).
- **CONVERGED** at iteration(s): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
