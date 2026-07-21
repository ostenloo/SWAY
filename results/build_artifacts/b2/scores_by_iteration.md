# B2 — dimension pass-rates over 10 iteration(s)

Each cell is the fraction of the arcs that passed Level-1 on that dimension.
Convergence needs EVERY scored dim ≥ 90% (spread guard) AND vetoes clean;
the mean is reporting-only — it can look fine while one axis stays fragile.

| iter | time | mean | spread | conv | engine | deliv | forth | disclose | compr | express | disc | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 13m58s | 78% | 100% | True | 100% | 0% | 70% | 100% | 100% | 100% | 0/10 | ★ |
| 1 | 13m20s | 83% | 100% | True | 100% | 0% | 100% | 100% | 100% | 100% | 0/10 | ★ |
| 2 | 13m17s | 82% | 100% | True | 100% | 0% | 90% | 100% | 100% | 100% | 0/10 |  |
| 3 | 13m48s | 80% | 90% | True | 90% | 0% | 90% | 100% | 100% | 100% | 0/10 |  |
| 4 | 11m57s | 82% | 100% | True | 100% | 0% | 90% | 100% | 100% | 100% | 0/10 |  |
| 5 | 12m35s | 80% | 100% | True | 100% | 0% | 80% | 100% | 100% | 100% | 0/10 |  |
| 6 | 14m28s | 77% | 100% | True | 100% | 0% | 60% | 100% | 100% | 100% | 0/10 |  |
| 7 | 13m29s | 82% | 100% | True | 100% | 20% | 70% | 100% | 100% | 100% | 0/10 |  |
| 8 | 14m32s | 81% | 86% | False | 86% | 0% | 100% | 100% | 100% | 100% | 3/10 |  |
| 9 | 13m23s | 82% | 90% | True | 90% | 0% | 100% | 100% | 100% | 100% | 0/10 |  |

- Total elapsed: **134m48s** over 10 iteration(s).
- Best iteration: **iter 1** (mean 83%).
- **CONVERGED** at iteration(s): [0, 1, 2, 3, 4, 5, 6, 7, 9].
