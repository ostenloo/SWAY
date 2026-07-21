# B3 — dimension pass-rates over 10 iteration(s)

Each cell is the fraction of the arcs that passed Level-1 on that dimension.
Convergence needs EVERY scored dim ≥ 90% (spread guard) AND vetoes clean;
the mean is reporting-only — it can look fine while one axis stays fragile.

| iter | time | mean | spread | conv | engine | deliv | forth | disclose | compr | express | disc | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 11m56s | 78% | 60% | False | 60% | 20% | 90% | 100% | 100% | 100% | 0/10 | ★ |
| 1 | 12m03s | 90% | 90% | True | 90% | 50% | 100% | 100% | 100% | 100% | 0/10 | ★ |
| 2 | 14m21s | 82% | 90% | True | 90% | 10% | 90% | 100% | 100% | 100% | 0/10 |  |
| 3 | 12m05s | 85% | 90% | True | 90% | 20% | 100% | 100% | 100% | 100% | 0/10 |  |
| 4 | 11m29s | 77% | 60% | False | 60% | 0% | 100% | 100% | 100% | 100% | 0/10 |  |
| 5 | 14m02s | 82% | 100% | True | 100% | 10% | 90% | 90% | 100% | 100% | 0/10 |  |
| 6 | 14m05s | 82% | 100% | True | 100% | 10% | 90% | 90% | 100% | 100% | 0/10 |  |
| 7 | 13m12s | 83% | 80% | True | 80% | 30% | 90% | 100% | 100% | 100% | 0/10 |  |
| 8 | 12m31s | 73% | 60% | False | 60% | 0% | 80% | 100% | 100% | 100% | 0/10 |  |
| 9 | 13m25s | 85% | 80% | True | 80% | 30% | 100% | 100% | 100% | 100% | 0/10 |  |

- Total elapsed: **129m10s** over 10 iteration(s).
- Best iteration: **iter 1** (mean 90%).
- **CONVERGED** at iteration(s): [1, 2, 3, 5, 6, 7, 9].
