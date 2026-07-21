# P1 — dimension pass-rates over 10 iteration(s)

Each cell is the fraction of the arcs that passed Level-1 on that dimension.
Convergence needs EVERY scored dim ≥ 90% (spread guard) AND vetoes clean;
the mean is reporting-only — it can look fine while one axis stays fragile.

| iter | time | mean | spread | conv | engine | deliv | forth | disclose | compr | express | disc | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 11m47s | 80% | 100% | True | 100% | 100% | 80% | 0% | 100% | 100% | 0/10 | ★ |
| 1 | 12m59s | 77% | 90% | True | 90% | 100% | 60% | 10% | 100% | 100% | 0/10 |  |
| 2 | 12m10s | 78% | 90% | True | 90% | 100% | 80% | 0% | 100% | 100% | 0/10 |  |
| 3 | 12m10s | 78% | 90% | True | 90% | 100% | 80% | 0% | 100% | 100% | 0/10 |  |
| 4 | 15m35s | 82% | 90% | True | 90% | 100% | 80% | 20% | 100% | 100% | 0/10 | ★ |
| 5 | 12m16s | 77% | 90% | True | 90% | 100% | 70% | 0% | 100% | 100% | 0/10 |  |
| 6 | 12m05s | 77% | 100% | True | 100% | 100% | 50% | 10% | 100% | 100% | 0/10 |  |
| 7 | 14m03s | 83% | 100% | False | 100% | 100% | 100% | 0% | 100% | 100% | 1/10 | ★ |
| 8 | 13m47s | 81% | 100% | False | 100% | 100% | 71% | 14% | 100% | 100% | 3/10 |  |
| 9 | 13m05s | 78% | 100% | False | 100% | 100% | 56% | 11% | 100% | 100% | 1/10 |  |

- Total elapsed: **129m57s** over 10 iteration(s).
- Best iteration: **iter 7** (mean 83%).
- **CONVERGED** at iteration(s): [0, 1, 2, 3, 4, 5, 6].
