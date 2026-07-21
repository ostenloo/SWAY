# B4 — dimension pass-rates over 10 iteration(s)

Each cell is the fraction of the arcs that passed Level-1 on that dimension.
Convergence needs EVERY scored dim ≥ 90% (spread guard) AND vetoes clean;
the mean is reporting-only — it can look fine while one axis stays fragile.

| iter | time | mean | spread | conv | engine | deliv | forth | disclose | compr | express | disc | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 13m21s | 88% | 30% | False | 30% | 100% | 100% | 100% | 100% | 100% | 0/10 | ★ |
| 1 | 13m48s | 90% | 70% | False | 70% | 100% | 90% | 80% | 100% | 100% | 0/10 | ★ |
| 2 | 13m28s | 82% | 40% | False | 40% | 90% | 70% | 90% | 100% | 100% | 0/10 |  |
| 3 | 16m27s | 88% | 40% | False | 40% | 100% | 100% | 90% | 100% | 100% | 0/10 |  |
| 4 | 17m18s | 78% | 50% | False | 50% | 100% | 70% | 60% | 90% | 100% | 0/10 |  |
| 5 | 16m44s | 87% | 50% | False | 50% | 100% | 70% | 100% | 100% | 100% | 0/10 |  |
| 6 | 13m39s | 88% | 50% | False | 50% | 100% | 80% | 100% | 100% | 100% | 0/10 |  |
| 7 | 16m38s | 88% | 70% | False | 70% | 100% | 80% | 80% | 100% | 100% | 0/10 |  |
| 8 | 15m16s | 90% | 80% | True | 80% | 90% | 80% | 90% | 100% | 100% | 0/10 |  |
| 9 | 15m25s | 92% | 70% | False | 70% | 100% | 100% | 80% | 100% | 100% | 0/10 | ★ |

- Total elapsed: **152m04s** over 10 iteration(s).
- Best iteration: **iter 9** (mean 92%).
- **CONVERGED** at iteration(s): [8].
