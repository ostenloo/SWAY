# P2 — dimension pass-rates over 10 iteration(s)

Each cell is the fraction of the arcs that passed Level-1 on that dimension.
Convergence needs EVERY scored dim ≥ 90% (spread guard) AND vetoes clean;
the mean is reporting-only — it can look fine while one axis stays fragile.

| iter | time | mean | spread | conv | engine | deliv | forth | disclose | compr | express | disc | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 12m52s | 47% | 0% | False | 0% | 100% | 80% | 100% | 0% | 0% | 0/10 | ★ |
| 1 | 13m19s | 48% | 0% | False | 0% | 100% | 90% | 100% | 0% | 0% | 0/10 | ★ |
| 2 | 14m12s | 50% | 0% | False | 0% | 100% | 100% | 100% | 0% | 0% | 0/10 | ★ |
| 3 | 13m58s | 47% | 0% | False | 0% | 90% | 90% | 100% | 0% | 0% | 0/10 |  |
| 4 | 13m42s | 47% | 0% | False | 0% | 100% | 80% | 100% | 0% | 0% | 0/10 |  |
| 5 | 13m35s | 48% | 0% | False | 0% | 100% | 90% | 100% | 0% | 0% | 0/10 |  |
| 6 | 13m35s | 47% | 0% | False | 0% | 100% | 80% | 100% | 0% | 0% | 0/10 |  |
| 7 | 13m14s | 47% | 0% | False | 0% | 100% | 80% | 100% | 0% | 0% | 0/10 |  |
| 8 | 16m08s | 42% | 0% | False | 0% | 80% | 70% | 100% | 0% | 0% | 0/10 |  |
| 9 | 14m03s | 48% | 0% | False | 0% | 90% | 100% | 100% | 0% | 0% | 0/10 |  |

- Total elapsed: **138m38s** over 10 iteration(s).
- Best iteration: **iter 2** (mean 50%).
- **No convergence — this is best-of-N SAMPLING.** Never cleared the spread guard; blocked by: engine_direction, comprehension, expression.
