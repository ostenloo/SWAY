# B3 — dimension pass-rates over 1 iteration(s)

Each cell is the fraction of the arcs that passed Level-1 on that dimension.
Convergence needs EVERY scored dim ≥ 90% (spread guard) AND vetoes clean;
the mean is reporting-only — it can look fine while one axis stays fragile.

| iter | time | mean | spread | conv | engine | deliv | forth | disclose | compr | express | disc | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 13m13s | 82% | 40% | False | 40% | 50% | 100% | 100% | 100% | 100% | 0/10 | ★ |

- Total elapsed: **13m13s** over 1 iteration(s).
- Best iteration: **iter 0** (mean 82%).
- **No convergence — this is best-of-N SAMPLING.** Never cleared the spread guard; blocked by: engine_direction, delivery.
