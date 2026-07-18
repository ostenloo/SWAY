# Batch03 Annotator-Model Sweep — κ / AC1 vs Human & LLM Raters

_Generated 2026-07-17. Method: CSV-based annotation of all 150 batch03 turns (context = `context_prev_assistant` + `patient_turn`, i.e. exactly what the human labeler saw), using the production annotator prompt (`build.annotator_system_prompt` / `annotator_user_prompt`). Cohen κ and Gwet AC1, aligned on turn_id, per-dim blanks dropped, labels lowercased. Raw per-model keys: `_key_batch03.<model>.csv`._

> **Bottom line.** With a second human (qingqing) the ceilings are now real: engine human–human AC1 **0.658**, delivery human–human AC1 **0.548**. Reasoning models *exceed* the human ceiling on engine and *match* it on delivery. Engine annotation is solved; delivery ~0.55 is intrinsic human subjectivity, not a model gap — report-only is correct and permanent.

## Models (VRAM / speed on the RTX 5090)

| model | lineage | VRAM (GB) | time / 150 turns | notes |
|---|---|---|---|---|
| llama3.1:8b | Llama | 5.9 | 1.9 min | current production annotator; non-reasoning |
| qwen3:14b | Qwen | 10.5 | 11.5 min | reasoning; **same family as the simulator** |
| deepseek-r1:8b | Llama+DeepSeek | 6.3 | 43 min | reasoning (Llama arch); very slow; 8 parse-blanks |
| gpt-oss:20b | OpenAI | 12.9 | ~8 min | reasoning; broken delivery (hot=2/150); crashed first run at 96/150 |
| phi4-reasoning:14b | Microsoft | 12.8 | ~62 min | reasoning; **best engine**; slow; needs robust JSON parse |

## ENGINE

### Inter-rater ceilings (human & LLM references)

| pair | Cohen κ | AC1 | n |
|---|---|---|---|
| austin × qingqing **← human–human ceiling** | 0.607 | 0.658 | 147 |
| austin × opus48 | 0.776 | 0.809 | 149 |
| qingqing × opus48 | 0.662 | 0.711 | 148 |

### Each model vs each reference (Cohen κ / AC1)

| model | vs austin | vs qingqing | vs opus48 |
|---|---|---|---|
| llama3.1:8b | 0.575 / 0.584 | 0.547 / 0.550 | 0.643 / 0.658 |
| qwen3:14b | 0.696 / 0.728 | 0.620 / 0.658 | 0.769 / 0.799 |
| deepseek-r1:8b | 0.671 / 0.731 | 0.581 / 0.646 | 0.789 / 0.833 |
| gpt-oss:20b | 0.647 / 0.703 | 0.589 / 0.652 | 0.779 / 0.821 |
| phi4-reasoning:14b | 0.744 / 0.777 | 0.632 / 0.678 | 0.852 / 0.876 |

### Full pairwise AC1 (references + models)

```
                austin   qingqing     opus48  llama3.1:  qwen3:14b  deepseek-  gpt-oss:2  phi4-reas
     austin      1.000      0.658      0.809      0.584      0.728      0.731      0.703      0.777
   qingqing      0.658      1.000      0.711      0.550      0.658      0.646      0.652      0.678
     opus48      0.809      0.711      1.000      0.658      0.799      0.833      0.821      0.876
  llama3.1:      0.584      0.550      0.658      1.000      0.735      0.654      0.588      0.685
  qwen3:14b      0.728      0.658      0.799      0.735      1.000      0.791      0.761      0.817
  deepseek-      0.731      0.646      0.833      0.654      0.791      1.000      0.852      0.851
  gpt-oss:2      0.703      0.652      0.821      0.588      0.761      0.852      1.000      0.828
  phi4-reas      0.777      0.678      0.876      0.685      0.817      0.851      0.828      1.000
```

## DELIVERY

### Inter-rater ceilings (human & LLM references)

| pair | Cohen κ | AC1 | n |
|---|---|---|---|
| austin × qingqing **← human–human ceiling** | 0.447 | 0.548 | 149 |
| austin × opus48 | 0.683 | 0.772 | 149 |
| qingqing × opus48 | 0.520 | 0.606 | 150 |

### Each model vs each reference (Cohen κ / AC1)

| model | vs austin | vs qingqing | vs opus48 |
|---|---|---|---|
| llama3.1:8b | 0.432 / 0.493 | 0.372 / 0.395 | 0.425 / 0.475 |
| qwen3:14b | 0.419 / 0.605 | 0.309 / 0.454 | 0.554 / 0.696 |
| deepseek-r1:8b | 0.409 / 0.537 | 0.489 / 0.556 | 0.573 / 0.668 |
| gpt-oss:20b | 0.218 / 0.545 | 0.335 / 0.540 | 0.254 / 0.585 |
| phi4-reasoning:14b | 0.298 / 0.595 | 0.277 / 0.493 | 0.507 / 0.718 |

### Full pairwise AC1 (references + models)

```
                austin   qingqing     opus48  llama3.1:  qwen3:14b  deepseek-  gpt-oss:2  phi4-reas
     austin      1.000      0.548      0.772      0.493      0.605      0.537      0.545      0.595
   qingqing      0.548      1.000      0.606      0.395      0.454      0.556      0.540      0.493
     opus48      0.772      0.606      1.000      0.475      0.696      0.668      0.585      0.718
  llama3.1:      0.493      0.395      0.475      1.000      0.480      0.487      0.231      0.286
  qwen3:14b      0.605      0.454      0.696      0.480      1.000      0.652      0.634      0.715
  deepseek-      0.537      0.556      0.668      0.487      0.652      1.000      0.549      0.604
  gpt-oss:2      0.545      0.540      0.585      0.231      0.634      0.549      1.000      0.843
  phi4-reas      0.595      0.493      0.718      0.286      0.715      0.604      0.843      1.000
```

## Interpretation

- **Engine is solved.** Two humans agree at AC1 0.658; reasoning models agree with austin at 0.70–0.78 — *above* the human–human ceiling. phi4-reasoning is best (0.777 austin / 0.876 opus48). The earlier "models share a bias austin doesn't" worry is refuted: the two humans diverge from each other *more* than the models diverge from austin.

- **Delivery ~0.55 is the human limit, not a model failure.** Two humans agree at AC1 0.548; models match it (qwen3 0.605 vs austin; deepseek 0.556 vs qingqing). The austin↔opus48 delivery of 0.772 was anomalous alignment; qingqing reveals the true ceiling. Report-only is correct and permanent.

- **Reasoning helps engine (inference), not delivery (surface tone).** On the hot↔flat boundary the reasoning models over-apply the "employer-grievance-isn't-hot" carve-out and under-call hot (qwen3 45% / gpt-oss 5% hot recall vs austin); llama over-calls (91% recall, low precision). Delivery disagreement is concentrated there and is genuinely subjective (humans split too).

- **Coupling concern is dead.** Non-qwen reasoning models (deepseek, gpt-oss, phi4) match or beat qwen3 on engine, so qwen3's edge is the general reasoning reading, not family common-mode with the qwen simulator.

- **qingqing is the slightly divergent human** (agrees with everyone ~0.65–0.71); austin/opus48/models form a tighter cluster.

## Recommendation

- **Engine annotator → a reasoning model.** phi4-reasoning has the best agreement but is slow (~22 s/turn) and needs the robust JSON parser; qwen3/deepseek are close. Any of them is at/above human reliability.

- **Delivery → keep report-only.** It's at the human ceiling; no model swap improves it. If a per-turn label is needed, llama's high hot-recall (91%) is the safer error for a gate (false-positive hot beats missed hot).

- Changing the fidelity annotator is a **convergence re-derivation trigger** (spec §8): re-derive thresholds under the new annotator before gating.
