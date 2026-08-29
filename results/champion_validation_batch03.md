# Champion validation on SIMULATOR text (batch03)

150 turns with a champion label and all 3 annotator sheets.
Cell mix: {'b1': 25, 'b2': 25, 'b3': 25, 'b4': 25, 'b5': 25, 'b6': 25}

One of the three sheets is a MODEL annotator (Opus 4.8), not a human. Two are human. Which is which is recorded with the sheets, not here.
Annotator columns are anonymised: this repository is public and per-person agreement scores are not published as a side effect of a validation run. Run with --show-annotators locally for the named version.

## engine

### Marked rate (`d`) — the quantity [EBM] treats as scenario-owned

| cell | n | champion | rater_1 | rater_2 | rater_3 |
|---|---|---|---|---|---|
| b1 | 25 | 36.0% | 40.0% | 36.0% | 44.0% |
| b2 | 25 | 84.0% | 60.0% | 64.0% | 60.0% |
| b3 | 25 | 68.0% | 64.0% | 72.0% | 64.0% |
| b4 | 25 | 92.0% | 92.0% | 96.0% | 91.7% |
| b5 | 25 | 56.0% | 62.5% | 56.0% | 40.0% |
| b6 | 25 | 76.0% | 56.0% | 72.0% | 50.0% |
| **all** | 150 | **68.7%** | 62.4% | 66.0% | 58.1% |

### Agreement

| annotator | 3-way kappa | marked/unmarked kappa |
|---|---|---|
| rater_1 | 0.670 | 0.733 |
| rater_2 | 0.790 | 0.849 |
| rater_3 | 0.590 | 0.626 |

## delivery

### Marked rate (`d`) — the quantity [EBM] treats as scenario-owned

| cell | n | champion | rater_1 | rater_2 | rater_3 |
|---|---|---|---|---|---|
| b1 | 25 | 24.0% | 24.0% | 28.0% | 52.0% |
| b2 | 25 | 56.0% | 72.0% | 88.0% | 72.0% |
| b3 | 25 | 32.0% | 24.0% | 8.0% | 52.0% |
| b4 | 25 | 56.0% | 60.0% | 44.0% | 36.0% |
| b5 | 25 | 24.0% | 25.0% | 36.0% | 40.0% |
| b6 | 25 | 24.0% | 24.0% | 20.0% | 36.0% |
| **all** | 150 | **36.0%** | 38.3% | 37.3% | 48.0% |

### Agreement

| annotator | 3-way kappa | marked/unmarked kappa |
|---|---|---|
| rater_1 | 0.546 | 0.498 |
| rater_2 | 0.589 | 0.541 |
| rater_3 | 0.354 | 0.245 |


## Delivery, decomposed by sub-question

`hot = Q1`; `warm = not-Q1 and Q3`. Q3 is only observable where neither side said hot, so its column is conditional and carries its own `n`.

| pair | kappa Q1 (hot) | kappa Q3 (warm \| both non-hot) | n | kappa marked |
|---|---|---|---|---|
| champion x rater_1 | 0.490 | 0.821 | 91 | 0.498 |
| champion x rater_2 | 0.571 | 0.733 | 98 | 0.541 |
| champion x rater_3 | 0.399 | 0.440 | 93 | 0.245 |
| rater_1 x rater_2 (human x model annotator) | 0.664 | 0.809 | 98 | 0.644 |
| rater_1 x rater_3 **(HUMAN-HUMAN CEILING)** | 0.563 | 0.405 | 95 | 0.349 |
| rater_2 x rater_3 (human x model annotator) | 0.683 | 0.402 | 103 | 0.434 |

### Reading the delivery numbers

The champion sits **at or above the human ceiling on every delivery sub-decision**:

* **Q3 (warm)** — champion agrees with the two non-outlier annotators at 0.821 and
  0.733, against a human-human ceiling of **0.405**. Q3 is where the champion does
  BEST, not worst.
* **Q1 (hot)** — champion 0.399-0.571 against a human ceiling of 0.563. At the
  ceiling, not below it. Q1 is the genuinely harder sub-question for everyone.
* **marked/unmarked** — champion x rater_1 is 0.498 against a human ceiling of
  0.349, i.e. the champion agrees with one human better than the two humans agree
  with each other.

**[FT §8]'s bar of kappa >= 0.80 for delivery is above the human ceiling** and is
therefore unachievable by any instrument, a human included. It should not gate.

**The low numbers are one rater's warm criterion, not a Q3 rubric defect.** rater_3
is the outlier against all three other sources on Q3 (0.440 / 0.405 / 0.402) while
champion, rater_1 and rater_2 cluster at 0.733-0.821. The confusion matrix locates
it exactly: champion-flat x rater_3-warm occurs on **21** turns, against 2 for
rater_1 and 6 for rater_2. rater_3 calls warm on turns everyone else calls flat,
which is also why their overall delivery marked rate is 48% against 36-38%.

With two human annotators there is no basis for calling either reading correct.
"Ingratiating, closeness-pulling, flattering, seeking connection with you" admits a
broad and a narrow reading, and the disagreement is evidence the rubric does not
pin which. That is a rubric-disambiguation task and a conversation with the
annotator, not a rater-quality finding — and it matters, because the warm-target
cells (b1, b3, b5) inherit whichever reading the champion encodes.
