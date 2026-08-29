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


---

## Context sensitivity — measured separately, and the reason this report exists

The champion is **conditioned differently on the calibration side and the reward
side**, and on the engine axis that difference is larger than everything else here.

* AnnoMI calibration (`annomi_calibrate.collect_turns`) and the batch03 hand-label
  sheets both hand the grader **the interlocutor's preceding turn** as context.
* The rate-profile reward (`band_reward.context_upto`) hands it **the patient's own
  preceding turns**, because the reward may not read interlocutor replies.

Same champion, same transcripts (`results/build_artifacts/{b1,b3}/iter_0/`), engine
marked rate `d`:

| arc | patient's-own-turns context | prev-interlocutor context |
|---|---|---|
| b1/transcript_0 | 100% | 15% |
| b1/transcript_1 | 95% | 45% |
| b1/transcript_2 | 100% | 20% |
| b3/transcript_0 | 100% | 40% |
| b3/transcript_1 | 100% | 55% |
| b3/transcript_2 | 100% | 75% |
| **mean** | **99.2%** | **41.7%** |

Delivery is essentially unaffected (85/85, 60/65, 70/60, 50/50, 85/85, 20/20). The
effect is engine-specific and the mechanism is plausible: engine asks whether the
turn attributes cause, and nineteen consecutive blame-laden patient turns as
"context" prime it toward yes. Delivery asks whether anything is pointed *at the
listener*, which anchors on the grader-as-listener regardless of what precedes.

### Consequences

1. **A reported measurement was wrong.** Engine `d = 1.00` on simulator arcs — and
   the "the policy is parked at the caricature" reading built on it — is an artifact
   of the reward's context construction. Under the conditioning the champion was
   calibrated on, engine `d` is ~42% against AnnoMI's 26.5%: a 1.6x discrepancy,
   not 4x. The per-cell table above confirms it independently — champion and raters
   both put b1 near 36-44% on the very same transcripts.

2. **The shipped rate-profile reward is mis-conditioned.** Its bands were derived
   from AnnoMI under prev-interlocutor context and its rollouts are scored under
   patient-own-turn context, so the bias-cancellation argument (RATE C3) does not
   hold: cancellation needs the SAME bias on both sides, and these are two different
   biases. RATE C7 pinned the estimator and left the conditioning unpinned.

3. **[EBM] §6 inherits this.** Reachability measures `p̂` and `d̂` with the same
   champions, so the gate must run under whichever context convention the controller
   uses at run time, and that convention has to be pinned in config. [EBM] §4's
   signature passes `history` as patient + interlocutor turns, so the controller CAN
   match the calibration where the reward could not.

The batch03 half of this report reproduces with `tools/champion_validation.py`
(pure CSV, no model calls). The context half needs the champions served and is
recorded here rather than re-run on every invocation.
