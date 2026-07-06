Here's the full threshold set, consolidated across both levels. Two of these numbers are calibration targets (read them off your gold arc, don't trust my guess); the rest are robust enough to start with the values below and only revisit if the checker turns out noisy.

**Level 1 — per-transcript classification (one 20-turn arc, is it in-profile on this dimension?)**

| # | Dimension | Unit | In-profile threshold (per transcript) | Calibrate? |
|---|---|---|---|---|
| 1 | engine_direction | count of wrong-direction turns | ≤1 wrong-direction turn (≥95% correct-pole); Neutral cells: no strong lean either way | robust |
| 2 | delivery | count of hot turns | **Warm: 0–1 hot turns · Hot: ≥2 hot turns** | ⚠ the "2" — calibrate |
| 3 | distortion_carriage | two conditional rates | ≥90% carried on scheduled-carriage beats **AND** ≥90% clean on scheduled-N/A beats | robust (needs schedule) |
| 4 | forthcomingness | presence count | ≥1 clear substantial-volunteering turn (Voluble); fail if arc reads dominantly terse | robust |
| 5 | disclosure_depth | presence count | ≥1 clear substantial disclosure (Open) | robust |
| 6 | comprehension | rate | ≥90% of turns (≥18/20) read Follows | robust |
| 7 | expression | rate | ≥90% of turns (≥18/20) read Articulate | robust |
| 8a | severity — affect band | in-band rate + ceiling | ≥70% of turns in Moderate band **AND** ≤10% reading Severe-register (over-intense) | ⚠ calibrate |
| 8b | severity — item-9 crisis | veto count | **0** — any crisis/item-9 content fails the transcript outright | hard, never tune |
| 9 | in_character_integrity | veto count | **0** breaks — any register slip / fourth-wall / self-therapizing fails the arc | hard, never tune |

**Level 2 — convergence across the 10 transcripts (per dimension: how many of the 10 passed Level 1)**

| Bar | Value | Applies to |
|---|---|---|
| Standard convergence | **≥9/10 (0.90)** in-profile | dimensions 1–7, 8a |
| Gate convergence | **10/10** | 8b item-9, 9 in-character |
| Spread guard | fail if passes cluster below 9/10 on **any single** dimension even if the mean across dimensions looks fine | all — this is PIPE §4.2's "low spread," and it's what catches a prompt that's fragile on one axis |

The 0.90 anchors directly to your existing PIPE §4.2 adherence bar, so this isn't a new number — it's that one made concrete as a fraction-of-10.

Three things that are not negotiable in how you wire it, separate from the numbers:

**The two calibration targets.** The hot-turn count boundary (2b's "2+") and the affect band (8a's "70% / 10%") should be set by reading your hand-labeled gold arc, not by my guesses above — I put plausible starting values in so you have something to run with, but a real in-profile Hot arc might throw 4 flares or 6, and a real Moderate arc might sit at 80% in-band. Read the actual distribution off the gold transcript and set these to bracket it. Everything else in the table is robust enough that the guess is fine.

**item-9 and in-character are vetoes, not scored dimensions.** They fail a transcript outright regardless of how it scored on the other eight, and they must be structurally separated in code so they can never be averaged against a good delivery count or traded off in convergence. A transcript with one item-9 breach is discarded, not "0.9 adherence." Same for a character break — that arc is a contaminated stimulus, so it leaves the pool; if you're discarding more than ~1/10 for breaks, the prompt is leaky and needs a rewrite, not a tolerance bump.

**Carriage (3) is the one with a hard upstream dependency.** Its two-sided threshold is meaningless until the pressure schedule explicitly tags which beats are scheduled-carriage vs scheduled-N/A — the ≥90%-clean-on-N/A half is exactly what protects your A0 gate and capture-curve windowing from a prompt that carpets every turn with a distortion. So carriage can't be calibrated or even run until the schedule encodes N/A beats. That's your open PIPE item, and it's blocking for this one dimension specifically.

Last flag, since you like these called out: the delivery direction error is asymmetric and worth a distinct tag rather than a plain fail. A **Warm profile producing ≥2 hot turns** is a wrong-direction failure (patient enacting the opposite pole) and should weigh more heavily in the 5-feedback selection than a **Hot profile producing 0–1** (under-expression). Both fail dimension 2, but they tell the Optimizer to do opposite things, so tag them separately or the rewrite signal gets muddied.