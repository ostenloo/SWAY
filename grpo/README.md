# SWAY GRPO Simulator finetune

Implementation of `specs/grpo_spec.md` — a **build-time**, weight-level GRPO
finetune of the Simulator (LLM-2) that replaces the prompt-optimization Optimizer
for cells where prompt conditioning can't reproduce personality adherence. The
finetuned Simulator is a new build-time artifact; run-time (gate, Judge, drift
scoring) is unchanged.

The code reuses the existing harness (`sway_harness/`: `client`, `parser`,
`fidelity`, `build`) and `tools/compute_kappa.py` via `grpo/_bootstrap.py`, which
puts both directories on `sys.path` (they use flat imports).

## One human step

§10 certification is the only place an adapter meets human judgment. It emits a
**blind** labelling sheet (model reads live in a separate key file) and ingests it
back, per finetune iteration, and `freeze_adapter` refuses an uncertified adapter
(C7).

Both C6 pre-flight gates were removed by researcher decision — see deviations.

## Layout → spec deliverables (§13)

| File | Spec | What it is |
|---|---|---|
| `reward/fidelity_reward.py` | §4, C1, C8 | The pure reward: partial credit over the two diagnostic binaries (no realism floor — see deviations). Import-clean (C1) + the C8 family-disjointness assertion. |
| `reward/delivery_decompose.py` | §8.1 | Q1 (hostility toward listener) / Q2 (grievance toward absent party), `hot = Q1` regardless of Q2. |
| `reward/backends.py` | §4, D0.2 | **Two distinct local champions** (engine, delivery-decomposed). No third grader, no Opus in-loop path. Folded/clean stubs for the gate tests. |
| `reward/turn_fidelity.py` | §4 | Per-turn label → binaries, reusing `fidelity.py`'s target poles. |
| `reward/trl_adapter.py` | §7 | Wraps `fidelity_reward` as a TRL `GRPOTrainer` reward function; feeds Q1/Q2 to the monitor. |
| `data/rollout.py` | §5 | State construction, per-turn grouping, bare interlocutor rollout. **Trains on one** interlocutor — a documented deviation from §5.3, see below. |
| `data/curriculum.py` | §5.5 | Near-manifold → target annealing as sequential training stages. The *target* anneals; the graders never do. |
| `train/rft_warmstart.py` | §6 | Reward-filtered SFT; also emits the kept turns raw. |
| `train/grpo_loop.py` | §7 | GRPO loop (TRL + vLLM, reference = disabled adapter). No pre-flight gates. |
| `gates/authored_pairs_smoketest.py` | §8.3 | Advisory smoke test on hand-authored pairs. **Not** a gate. |
| `monitor/online_audit.py` | §9 | High-advantage audit (split Q1/Q2), reward↔fidelity gap, grievance→hot watch, group collapse — plus the TrainerCallback that fires them. |
| `cert/certify_and_freeze.py`, `cert/rubric_frozen.md` | §10 | Human per-iteration certification: frozen rubric, fixed gold subset, per-iteration κ, freeze manifest with both champion identities. |
| `analysis/reward_shapes.py` | §4 | Compares candidate reward shapes on real sampled groups. No training, no GPU. |
| `stats.py` | §0.1, §10 | Thin reuse of `tools/compute_kappa.py` (κ, Gwet's AC1, bootstrap CI). |
| `configs/grpo.yaml`, `config.py` | §11 | Config surface + loader. |
| `tests/` | A1, A2 | C1 guard, reward shape, C8 assertion, decomposition, both gates blocking. |

## Two decisions (spec §0)

- **D0.1 base checkpoint** — `base_model: Qwen/Qwen2.5-14B-Instruct` (the earlier
  "llama2.5:14b" is not a real model). C8 is therefore anchored to Qwen: **both
  champions must be non-Qwen**, asserted at reward construction.
- **D0.2 reward backend** — **no external API spend.** The reward is two local
  axis-specialists: `command-r7b` (engine) + `glm4:9b` (delivery, decomposed).
  There is no realism backend (see deviations). Because engine and delivery are
  answered by *different* models, their blind spots are axis-local and uncorrelated (R4 downgraded). A
  single distilled both-axis checker is **shelved** and is not constructible from
  the config. There is no Opus reward path: a chat-window Opus cannot sit in the
  gradient loop. Chat-Opus survives only as the offline §10 cross-check.

## Known deviations from the spec

Recorded here rather than buried in code comments, because a reader comparing this
tree to `specs/grpo_spec.md` should not have to discover them.

- **§5.3 cross-interlocutor spread — trains on ONE interlocutor, not ≥2.**
  Researcher decision. The single partner is `huihui_ai/qwen2.5-abliterate:14b`,
  the same `reference_interlocutor` the prompt-opt builds used
  (`sway_harness/config.json`), so warm-start rolls on the distribution the
  existing artifacts came from and the A3 control stays like-for-like. It also
  frees ~9 GB on a card where §12's budget is the binding constraint. **The
  consequence:** the interlocutor-robustness claim now rests entirely on §10 step
  2 (certification against an unseen bare model), which is therefore doing more
  work than the spec assumed — do not skip it, and consider varying it across
  iterations so repeated model selection doesn't erode its held-out status.
- **"Bare" is near-zero, not zero.** Interlocutors use the harness's
  `REF_SYSTEM_PROMPT` ("...do not therapize, advise, or take sides") rather than
  an empty system prompt, again to keep GRPO and prompt-opt comparable.
- **Both C6 pre-flight gates — REMOVED.** Researcher decision. §8.2's stratified
  delivery validation (it oversampled specific contested cases) and §0.1's
  non-convergence diagnostic are both gone, along with the assertions that read
  them. **Consequence:** nothing checks before training that GRPO is the right
  tool for this non-convergence — §0.1 existed to fork *policy capability
  ceiling* from *miscalibrated checker*, and those are indistinguishable from
  convergence data alone. If the checker is what's wrong, the loop optimizes
  confidently toward a wrong target and the curve still goes up. Nothing
  validates the delivery champion against human labels either, so §8.1's
  `hot = Q1` accuracy is an assumption. The §9 audit during training and §10
  certification after are the only remaining places either failure would
  surface. The §8.3 authored-pair smoke test stays available and advisory.
- **No realism floor at all.** §4 specifies a multiplicative realism floor;
  it has been **removed** by researcher decision. The reward is the two diagnostic
  binaries alone. **Consequence:** R3 (degenerate-but-on-profile collapse — the
  documented RL-sim failure where a finetuned simulator drifts into on-target but
  degenerate text) is now mitigated by the KL penalty alone. The §9 high-advantage
  audit remains the observability path; degenerate turns earning advantage will
  surface there, but nothing stops them being reinforced in the meantime.
- **Measured while removing it, and worth keeping on file.** Probing four local
  annotators on the realism dimensions:

  | case | command-r7b | gemma2 | llama3.1 | glm4 |
  |---|---|---|---|---|
  | therapist register | miss | catch | catch | miss |
  | "As an AI language model…" | miss | **miss** | **miss** | **miss** |
  | self-therapizing | miss | miss | miss | miss |
  | crisis / suicidal ideation | **miss** | catch | catch | catch |
  | legit furious / rambling | clear | clear | clear | 1 false positive |

  **Every** model scores "As an AI language model I am simulating a patient" as
  clean, and all four miss self-therapizing. That is a hole in the shared annotator
  prompt rather than a model-choice problem — and since `fidelity.py` uses
  `in_character_break` as a hard veto for arc classification, **it affects the
  prompt-opt pipeline too, not just GRPO.** Worth chasing independently of this
  finetune.

## Non-negotiable constraints (§2) — where they live

- **C1** reward reads only `(patient_turn, P, context, cell)` — `fidelity_reward.py`
  imports nothing drift-side; `tests/test_c1_import_guard.py` greps + unit-proves it.
- **C3** partial-credit diagnostics, no derived 0-3 — `fidelity_reward.py`. §4's
  multiplicative realism floor is removed (see deviations).
- **C4** graders temp-0 / frozen — `_LocalCore._annotate`. The curriculum anneals
  the *target*, never the graders.
- **C6** REMOVED (both gates) — `run_grpo` starts as soon as it is called.
- **C7** certify before freeze — `freeze_adapter` refuses a non-certified adapter.
- **C8** reward family-disjoint from the base — `assert_family_disjoint`, called
  from `build_champion_backends`; an unrecognised model tag is a failure, not a pass.

## Running

Gates, reward, monitor, cert scoring and the tests need no GPU:

```bash
pip install -r grpo/requirements-grpo.txt      # (uncomment the training stack on the GPU host)
python -m pytest grpo/tests/ -q
```

The pipeline, in the order the spec requires — note §8.2 runs **after** warm-start,
because the RFT-filtered set is its validation distribution:

```bash
python -m grpo.run smoketest                   # optional §8.3, advisory only

# Optional: which reward shape? Every candidate is a pure function of the same
# three binaries, so one scoring pass settles all of them. No training, no GPU.
python -m grpo.run reward-sweep --cells b1 b2 --states 6 --group-size 8

# §6 warm-start
python -m grpo.run warmstart

# §7
python -m grpo.run grpo

# §10 — per iteration, against the frozen rubric
python -m grpo.run cert-build  --iteration 1   # serve the CANDIDATE at policy.base_url
#   ... hand-label against grpo/cert/rubric_frozen.md ...
python -m grpo.run cert-score  --iteration 1
python -m grpo.run cert-freeze --iteration 1 --adapter results/grpo/adapters/grpo
```

There is no `all` subcommand: certification waits on a human.

## Acceptance (§13)

- **A1** — `test_c1_import_guard.py` (no drift-side imports; reward reads only the
  four inputs) + `test_c8_family_disjoint.py` (C8 holds for the configured base,
  rejects same-family and unknown models).
- **A2** — `test_delivery_decomposition.py`: `hot = Q1` under a yes/yes stub,
  not-hot under a grievance-only stub; the §8.3 smoke test separates a folded
  delivery backend from a clean one. The §8.2 half of A2 no longer applies.
- **A6** — no longer applies: both C6 gates were removed.
- **A3–A5** — GPU-host runs: GRPO lifts held-out fidelity vs the prompt-opt control
  without §9 divergence or a rising grievance→hot watch; each iteration's adapter
  passes §10 and freezes at deployment quant; `freeze_manifest.json` records base +
  quant + **both** champion identities + seed + that iteration's certification κ.
