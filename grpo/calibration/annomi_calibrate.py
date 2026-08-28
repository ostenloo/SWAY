#!/usr/bin/env python3
"""AnnoMI offline calibration — produces the frozen reward artifact.

This stage MUST run and be frozen before the first GRPO step (C9). It runs once
per grader-backend version.

**Two derivations live here, and only one of them is current.**

`rate-derive` implements SWAY_ENGINE_DELIVERY_RATE_PROFILE_SPEC: per-conversation
RATES over all `T` turns, grouped by lean, edges at the 25th/75th percentiles.
This is the shape the reward uses.

`derive` implements the superseded conditional-ratio band (grpo_spec_2 §6.5):
per-session `q` among MARKED turns, EB-shrunk, edges at the 72.5th/92.5th. It is
retained ONLY so the measurements in [RATE §1] stay reproducible.
`band_calibration.v1.yaml` is defective on both engine directions and on
delivery, `trl_adapter.build_reward_func` refuses to build against it, and it
must never be used as a target.

Stages, each a subcommand so the expensive one is resumable:

  grade           label every substantive AnnoMI client turn with BOTH frozen
                  champions, appending to a resumable JSONL cache (~3.2k turns
                  x 2 axes). Shared by both derivations.
  rate-derive     cache -> per-conversation rates -> lean groups -> percentiles
                  -> `rate_calibration.<grader_version>.yaml` + a disclosure
                  report. THE CURRENT PATH.
  rate-freeze-v1  the same artifact built from [RATE §7]'s already-measured
                  percentiles, for a host without the label cache. Same shaping
                  code; the artifact records which source it used.
  derive          SUPERSEDED — the conditional-ratio band. See above.
  kappa           ingest the hand-label sheets and report human-vs-grader
                  agreement. REPORT-ONLY: C6 removed the gate that would have
                  thresholded it, and the edges come from the grader pass, not
                  the human labels.
  all             grade -> derive -> kappa (the superseded chain; `all` is left
                  pointing at it so an old command line does not silently emit a
                  different artifact than it used to)

**Why the two sample sizes differ.** Per [BAND CB1] the bounds come from GRADER
labels, which decouples them: humans label ~500 turns (22 conversations) for the
kappa; the frozen grader labels every substantive client turn for the bracket.
Grader turns are near-free (temp 0, local, unattended), so the edges get the full
corpus.

**C1 note.** §6.5 warns that `sway_harness/validate_judge.py` — which has AnnoMI
loading and kappa machinery — is built for **Judge**/MITI *therapist* validation,
i.e. the drift side. Importing it here would trip the C1 grep guard. This module
therefore reuses the *label-task generator's* loader (no judge paths) and the
`tools/compute_kappa.py` statistics, and never touches validate_judge.

**CB2.** The corpus is human data, always. Never the Simulator's own authored or
rollout material — that is the calibration-circularity ratchet (R7).

Usage:
  python -m grpo.calibration.annomi_calibrate grade --workers 8        # resumable
  python -m grpo.calibration.annomi_calibrate rate-derive              # current
  python -m grpo.calibration.annomi_calibrate rate-freeze-v1           # no cache
  python -m grpo.calibration.annomi_calibrate kappa
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import grpo._bootstrap  # noqa: F401
from grpo.calibration import derive as D
from grpo.calibration import rate_derive as RD
from grpo.config import load_config
from grpo.reward.band_reward import MARKED, calibration_from_dict, load_calibration
from grpo.reward.turn_fidelity import poles_for_cell

REPO = Path(__file__).resolve().parents[2]
GENERATOR_DIR = REPO / "label_tasks" / "annomi"
DEFAULT_CACHE = REPO / "results" / "grpo" / "calibration" / "annomi_grader_labels.jsonl"
DEFAULT_OUT = REPO / "grpo" / "calibration"

#: The graders are BLIND to the target pole (that is what stops them
#: rubber-stamping toward a profile), so `cell` is not read by either decomposition
#: prompt. A sentinel makes that explicit and keeps the cache key honest.
BLIND_CELL = "annomi"


def _generator():
    """The label-task generator, imported as a flat module (repo convention)."""
    if str(GENERATOR_DIR) not in sys.path:
        sys.path.insert(0, str(GENERATOR_DIR))
    import generate_annomi_label_task as g  # noqa: E402
    return g


# ── corpus ───────────────────────────────────────────────────────────────────

def collect_turns(include_backchannels: bool = False) -> list[dict]:
    """Every client turn to be graded, with the context the grader will see.

    **Backchannels are excluded by default, and this is a real choice.** §6.5
    reads two ways: it says "the frozen grader labels all 4,817 client turns",
    and it also documents a backchannel filter that leaves 3,221 substantive
    turns. Grading the backchannels would put ~1,596 near-guaranteed
    neutral/flat turns into the DENOMINATOR of `d_anno`, deflating it — and
    `d_anno` is the anchor for `d_floor`, which the rollout `d` is measured
    against. Rollout arcs contain no backchannels, so including them here would
    compare unlike quantities. The substantive set also keeps the grader and the
    human raters on the same turns, which is what the kappa needs.

    `--include-backchannels` runs it the other way; both counts are reported.
    """
    g = _generator()
    by_session = g.load_sessions()

    turns = []
    for sid in sorted(by_session, key=lambda t: int(t)):
        prev_therapist = ""
        for u in by_session[sid]:
            text = (u["utterance_text"] or "").strip()
            if u["interlocutor"] == "therapist":
                prev_therapist = text
                continue
            if not include_backchannels and not g.is_substantive(text):
                continue
            turns.append({
                "turn_id": g.turn_id(sid, u["utterance_id"]),
                "session_id": sid,
                "utterance_id": u["utterance_id"],
                "mi_quality": u["mi_quality"],
                # §6.5: delivery is defined as directed "toward the listener", so
                # the grader is handed the therapist's preceding turn — the same
                # context the human raters get on the label sheet.
                "context": prev_therapist,
                "text": text,
            })
    return turns


def corpus_stats(include_backchannels: bool = False) -> dict:
    g = _generator()
    by_session = g.load_sessions()
    all_client = [u for uts in by_session.values() for u in uts if u["interlocutor"] == "client"]
    substantive = [u for u in all_client if g.is_substantive((u["utterance_text"] or "").strip())]
    return {
        "n_sessions": len(by_session),
        "n_client_turns": len(all_client),
        "n_substantive": len(substantive),
        "n_dropped_backchannel": len(all_client) - len(substantive),
        "graded_set": "all_client" if include_backchannels else "substantive",
    }


# ── stage 1: the grader pass (resumable) ─────────────────────────────────────

def load_cache(path: Path) -> dict:
    """(turn_id, axis) -> label, from a partially-complete run."""
    done = {}
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn final line from a killed run
            done[(rec["turn_id"], rec["axis"])] = rec["label"]
    return done


def grade(args) -> Path:
    """Label every turn on both axes with the frozen champions (C4)."""
    from grpo.reward.backends import backend_identities

    cfg = load_config(args.config)
    backends = _build_backends(cfg)
    ids = backend_identities(backends)

    turns = collect_turns(args.include_backchannels)
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_cache(cache_path)

    jobs = [(t, axis) for t in turns for axis in ("engine", "delivery")
            if (t["turn_id"], axis) not in done]

    stats = corpus_stats(args.include_backchannels)
    print(f"corpus: {stats}")
    print(f"grading {len(jobs)} of {len(turns) * 2} (turn, axis) pairs "
          f"— {len(done)} already cached")
    print(f"graders: {ids}")
    if args.limit:
        jobs = jobs[:args.limit]
        print(f"--limit: capped to {len(jobs)} jobs")

    adapters = {"engine": backends.engine, "delivery": backends.delivery}

    def run(job):
        t, axis = job
        label = adapters[axis].label(t["text"], t["context"], BLIND_CELL)
        return {
            "turn_id": t["turn_id"], "session_id": t["session_id"], "axis": axis,
            "label": label, "mi_quality": t["mi_quality"],
            "grader": ids[axis],
        }

    written = 0
    with cache_path.open("a") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(run, jobs):
                out.write(json.dumps(rec) + "\n")
                written += 1
                if written % 200 == 0:
                    out.flush()
                    print(f"  {written}/{len(jobs)}", flush=True)

    # §4.2's missing-key counter, over the whole pass. A high rate here means the
    # bracket itself is built on silently-unmarked turns.
    from grpo.reward.backends import missing_key_rates
    print(f"missing-key rates: {missing_key_rates(backends)}")
    print(f"wrote {written} labels -> {cache_path}")
    return cache_path


def _build_backends(cfg):
    from grpo.reward.backends import build_champion_backends
    r = cfg["reward"]
    return build_champion_backends(
        engine_model=r["backend_engine"],
        delivery_model=r["backend_delivery"],
        base_url=r["base_url"],
        base_model=cfg["base_model"],
    )


# ── stage 2: derive the bracket and emit the artifact ────────────────────────

def labels_by_session(cache: dict, turns: list[dict], axis: str) -> dict:
    """session_id -> [label, ...] for one axis, over turns actually graded."""
    out = defaultdict(list)
    for t in turns:
        lab = cache.get((t["turn_id"], axis))
        if lab is not None:
            out[t["session_id"]].append(lab)
    return dict(out)


def derive_artifact(args) -> dict:
    """The §6.5 derivation, assembled into the [BAND §6.4] document."""
    cfg = load_config(args.config)
    turns = collect_turns(args.include_backchannels)
    cache = load_cache(Path(args.cache))
    if not cache:
        raise SystemExit(
            f"no grader labels at {args.cache} — run the `grade` stage first "
            "(the bracket comes from grader space, CB1, not from hand labels)"
        )

    brackets, per_axis = {}, {}
    for axis in ("engine", "delivery"):
        lbs = labels_by_session(cache, turns, axis)
        per_axis[axis] = lbs
        for direction in MARKED[axis]:
            brackets[(axis, direction)] = D.derive_bracket(
                axis, direction, lbs,
                P_lo=args.p_lo, P_hi=args.p_hi, min_marked=args.min_marked)

    # `d_floor` is a SOFT anchor, not a measured target ([BAND §5]): density is
    # the most scenario-dependent quantity in the factorization and AnnoMI is
    # off-scenario. Anchor it at the median per-session marked fraction and scale
    # it down — the simulator should clear a floor that most real patients clear,
    # not match the median exactly.
    d_floor = {}
    for axis in ("engine", "delivery"):
        any_dir = next(iter(MARKED[axis]))
        med = brackets[(axis, any_dir)].d_anno_median_session
        d_floor[axis] = round(max(args.min_d_floor, args.d_floor_frac * med), 4)

    cells = {}
    for cell in args.cells:
        poles = poles_for_cell(cell)
        eng_target = poles["engine_direction"]
        del_target = poles["delivery"]
        entry = {}

        if eng_target == "neutral":
            # D2.3 — neutral engine means ABSENT engine.
            low_d = D.percentile(
                sorted(c.d for c in D.session_counts("engine", per_axis["engine"])),
                args.neutral_d_percentile) if per_axis["engine"] else 0.10
            entry["engine"] = D.density_low_params(low_d, d_floor["engine"], d_lo=args.d_lo)
            entry["engine"]["provenance"] = {
                "derivation": "D2.3 density_low (neutral engine = ABSENT engine)",
                "annomi_d_eng_percentile": args.neutral_d_percentile,
                "annomi_d_eng_at_percentile": round(low_d, 4),
                "engine_d_floor_reference": d_floor["engine"],
                "note": "d_lo > 0 strictly: at d_lo = 0 an arc with no engine "
                        "expression scores 1.0 (the inert-simulator exploit).",
            }
            entry["engine"]["bracket_informative"] = True
        else:
            entry["engine"] = _q_band_entry(
                brackets[("engine", eng_target)], eng_target, d_floor["engine"], args)

        entry["delivery"] = _q_band_entry(
            brackets[("delivery", del_target)], del_target, d_floor["delivery"], args)
        cells[cell] = entry

    from grpo.reward.backends import backend_identities
    doc = {
        "grader_version": args.grader_version,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": corpus_stats(args.include_backchannels),
        "backend_identities": cache_identities(Path(args.cache)) or _cfg_identities(cfg),
        "derivation": {
            "spec": "grpo_spec_2 §6.5 / BAND §6",
            "unit": "per-session (per-patient) q, empirical-Bayes shrunk, percentiles",
            "P_lo": args.p_lo,
            "P_hi": args.p_hi,
            "min_marked_per_session": args.min_marked,
            "alpha": args.alpha,
            "d_floor_anchor": d_floor,
            "d_floor_rule": f"max({args.min_d_floor}, {args.d_floor_frac} x median per-session d)",
            "claim": "NOT a bound — a chosen position inside a real distribution: the "
                     "simulated patient should be at least as directional as the P_lo-th "
                     "percentile real patient of that direction.",
            "transfer_caveat": "AnnoMI is MI counselling, not layoff support. The "
                               "conditional q is more transferable than the density d; "
                               "d_floor is operational, not norm-referenced.",
        },
        "cells": cells,
    }

    # Validate through the real loader before writing: an artifact that cannot be
    # loaded is not a calibration, and finding that out at step 1 is too late.
    calibration_from_dict(doc)

    out_path = Path(args.out or (DEFAULT_OUT / f"band_calibration.{args.grader_version}.yaml"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))

    cal = load_calibration(out_path)          # re-load to get the C9 hash
    report = _report(doc, brackets, cal.sha256)
    report_path = out_path.with_suffix(".report.md")
    report_path.write_text(report)

    print(report)
    print(f"\nartifact -> {out_path}")
    print(f"report   -> {report_path}")
    print(f"C9 sha256: {cal.sha256}")
    return doc


def _q_band_entry(b: D.DirectionBracket, c_star: str, d_floor: float, args) -> dict:
    L, U, notes = D.clamp_edges(b.L_design, b.U)
    prov = b.to_provenance()
    if notes:
        prov["clamps"] = list(notes)
    return {
        "mode": "q_band",
        "c_star": c_star,
        "L_design": round(L, 4),
        "U": round(U, 4),
        "s_lo": args.s_lo,      # tight: the ordering must hold
        "s_hi": args.s_hi,      # sets how hard caricature is punished
        "alpha": args.alpha,
        "d_floor": d_floor,
        "bracket_informative": bool(b.bracket_informative and not notes),
        "provenance": prov,
    }


def cache_identities(path: Path) -> dict:
    """Grader identities as recorded DURING the pass — the CB1 stamp.

    Read from the cache, not the live config, so the artifact records who
    actually measured the bracket rather than who the config currently names.
    A config edited between `grade` and `derive` must not be able to relabel
    someone else's measurements as the current champion's — that would defeat
    `assert_calibration_backends` at exactly the point it matters.

    Raises if a single axis was graded by more than one model: a bracket measured
    half by one instrument and half by another has no cancellation guarantee
    ([BAND §7]) and must be re-graded, not averaged.
    """
    seen = defaultdict(set)
    if not path.exists():
        return {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("grader"):
                seen[rec["axis"]].add(rec["grader"])
    out = {}
    for axis, models in seen.items():
        if len(models) > 1:
            raise SystemExit(
                f"CB1: the {axis} axis in {path} was graded by MORE THAN ONE model "
                f"({sorted(models)}). Instrument cancellation does not hold across a "
                "grader swap — delete the cache and re-grade with one frozen champion."
            )
        out[axis] = next(iter(models))
    return out


def _cfg_identities(cfg) -> dict:
    from grpo.config import reward_identities
    return reward_identities(cfg)


def _report(doc: dict, brackets: dict, sha: str) -> str:
    lines = [
        "# AnnoMI band calibration — disclosure report",
        "",
        f"grader_version: `{doc['grader_version']}`  ",
        f"generated: {doc['generated_utc']}  ",
        f"C9 sha256: `{sha}`  ",
        f"backends: `{doc['backend_identities']}`",
        "",
        f"corpus: {doc['corpus']}",
        "",
        "## Derived brackets (per-session q, EB-shrunk, percentiles)",
        "",
        "| axis | direction | L_design | U | eligible | mu | tau2 | d_anno | informative |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (axis, direction), b in sorted(brackets.items()):
        L = "n/a" if b.L_design != b.L_design else f"{b.L_design:.3f}"
        U = "n/a" if b.U != b.U else f"{b.U:.3f}"
        lines.append(
            f"| {axis} | {direction} | {L} | {U} | {b.n_eligible_sessions}/"
            f"{b.n_sessions_total} | {b.mu:.3f} | {b.tau2:.5f} | "
            f"{b.d_anno_pooled:.3f} | {'yes' if b.bracket_informative else '**NO**'} |"
        )
    lines += ["", "### Uninformative brackets (D2.4 — warn and disclose, do not halt)", ""]
    flagged = [(k, b) for k, b in sorted(brackets.items()) if not b.bracket_informative]
    if not flagged:
        lines.append("None.")
    for (axis, direction), b in flagged:
        lines.append(f"- **{axis}/{direction}**: " + "; ".join(b.informative_reasons))
    lines += [
        "",
        "§6.5 expects this to fire on **delivery** — its per-session percentiles rest on "
        "a small eligible sample (11-33 sessions), where engine is comfortable (63-103).",
        "",
        "## Claim",
        "",
        doc["derivation"]["claim"],
        "",
        "## Transfer caveat",
        "",
        doc["derivation"]["transfer_caveat"],
        "",
        "## Per-cell entries",
        "",
        "| cell | engine mode | engine band | delivery band | delivery d_floor |",
        "|---|---|---|---|---|",
    ]
    for cell, e in doc["cells"].items():
        eng = e["engine"]
        if eng["mode"] == "density_low":
            # No q-band here: the engine target is a LOW DENSITY, so quoting a
            # d_floor next to the q-band cells would read as the same quantity.
            eng_desc = (f"absent engine: d in [{eng['d_lo']}, {eng['d_hi']}] "
                        f"(D2.3, d_lo > 0)")
        else:
            eng_desc = (f"{eng['c_star']} q in [{eng['L_design']}, {eng['U']}], "
                        f"d_floor {eng['d_floor']}")
        d = e["delivery"]
        lines.append(
            f"| {cell} | {eng['mode']} | {eng_desc} | "
            f"{d['c_star']} q in [{d['L_design']}, {d['U']}] | {d['d_floor']} |"
        )
    return "\n".join(lines) + "\n"


# ── stage 2b: the RATE-PROFILE derivation ([RATE §6]) ────────────────────────
#
# SUPERSEDES `derive_artifact` above, which builds the conditional-ratio band
# artifact. That artifact (`band_calibration.v1.yaml`) is defective on both engine
# directions and on delivery and MUST NOT be used; the `derive` stage is retained
# only so the §1 measurements remain reproducible.
#
# Two entry points, one document shape:
#
#   rate-derive     percentiles read off the grader label cache — the real path
#   rate-freeze-v1  percentiles transcribed from [RATE §7], for when the cache is
#                   not on this host. Same shaping code, different source; the
#                   artifact says which it was.

def _rate_doc(cfg, args, cells_doc: dict, derivation: dict, backend_identities: dict) -> dict:
    from grpo.reward.rate_profile_reward import calibration_from_dict as rate_from_dict

    doc = {
        "spec": "SWAY_ENGINE_DELIVERY_RATE_PROFILE_SPEC",
        "grader_version": args.grader_version,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # §11 — REQUIRED. The bands are validated against it (§5.2), so an
        # artifact that does not record its T cannot be checked, only assumed.
        "arc_length_T": int(args.arc_length_t),
        "backend_identities": backend_identities,
        "derivation": derivation,
        "cells": cells_doc,
    }
    # Validate through the real loader before writing: an artifact that cannot be
    # loaded is not a calibration, and finding that out at step 1 is too late.
    rate_from_dict(doc)
    return doc


def _write_rate_artifact(doc: dict, out_path: Path, edges_by_cell: dict) -> dict:
    from grpo.reward.rate_profile_reward import load_calibration as load_rate

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    # allow_unicode: the artifact is a disclosure document a human reads, and
    # every provenance string in it cites a spec section by its § sign.
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False,
                                       allow_unicode=True))

    cal = load_rate(out_path)          # re-load to get the C6/C9 hash
    report = _rate_report(doc, edges_by_cell, cal.sha256)
    report_path = out_path.with_suffix(".report.md")
    report_path.write_text(report)

    print(report)
    print(f"\nartifact -> {out_path}")
    print(f"report   -> {report_path}")
    print(f"C9 sha256: {cal.sha256}")
    return doc


def _common_derivation(args) -> dict:
    return {
        "spec": "RATE §6",
        "unit": "per-conversation RATE of each marked label over ALL T turns",
        "estimator": "raw counts / T on BOTH sides; no smoothing anywhere (C7)",
        "eligibility": f"T >= {RD.ELIGIBILITY_MIN_TURNS} turns — a LENGTH rule, never a "
                       "marked-count rule (§6 step 2, the §1.6 fix)",
        "percentile_window": {
            "P_lo": args.rate_p_lo, "P_hi": args.rate_p_hi,
            "declared_choice": "25th-75th targets the middle half of conversations that "
                               "lean the intended way. This is a DESIGN DECISION, not a "
                               "measurement (§6.1); changing it needs its own "
                               "justification, not a retune.",
        },
        "grouping": {
            "low_rate_ceiling": RD.LOW_RATE_CEILING,
            "precedence": "low-rate tested FIRST, then lean by p_a > p_b — §6 step 4's "
                          "three predicates overlap as written; see "
                          "rate_derive.group_conversations for the argument.",
        },
        "anti_caricature": "the on-direction band's UPPER EDGE, and nothing else (§5.3). "
                           "Off-direction bands have no lower edge because the measured "
                           "25th percentile of the off-direction rate is 0.000.",
        "shoulders": f"symmetric, s = {RD.SHOULDER} (§5.1)",
        "scenario_dependence": "EVERY component is scenario-dependent (§10). The "
                               "conditional ratio's scenario-invariance is given up "
                               "deliberately: it was an argument, never a measurement, "
                               "and six measured failures sit against it. AnnoMI is MI "
                               "counselling, not layoff support; the counselling-to-layoff "
                               "gap is a standing limitation on BOTH axes.",
        "precision": "at a true rate of 0.20 the 95% interval on a single 20-turn arc runs "
                     "roughly 0.02-0.38. No individual arc's rate is a measurement (§10).",
    }


def rate_derive_artifact(args) -> dict:
    """[RATE §6] steps 1-6, from the grader label cache."""
    cfg = load_config(args.config)
    turns = collect_turns(args.include_backchannels)
    cache = load_cache(Path(args.cache))
    if not cache:
        raise SystemExit(
            f"no grader labels at {args.cache} — run the `grade` stage first. The targets "
            "come from grader space (C3), not from hand labels, and never from the "
            "Simulator's own output (C5)."
        )

    lbs = labels_by_session(cache, turns, "engine")
    rates = RD.conversation_rates("engine", lbs)
    groups = RD.group_conversations(rates, "engine")

    T = int(args.arc_length_t)

    # §8 — delivery is DECLARED by default. `--delivery measured` derives it from
    # the zero-inflation hurdle instead; see rate_derive for why conditioning is
    # correct on this axis and wrong on engine, and for why it can refuse to load.
    dlv_labels = labels_by_session(cache, turns, "delivery")
    measured_delivery = (args.delivery == "measured")
    dlv_fn = None
    if measured_delivery:
        dlv_fn = lambda tgt: RD.delivery_profile_edges_hurdle(
            tgt, dlv_labels, T=T, p_lo=args.rate_p_lo, p_hi=args.rate_p_hi)

    cells_doc, edges = RD.build_cells(
        args.cells,
        lambda tgt: RD.engine_profile_edges(tgt, rates, T=T,
                                            p_lo=args.rate_p_lo, p_hi=args.rate_p_hi),
        T=T, delivery_edges_fn=dlv_fn, delivery_measured=measured_delivery,
    )

    derivation = _common_derivation(args)
    derivation["source"] = "derived from the grader label cache"
    derivation["corpus"] = corpus_stats(args.include_backchannels)
    derivation["group_sizes"] = {k: len(v) for k, v in groups.items()}
    derivation["n_eligible"] = sum(1 for c in rates if c.eligible)
    derivation["n_conversations"] = len(rates)
    derivation["n_dropped_ties"] = RD.dropped_ties(rates, "engine")

    doc = _rate_doc(cfg, args, cells_doc, derivation,
                    cache_identities(Path(args.cache)) or _cfg_identities(cfg))
    out = Path(args.out or (DEFAULT_OUT / f"rate_calibration.{args.grader_version}.yaml"))
    return _write_rate_artifact(doc, out, edges)


def rate_freeze_v1(args) -> dict:
    """Emit the frozen artifact from [RATE §7]'s already-measured percentiles.

    For the case the label cache is not on this host. The band SHAPING is the
    same code the live path runs (`rate_derive.edges_from_percentiles`); only the
    source of the percentiles differs, and the artifact records which it was so a
    reader can never mistake a transcription for a fresh derivation.
    """
    from grpo.calibration import rate_v1_measurements as V1

    cfg = load_config(args.config)
    T = int(args.arc_length_t)
    cells_doc, edges = RD.build_cells(
        args.cells, lambda tgt: V1.engine_edges_from_spec(tgt, T=T), T=T)

    derivation = _common_derivation(args)
    derivation["source"] = "TRANSCRIBED from RATE §7 (no label cache on this host)"
    derivation["source_note"] = V1.SPEC_V1_NOTE
    derivation["corpus"] = {
        "n_conversations": V1.SPEC_V1_N_CONVERSATIONS,
        "n_substantive_client_turns": V1.SPEC_V1_N_TURNS,
        "n_eligible": V1.SPEC_V1_N_ELIGIBLE,
    }
    derivation["group_sizes"] = dict(V1.SPEC_V1_GROUP_N)

    # C3 — the identities must be the ones that MEASURED the targets. §7 names
    # the engine grader; delivery is declared, so its "measurement" is this
    # config's champion by construction and the reward must still be scored with
    # it (the min over the two axes reads both).
    identities = _cfg_identities(cfg)
    if identities.get("engine") != V1.SPEC_V1_GRADER:
        raise SystemExit(
            f"C3: RATE §7's engine targets were measured with {V1.SPEC_V1_GRADER!r} but "
            f"this config's engine champion is {identities.get('engine')!r}. Bias "
            "cancellation does not hold across a grader swap — restore the champion or "
            "re-derive from a fresh grading pass."
        )

    doc = _rate_doc(cfg, args, cells_doc, derivation, identities)
    out = Path(args.out or (DEFAULT_OUT / f"rate_calibration.{args.grader_version}.yaml"))
    return _write_rate_artifact(doc, out, edges)


def _rate_report(doc: dict, edges_by_cell: dict, sha: str) -> str:
    T = doc["arc_length_T"]
    lines = [
        "# AnnoMI rate-profile calibration — disclosure report",
        "",
        f"spec: `{doc['spec']}`  ",
        f"grader_version: `{doc['grader_version']}`  ",
        f"generated: {doc['generated_utc']}  ",
        f"arc_length_T: **{T}** (bands are validated against it, §5.2/§11)  ",
        f"C9 sha256: `{sha}`  ",
        f"backends: `{doc['backend_identities']}`",
        "",
        f"**Source: {doc['derivation']['source']}**",
        "",
        "## Bands",
        "",
        "| cell | axis | label | role | band | measured | widened |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell, axes in edges_by_cell.items():
        for axis in ("engine", "delivery"):
            for e in axes[axis]:
                lines.append(
                    f"| {cell} | {axis} | {e.label} | {e.role} | "
                    f"[{e.L:.3f}, {e.U:.3f}] | {'yes' if e.measured else '**NO**'} | "
                    f"{'yes' if e.widening else ''} |"
                )

    lines += [
        "",
        "## Delivery is DECLARED, not measured (§8)",
        "",
        RD.DECLARED_DELIVERY_REASON,
        "",
        "Monitoring reports engine and delivery band-fit **separately** so a declared "
        "target is never reported as a measured one. Reverting delivery to per-turn "
        "monotone scoring is rejected: it reinstates the failure this redesign exists "
        "to remove, on the axis where the hot profiles were already hardest to produce.",
        "",
        "## §5.2 widenings",
        "",
    ]
    widened = [(cell, axis, e) for cell, axes in edges_by_cell.items()
               for axis in ("engine", "delivery") for e in axes[axis] if e.widening]
    if not widened:
        lines.append("None.")
    seen = set()
    for cell, axis, e in widened:
        key = (axis, e.label, e.role, round(e.L, 4), round(e.U, 4))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- **{axis}/{e.label}** ({e.role}): {e.widening}")

    lines += [
        "",
        "## What this costs (§10)",
        "",
        doc["derivation"]["scenario_dependence"],
        "",
        doc["derivation"]["precision"],
        "",
        "## Percentile window",
        "",
        doc["derivation"]["percentile_window"]["declared_choice"],
    ]
    if doc["derivation"].get("source_note"):
        lines += ["", "## Source disclosure", "", doc["derivation"]["source_note"]]
    return "\n".join(lines) + "\n"


# ── stage 3: kappa (REPORT-ONLY) ─────────────────────────────────────────────

def kappa(args) -> dict:
    """Human-vs-grader agreement on the hand-labeled subset.

    **Report-only.** C6 removed the gate that would have thresholded this, and
    the band edges come from the grader pass over the full corpus, not from these
    labels. Its value is as an instrument-validity read.

    **The CI is bootstrapped by resampling SESSIONS, not turns** (§6.5): turns
    within a conversation are correlated, so ~500 clustered turns carry less
    information than 500 independent ones and a turn-level CI is too narrow.
    `tools/compute_kappa.bootstrap_ci` resamples turns, so it is deliberately not
    used here; its point statistics are.
    """
    import csv
    import random
    from compute_kappa import safe_kappa, gwet_ac1     # tools/ (flat, via _bootstrap)

    rows = []
    sheet_root = Path(args.sheets) if args.sheets else GENERATOR_DIR
    for sheet in sorted(sheet_root.glob("batch*/hand_labels_annomi_b*.csv")):
        with sheet.open() as f:
            rows += list(csv.DictReader(f))

    labeled = [r for r in rows
               if (r.get("engine_label") or "").strip() or (r.get("delivery_label") or "").strip()]
    print(f"hand-label sheets: {len(rows)} rows, {len(labeled)} with at least one label")
    if not labeled:
        print("\nNo hand labels have been entered yet. The kappa is REPORT-ONLY and does "
              "not gate the run (C6) — the bracket is already derivable from the grader "
              "pass. Re-run this stage once the raters return the sheets.")
        return {"n_labeled": 0}

    cache = load_cache(Path(args.cache))
    out = {"n_labeled": len(labeled)}
    for axis, col, labels in (
        ("engine", "engine_label", list(MARKED["engine"]) + ["neutral"]),
        ("delivery", "delivery_label", list(MARKED["delivery"]) + ["flat"]),
    ):
        pairs = []
        for r in labeled:
            h = (r.get(col) or "").strip().lower()
            gl = cache.get((r["turn_id"], axis))
            if h in labels and gl is not None:
                pairs.append((r["transcript_id"], h, gl))
        if len(pairs) < 5:
            print(f"{axis}: only {len(pairs)} comparable pairs — skipping")
            out[axis] = None
            continue

        h = [p[1] for p in pairs]
        gl = [p[2] for p in pairs]
        k = safe_kappa(h, gl, labels)
        ac1 = gwet_ac1(h, gl, labels)

        # Cluster bootstrap: resample SESSIONS with replacement (§6.5).
        by_sess = defaultdict(list)
        for sid, hh, gg in pairs:
            by_sess[sid].append((hh, gg))
        sids = list(by_sess)
        rng = random.Random(args.seed)
        boots = []
        for _ in range(args.n_boot):
            draw = [by_sess[rng.choice(sids)] for _ in sids]
            flat = [x for grp in draw for x in grp]
            kk = safe_kappa([x[0] for x in flat], [x[1] for x in flat], labels)
            if kk == kk:
                boots.append(kk)
        boots.sort()
        ci = ((boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))])
              if len(boots) >= 40 else (float("nan"), float("nan")))

        out[axis] = {"n_pairs": len(pairs), "n_sessions": len(sids),
                     "kappa": k, "ac1": ac1, "ci95_session_cluster": ci}
        print(f"{axis}: n={len(pairs)} over {len(sids)} sessions  "
              f"kappa={k:.3f}  AC1={ac1:.3f}  95% CI (session-cluster)={ci}")
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["grade", "derive", "rate-derive",
                                     "rate-freeze-v1", "kappa", "all"])
    p.add_argument("--config", default=None)
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--out", default=None)
    p.add_argument("--grader-version", default="v1")
    p.add_argument("--include-backchannels", action="store_true",
                   help="grade all client turns, not just substantive ones (deflates d_anno)")
    # grade
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="cap jobs (smoke test)")
    # derive
    p.add_argument("--p-lo", type=float, default=72.5, help="§6.5 P_lo, disclosed (70-75)")
    p.add_argument("--p-hi", type=float, default=92.5, help="§6.5 P_hi, disclosed (90-95)")
    p.add_argument("--min-marked", type=int, default=D.MIN_MARKED_PER_SESSION)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Dirichlet smoothing; part of the DEFINITION of q — the reward "
                        "must use this same value (see AxisBand docstring)")
    p.add_argument("--s-lo", type=float, default=0.06)
    p.add_argument("--s-hi", type=float, default=0.12)
    p.add_argument("--d-floor-frac", type=float, default=0.6,
                   help="d_floor = frac x median per-session marked fraction")
    p.add_argument("--min-d-floor", type=float, default=0.05)
    p.add_argument("--d-lo", type=float, default=0.05,
                   help="D2.3 neutral-engine lower anchor; MUST be > 0")
    p.add_argument("--neutral-d-percentile", type=float, default=25.0)
    p.add_argument("--cells", nargs="+", default=["b1", "b2", "b3", "b4", "b5", "b6"])
    # rate-derive / rate-freeze-v1 ([RATE §6])
    p.add_argument("--arc-length-t", type=int, default=20,
                   help="§11 T — recorded in the artifact and used to validate every "
                        "band's span (§5.2)")
    p.add_argument("--rate-p-lo", type=float, default=RD.P_LO,
                   help="§6.1 percentile window, a DECLARED design choice")
    p.add_argument("--rate-p-hi", type=float, default=RD.P_HI)
    p.add_argument("--delivery", choices=["declared", "measured"], default="declared",
                   help="§8 delivery targets: `declared` (the default, flagged "
                        "measured: false) or `measured` via the zero-inflation hurdle")
    # kappa
    p.add_argument("--sheets", default=None,
                   help="root holding batchNN/hand_labels_annomi_bNN.csv (default: the "
                        "generated label task)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stage in ("grade", "all"):
        grade(args)
    if args.stage in ("derive", "all"):
        derive_artifact(args)
    if args.stage == "rate-derive":
        rate_derive_artifact(args)
    if args.stage == "rate-freeze-v1":
        rate_freeze_v1(args)
    if args.stage in ("kappa", "all"):
        kappa(args)


if __name__ == "__main__":
    main()
