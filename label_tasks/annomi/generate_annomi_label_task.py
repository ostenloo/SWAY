#!/usr/bin/env python3
"""Build the AnnoMI engine/delivery hand-labeling task (grpo_spec_2 §6.5).

The sampling unit is the **conversation, not the turn**: we pick whole sessions
and label every substantive client turn in them. That is what lets the human
labels support a *per-patient* `q` — the quantity the band bracket is actually
built from (§6.5) — rather than only a per-turn kappa. It also matches how the
grader reads the data (`.label(turn, context, cell)` sees context), and both
constructs are context-dependent: delivery is defined as directed *toward the
listener*, which cannot be judged from an isolated turn.

Outputs, all deterministic under --seed:

  annomi_sample_manifest.csv     one row per selected session
  batchNN/hand_labels_annomi_bNN.csv    the label sheet (blank label columns)
  batchNN/transcripts_bNN.md            full readable conversations for context

Conversations are never split across batches.

Usage:
  python label_tasks/annomi/generate_annomi_label_task.py
  python label_tasks/annomi/generate_annomi_label_task.py --target-turns 500 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANNOMI = REPO / "AnnoMI" / "AnnoMI-simple.csv"
OUT = Path(__file__).resolve().parent

# §6.5: a session enters the per-patient distribution only with enough marked
# turns. We cannot know `marked` before grading, so eligibility here uses
# substantive turns as the proxy and the real marked-turn filter is applied
# later, in the grader pass.
MIN_SUBSTANTIVE = 8

# Backchannel filter. §6.5 requires the drop count be logged, since it moves
# `d_anno`. Backchannels are dropped from the LABEL SHEET but kept in the
# transcript the rater reads — the arc should be complete.
MAX_BACKCHANNEL_WORDS = 3

# The closed-vocabulary half of the filter. A turn is dropped when EVERY token is
# filler, regardless of length — this catches "Oh, oh, yeah, the—Yeah", which the
# length rule alone lets through.
#
# Deliberately narrow. "Carries no engine/delivery signal" is very nearly the
# definition of neutral/flat, which is a label we need in the denominator of
# `d_anno`, so an aggressive filter would inflate `d_anno` rather than clean it.
# "Yeah, it's been a while" survives and should: it is bland, not absent.
BACKCHANNEL_VOCAB = {
    "a", "ah", "alright", "and", "aw", "ay", "eh", "er", "exactly", "gotcha",
    "hm", "hmm", "huh", "i", "it", "its", "kay", "like", "m", "mhm", "mhmm",
    "mm", "mmhm", "mmm", "n", "nah", "no", "nope", "o", "of", "oh", "ok",
    "okay", "oo", "ooh", "quite", "really", "right", "so", "sure", "that",
    "thats", "the", "true", "uh", "uhh", "uhhuh", "uhm", "um", "umm", "well",
    "wow", "y", "ya", "yea", "yeah", "yep", "yes", "yup",
}

LABEL_FIELDS = [
    "turn_id",
    "transcript_id",
    "utterance_id",
    "turn_index",
    "mi_quality",
    "context_prev_assistant",
    "patient_turn",
    "engine_label",
    "delivery_label",
    "flag_ambiguous",
    "notes",
]


def turn_id(transcript_id: str, utterance_id: str) -> str:
    return hashlib.sha1(f"annomi:{transcript_id}:{utterance_id}".encode()).hexdigest()[:12]


def load_sessions() -> dict[str, list[dict]]:
    """transcript_id -> utterances in order (therapist and client both)."""
    rows = list(csv.DictReader(ANNOMI.open()))
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_session[r["transcript_id"]].append(r)
    for tid in by_session:
        by_session[tid].sort(key=lambda r: int(r["utterance_id"]))
    return by_session


def _tokens(text: str) -> list[str]:
    cleaned = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in text)
    return cleaned.split()


def is_substantive(text: str) -> bool:
    """§6.5's length + closed-vocabulary backchannel filter."""
    toks = _tokens(text)
    if len(toks) <= MAX_BACKCHANNEL_WORDS:
        return False
    if all(t in BACKCHANNEL_VOCAB for t in toks):
        return False
    return True


def select_sessions(by_session, target_turns: int, low_quota_turns: int, seed: int):
    """Draw whole sessions until ~target_turns substantive client turns.

    `low_quota_turns` reserves part of the target for mi_quality=low sessions —
    a purely random draw under-represents them (only 16 of 109 eligible), and
    low-quality sessions plausibly carry a different engine/delivery mix.
    """
    eligible = {"high": [], "low": []}
    for tid, uts in by_session.items():
        sub = [u for u in uts if u["interlocutor"] == "client" and is_substantive(u["utterance_text"])]
        if len(sub) >= MIN_SUBSTANTIVE:
            eligible[uts[0]["mi_quality"]].append((tid, len(sub)))

    rng = random.Random(seed)
    for q in eligible:
        eligible[q].sort()           # deterministic order before shuffling
        rng.shuffle(eligible[q])

    picked, total = [], 0
    for tid, n in eligible["low"]:
        if total >= low_quota_turns:
            break
        picked.append(tid)
        total += n
    for tid, n in eligible["high"]:
        if total >= target_turns:
            break
        picked.append(tid)
        total += n

    counts = {q: len(v) for q, v in eligible.items()}
    return picked, total, counts


def batch_sessions(picked, by_session, n_batches: int):
    """Greedy: keep batches near-equal in labelable turns, never split a session."""
    sizes = {
        tid: sum(
            1
            for u in by_session[tid]
            if u["interlocutor"] == "client" and is_substantive(u["utterance_text"])
        )
        for tid in picked
    }
    # Largest-first into the currently-lightest batch — standard greedy balance.
    order = sorted(picked, key=lambda t: -sizes[t])
    batches: list[list[str]] = [[] for _ in range(n_batches)]
    loads = [0] * n_batches
    for tid in order:
        i = loads.index(min(loads))
        batches[i].append(tid)
        loads[i] += sizes[tid]
    return batches, sizes


def write_batch(idx: int, tids: list[str], by_session, sizes):
    bdir = OUT / f"batch{idx:02d}"
    bdir.mkdir(parents=True, exist_ok=True)

    label_rows = []
    md = [
        f"# AnnoMI label batch {idx:02d} — full conversations\n",
        "Read each conversation start to finish before labeling its client turns.\n",
        "Backchannels appear here for context but are **not** on the label sheet.\n",
        "Rows marked **[LABEL]** correspond to a row in the CSV, matched by `turn_id`.\n",
        "\n---\n",
    ]

    for tid in sorted(tids, key=lambda t: int(t)):
        uts = by_session[tid]
        head = uts[0]
        md.append(f"\n## Conversation `{tid}` — MI quality: **{head['mi_quality']}**\n")
        md.append(f"*Topic:* {head['topic']}\n")
        md.append(f"*Client turns to label:* {sizes[tid]}\n\n")

        prev_therapist = ""
        client_i = 0
        for u in uts:
            text = u["utterance_text"].strip()
            if u["interlocutor"] == "therapist":
                md.append(f"**Therapist:** {text}\n\n")
                prev_therapist = text
                continue

            if is_substantive(text):
                client_i += 1
                tuid = turn_id(tid, u["utterance_id"])
                md.append(f"**[LABEL {tuid}] Client:** {text}\n\n")
                label_rows.append(
                    {
                        "turn_id": tuid,
                        "transcript_id": tid,
                        "utterance_id": u["utterance_id"],
                        "turn_index": client_i,
                        "mi_quality": head["mi_quality"],
                        "context_prev_assistant": prev_therapist,
                        "patient_turn": text,
                        "engine_label": "",
                        "delivery_label": "",
                        "flag_ambiguous": "",
                        "notes": "",
                    }
                )
            else:
                md.append(f"*Client (backchannel, not labeled):* {text}\n\n")

        md.append("\n---\n")

    csv_path = bdir / f"hand_labels_annomi_b{idx:02d}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(label_rows)

    (bdir / f"transcripts_b{idx:02d}.md").write_text("".join(md))
    return len(label_rows), len(tids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-turns", type=int, default=500)
    ap.add_argument("--low-quota-turns", type=int, default=100)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_session = load_sessions()
    picked, total, eligible_counts = select_sessions(
        by_session, args.target_turns, args.low_quota_turns, args.seed
    )
    batches, sizes = batch_sessions(picked, by_session, args.batches)

    manifest = []
    for i, tids in enumerate(batches, start=1):
        n_turns, n_sess = write_batch(i, tids, by_session, sizes)
        for tid in tids:
            head = by_session[tid][0]
            manifest.append(
                {
                    "transcript_id": tid,
                    "mi_quality": head["mi_quality"],
                    "topic": head["topic"],
                    "batch": f"{i:02d}",
                    "n_client_turns": sum(
                        1 for u in by_session[tid] if u["interlocutor"] == "client"
                    ),
                    "n_labelable": sizes[tid],
                    "video_url": head["video_url"],
                }
            )
        print(f"  batch{i:02d}: {n_sess:2d} conversations, {n_turns:3d} labelable turns")

    manifest.sort(key=lambda r: (r["batch"], int(r["transcript_id"])))
    with (OUT / "annomi_sample_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(manifest)

    n_low = sum(1 for r in manifest if r["mi_quality"] == "low")
    # §6.5 requires the drop count be logged — it moves `d_anno`. Broken out by
    # rule so the sensitivity of `d_anno` to this threshold stays visible.
    client = [u for uts in by_session.values() for u in uts if u["interlocutor"] == "client"]
    by_len = sum(1 for u in client if len(_tokens(u["utterance_text"])) <= MAX_BACKCHANNEL_WORDS)
    by_vocab = sum(
        1
        for u in client
        if len(_tokens(u["utterance_text"])) > MAX_BACKCHANNEL_WORDS
        and not is_substantive(u["utterance_text"])
    )
    print(f"\neligible sessions: high={eligible_counts['high']} low={eligible_counts['low']}")
    print(f"selected: {len(picked)} conversations ({n_low} low-quality), {total} labelable turns")
    print(f"\nbackchannel drops corpus-wide (of {len(client)} client turns):")
    print(f"  by length (<= {MAX_BACKCHANNEL_WORDS} tokens): {by_len}")
    print(f"  by closed vocabulary (all-filler, longer):    {by_vocab}")
    print(f"  substantive remaining:                        {len(client) - by_len - by_vocab}")
    print(f"seed={args.seed}  -> rerunning with the same seed reproduces this exact sample")


if __name__ == "__main__":
    main()
