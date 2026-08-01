"""BLOCKING GATE C6-i — is GRPO even the right tool? (grpo_spec §0.1)

"Across-the-board" non-convergence in the prompt-opt pipeline is equally
consistent with two very different worlds:

  (a) a real Simulator **capability ceiling** — the stated GRPO motivation; or
  (b) a **miscalibrated fidelity checker** scoring genuinely on-profile turns as
      off-profile.

Convergence evidence cannot distinguish them: in both worlds the loop fails to
converge, for opposite reasons. Scorer validity needs *external grounding*, and
the only external ground available is a human reading the turns.

The fork:

  * Turns look off-profile to the human **and** the checker -> policy problem ->
    GRPO is the right tool. **Proceed.**
  * Turns look on-profile to the human but the checker marked them off -> **ruler
    problem**. GRPO would optimize hard toward a wrong target, and every
    downstream artifact would be a confident measurement of the wrong thing.
    **Do not proceed**; fix the checker first.

Cheap (an afternoon) and it forks the whole branch, which is why it is a hard
entry-point assertion rather than a recommendation.

Two-phase, because the middle step is human:

    python -m grpo.run preflight-build   --artifacts results/build_artifacts
    # ... hand-label the emitted sheet on engine + delivery ...
    python -m grpo.run preflight-score   --labels <sheet> --signed-off-by "<name>"

`assert_preflight_signed_off` reads the persisted record and refuses to start GRPO
without a signed-off verdict of `policy_ceiling`.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import grpo._bootstrap  # noqa: F401

from grpo.reward import turn_fidelity
from grpo.stats import agreement, ENGINE_LABELS, DELIVERY_LABELS


DEFAULT_RESULT_PATH = "results/grpo/gates/preflight_nonconvergence.json"

VERDICT_POLICY = "policy_ceiling"     # human agrees the turns are off-profile
VERDICT_RULER = "ruler_problem"       # human says on-profile, checker says off
VERDICT_MIXED = "mixed"               # neither reading dominates

#: Share of checker-flagged-off turns the human must ALSO call off-profile for
#: the policy-ceiling reading. Below this, enough of the checker's "off" calls
#: are human-disputed that the ruler is implicated. Conservative by design: this
#: gate exists to stop a confident optimization against a wrong target.
DEFAULT_AGREEMENT_BAR = 0.80


@dataclass
class DiagnosticItem:
    """One checker-flagged-off turn, awaiting a human read."""

    turn_id: str
    cell: str
    iteration: str
    sample: int
    turn_index: int
    turn: str
    context: str
    checker_engine: str
    checker_delivery: str
    checker_engine_pass: int
    checker_delivery_pass: int


def _iter_artifact_turns(artifacts_dir: str, cells: List[str]):
    """Walk the prompt-opt build artifacts, yielding per-turn checker labels.

    Reads `<cell>/<iter_N>/fidelity_results.json` (the checker's observed labels,
    one list per transcript) alongside `transcript_<sample>.json` (the turns
    themselves). Assistant turns and label rows are positionally aligned, which is
    how the build loop writes them.
    """
    root = Path(artifacts_dir)
    for cell in cells:
        cell_dir = root / cell
        if not cell_dir.is_dir():
            continue
        for iter_dir in sorted(cell_dir.glob("iter_*")):
            fr = iter_dir / "fidelity_results.json"
            if not fr.exists():
                continue
            try:
                data = json.loads(fr.read_text())
            except json.JSONDecodeError:
                continue
            for tr in data.get("transcripts", []):
                if tr.get("discarded"):
                    continue
                sample = tr.get("sample", 0)
                labels = tr.get("labels") or []
                tpath = iter_dir / f"transcript_{sample}.json"
                if not tpath.exists():
                    continue
                try:
                    msgs = json.loads(tpath.read_text())
                except json.JSONDecodeError:
                    continue
                patient = [m for m in msgs if m.get("role") == "assistant"]
                for idx, lab in enumerate(labels):
                    if idx >= len(patient):
                        break
                    context = _context_before(msgs, patient[idx])
                    yield cell, iter_dir.name, sample, idx, patient[idx]["content"], context, lab


def _context_before(msgs: List[dict], target: dict) -> str:
    """The conversation prefix before `target`, in the annotator's format."""
    out = []
    for m in msgs:
        if m is target:
            break
        speaker = "Patient" if m.get("role") == "assistant" else "Model"
        out.append(f"[{speaker}]: {m.get('content', '')}")
    return "\n".join(out[-6:])


def build_diagnostic_sheet(
    artifacts_dir: str,
    cells: List[str],
    out_labels: str,
    out_key: str,
    per_cell: int = 25,
    seed: int = 0,
) -> List[DiagnosticItem]:
    """Sample turns the CHECKER scored off-profile and emit a blind labelling sheet.

    Only checker-flagged-off turns are sampled: the §0.1 question is specifically
    whether the checker's *negative* calls are trustworthy, so the on-profile mass
    carries no information for this fork. The human never sees the checker's
    labels (they live in the key file) — otherwise the fork collapses into
    anchoring on the ruler the gate is trying to audit.
    """
    rng = random.Random(seed)
    by_cell: Dict[str, List[DiagnosticItem]] = {c: [] for c in cells}
    counter = 0

    for cell, iteration, sample, idx, turn, context, lab in _iter_artifact_turns(artifacts_dir, cells):
        e_pass = turn_fidelity.engine_pass(lab, cell)
        d_pass = turn_fidelity.delivery_pass(lab, cell)
        if e_pass and d_pass:
            continue                      # checker says on-profile: not the question
        by_cell[cell].append(DiagnosticItem(
            turn_id=f"p{counter:05d}", cell=cell, iteration=iteration, sample=sample,
            turn_index=idx, turn=turn, context=context,
            checker_engine=str(lab.get("engine_direction", "")),
            checker_delivery=str(lab.get("delivery", "")),
            checker_engine_pass=e_pass, checker_delivery_pass=d_pass,
        ))
        counter += 1

    items: List[DiagnosticItem] = []
    for cell in cells:
        pool = by_cell.get(cell, [])
        rng.shuffle(pool)
        items.extend(pool[:per_cell])

    if not items:
        raise RuntimeError(
            f"No checker-flagged-off turns found under {artifacts_dir} for cells {cells}. "
            "Either the artifacts are missing or nothing was scored off-profile — in "
            "which case the stated GRPO motivation (across-the-board non-convergence) "
            "is not visible in these artifacts and should be re-examined before "
            "committing to a finetune."
        )

    _write_sheet(items, out_labels)
    _write_key(items, out_key)
    return items


def _write_sheet(items: List[DiagnosticItem], out_labels: str) -> None:
    p = Path(out_labels)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "cell", "context", "turn", "engine_label", "delivery_label"])
        for it in items:
            # `cell` is shown because the human must judge on-profile RELATIVE to the
            # target pole; it is the checker's LABELS that stay hidden.
            w.writerow([it.turn_id, it.cell, it.context, it.turn, "", ""])


def _write_key(items: List[DiagnosticItem], out_key: str) -> None:
    p = Path(out_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "cell", "iteration", "sample", "turn_index",
                    "checker_engine", "checker_delivery",
                    "checker_engine_pass", "checker_delivery_pass"])
        for it in items:
            w.writerow([it.turn_id, it.cell, it.iteration, it.sample, it.turn_index,
                        it.checker_engine, it.checker_delivery,
                        it.checker_engine_pass, it.checker_delivery_pass])


@dataclass
class PreflightResult:
    """§0.1 outcome — the fork, plus the evidence behind it."""

    verdict: str
    n_labelled: int
    human_agrees_off_rate: float
    bar: float
    disputed_engine: int = 0
    disputed_delivery: int = 0
    by_cell: Dict[str, dict] = field(default_factory=dict)
    label_agreement: Dict[str, dict] = field(default_factory=dict)
    signed_off_by: str = ""
    signed_off_at: str = ""
    notes: str = ""

    @property
    def clears(self) -> bool:
        return self.verdict == VERDICT_POLICY and bool(self.signed_off_by)

    def to_dict(self) -> dict:
        return {
            "gate": "preflight_nonconvergence (grpo_spec §0.1, C6-i)",
            "verdict": self.verdict,
            "clears": self.clears,
            "n_labelled": self.n_labelled,
            "human_agrees_off_rate": round(self.human_agrees_off_rate, 4),
            "bar": self.bar,
            "disputed_engine": self.disputed_engine,
            "disputed_delivery": self.disputed_delivery,
            "by_cell": self.by_cell,
            "label_agreement": self.label_agreement,
            "signed_off_by": self.signed_off_by,
            "signed_off_at": self.signed_off_at,
            "notes": self.notes,
        }


def score_diagnostic(
    labels_path: str,
    key_path: str,
    signed_off_by: str = "",
    bar: float = DEFAULT_AGREEMENT_BAR,
    notes: str = "",
    result_path: Optional[str] = DEFAULT_RESULT_PATH,
) -> PreflightResult:
    """Compare the human read to the checker's, and record the fork.

    For each sampled turn the checker called off-profile on some dimension, the
    human's own labels are re-run through the SAME cell-relative pass rule. If the
    human's labels also fail that dimension, the two agree the turn is off-profile
    (policy problem). If the human's labels pass where the checker failed, that
    turn is a **disputed** call and counts toward the ruler reading.
    """
    labels = _read_csv(labels_path)
    key = {r["turn_id"]: r for r in _read_csv(key_path)}

    agreed = disputed = 0
    disputed_engine = disputed_delivery = 0
    per_cell: Dict[str, List[int]] = {}
    h_eng, m_eng, h_dlv, m_dlv = [], [], [], []

    for row in labels:
        tid = (row.get("turn_id") or "").strip()
        k = key.get(tid)
        if not k:
            continue
        he = (row.get("engine_label") or "").strip().lower()
        hd = (row.get("delivery_label") or "").strip().lower()
        if not he and not hd:
            continue                      # unlabelled row
        cell = k.get("cell", "")

        human_labels = {"engine_direction": he, "delivery": hd}
        h_e_pass = turn_fidelity.engine_pass(human_labels, cell) if he else 1
        h_d_pass = turn_fidelity.delivery_pass(human_labels, cell) if hd else 1
        c_e_pass = int(k.get("checker_engine_pass", 1))
        c_d_pass = int(k.get("checker_delivery_pass", 1))

        # A turn is "agreed off-profile" when the human ALSO fails a dimension the
        # checker failed. Disputed when the human passes everything the checker failed.
        overlap_off = ((not c_e_pass and not h_e_pass) or (not c_d_pass and not h_d_pass))
        if overlap_off:
            agreed += 1
        else:
            disputed += 1
            if not c_e_pass and h_e_pass:
                disputed_engine += 1
            if not c_d_pass and h_d_pass:
                disputed_delivery += 1

        per_cell.setdefault(cell, []).append(1 if overlap_off else 0)

        if he in ENGINE_LABELS and k.get("checker_engine") in ENGINE_LABELS:
            h_eng.append(he); m_eng.append(k["checker_engine"])
        if hd in DELIVERY_LABELS and k.get("checker_delivery") in DELIVERY_LABELS:
            h_dlv.append(hd); m_dlv.append(k["checker_delivery"])

    n = agreed + disputed
    if n == 0:
        raise RuntimeError(
            f"No hand labels found in {labels_path}. §0.1 is a HUMAN diagnostic — it "
            "cannot be satisfied by re-running the checker against itself."
        )

    rate = agreed / n
    if rate >= bar:
        verdict = VERDICT_POLICY
    elif rate <= 0.5:
        verdict = VERDICT_RULER
    else:
        verdict = VERDICT_MIXED

    label_agreement = {}
    if h_eng:
        label_agreement["engine"] = agreement(h_eng, m_eng, ENGINE_LABELS).to_dict()
    if h_dlv:
        label_agreement["delivery"] = agreement(h_dlv, m_dlv, DELIVERY_LABELS).to_dict()

    result = PreflightResult(
        verdict=verdict, n_labelled=n, human_agrees_off_rate=rate, bar=bar,
        disputed_engine=disputed_engine, disputed_delivery=disputed_delivery,
        by_cell={c: {"n": len(v), "agree_rate": round(sum(v) / len(v), 4)}
                 for c, v in per_cell.items() if v},
        label_agreement=label_agreement,
        signed_off_by=signed_off_by,
        signed_off_at=datetime.now(timezone.utc).isoformat() if signed_off_by else "",
        notes=notes,
    )
    if result_path:
        p = Path(result_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result.to_dict(), indent=2))
    return result


def _read_csv(path: str) -> List[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


class PreflightGateError(AssertionError):
    """Raised when the §0.1 diagnostic has not cleared GRPO to proceed."""


def assert_preflight_signed_off(result_path: str = DEFAULT_RESULT_PATH) -> dict:
    """Hard, blocking assertion for the GRPO entry point (C6-i)."""
    p = Path(result_path)
    if not p.exists():
        raise PreflightGateError(
            f"C6-i NOT SATISFIED: no §0.1 diagnostic result at {result_path}. Before "
            "committing to GRPO, hand-check a sample of the non-converged turns against "
            "what the checker scored — it forks policy-capability-ceiling (GRPO is the "
            "right tool) from miscalibrated-ruler (GRPO would optimize against a bad "
            "target). Run `grpo.run preflight-build`, label the sheet, then "
            "`grpo.run preflight-score --signed-off-by <name>`."
        )
    data = json.loads(p.read_text())

    if not data.get("signed_off_by"):
        raise PreflightGateError(
            f"C6-i NOT SIGNED OFF: {result_path} exists but carries no `signed_off_by`. "
            "The spec requires a signed-off result, not merely a computed one — a human "
            "has to own the fork."
        )
    verdict = data.get("verdict")
    if verdict != VERDICT_POLICY:
        detail = {
            VERDICT_RULER: (
                "The turns read ON-profile to the human but the checker marked them off. "
                "This is a RULER problem: GRPO would optimize hard toward a wrong target. "
                "Fix the checker first."
            ),
            VERDICT_MIXED: (
                "Neither reading dominates — the checker is partly disputed. Resolve the "
                "disputed calls before committing a finetune to this target."
            ),
        }.get(verdict, "Unrecognised verdict.")
        raise PreflightGateError(
            f"C6-i FAILED: §0.1 verdict is {verdict!r} "
            f"(human_agrees_off_rate={data.get('human_agrees_off_rate')}, "
            f"bar={data.get('bar')}, disputed_engine={data.get('disputed_engine')}, "
            f"disputed_delivery={data.get('disputed_delivery')}). {detail} "
            "GRPO must NOT start."
        )
    return data
