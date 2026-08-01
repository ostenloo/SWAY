"""BLOCKING GATE C6-ii — stratified delivery validation on real rollouts (§8.2).

Delivery is single-covered, and its characteristic error (employer-grievance read
as listener-hostility) is unopposed. §8.1 hardens the discriminator by decomposing
into Q1/Q2; this module measures whether the hardening actually worked, **on the
distribution the optimizer will drive toward**.

Three things make this the load-bearing gate rather than the authored-pair probe:

1. **The right distribution.** There is no natural corpus, and the prompt-opt
   corpus is off-distribution and untrusted — the non-converged cells are exactly
   the ones where the Simulator overrode the profile. So validation runs on the
   **RFT-filtered set** (§6), the first real rollout distribution the delivery
   champion is asked to score in anger. `gates/authored_pairs_smoketest.py` (§8.3)
   remains available as a weaker pre-warm-start smoke test; it is NOT this gate.

2. **Stratified, not pooled.** Grievance-heavy turns are rare in the pool, so a
   systematic grievance->hot bias hides in the off-diagonal mass: it can clear
   pooled 0.80 while failing the entire stratum the optimizer will relocate mass
   onto. This module reports both and treats the *stratum* as the gate. A pooled
   pass with a stratum failure is not a near-miss — per §8.2 step 4 that gap
   **is** the hole, made precise.

3. **CI lower bound.** The bar is 0.80 on the **bootstrap-CI lower bound**, not
   the point estimate (`grpo.stats`, reusing `tools/compute_kappa.py`).

Two-phase, because step 2 is human:

    python -m grpo.run stratum-build   --rft results/grpo/rft_dataset.jsonl
    # ... hand-label the emitted sheet on Q1/Q2 ...
    python -m grpo.run stratum-score   --labels <sheet>

`score_stratum` persists a signed result JSON; `assert_stratified_gate` (called by
the GRPO entry point) reads that file and refuses to start unless a human-labelled
result for THIS delivery backend clears the bar.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import grpo._bootstrap  # noqa: F401

from grpo.stats import KAPPA_BAR, HOT_LABELS, Agreement, agreement
from grpo.reward import delivery_decompose as dd


DEFAULT_RESULT_PATH = "results/grpo/gates/delivery_stratified_validation.json"

#: Strata a turn can land in, from the champion's own Q1/Q2 read. The gate is
#: scored on CONTESTED = grievance | hostility | both: the region where the
#: grievance->hot confusion lives. `neither` turns are uninformative here (they
#: are the easy not-hot mass that inflates pooled kappa).
STRATUM_GRIEVANCE = "grievance_only"     # Q2 yes, Q1 no  <- the exploit's home
STRATUM_HOSTILITY = "hostility"          # Q1 yes, Q2 no
STRATUM_BOTH = "both"                    # Q1 yes, Q2 yes
STRATUM_NEITHER = "neither"
CONTESTED_STRATA = (STRATUM_GRIEVANCE, STRATUM_HOSTILITY, STRATUM_BOTH)


@dataclass
class StratumItem:
    """One turn in the validation stratum, with the champion's blind read."""

    turn_id: str
    cell: str
    context: str
    turn: str
    model_q1: bool
    model_q2: bool
    stratum: str

    @property
    def model_hot(self) -> str:
        return "hot" if self.model_q1 else "not_hot"


def _stratum_of(q1: bool, q2: bool) -> str:
    if q1 and q2:
        return STRATUM_BOTH
    if q1:
        return STRATUM_HOSTILITY
    if q2:
        return STRATUM_GRIEVANCE
    return STRATUM_NEITHER


def load_rft_turns(rft_path: str) -> List[dict]:
    """Read the RFT-filtered set emitted by `train/rft_warmstart.py`."""
    p = Path(rft_path)
    if not p.exists():
        raise FileNotFoundError(
            f"RFT dataset not found: {rft_path}. §8.2 validates on the warm-start "
            "output, so run `python -m grpo.run warmstart` first — this gate cannot "
            "be satisfied before the RFT set exists (that is the point of the "
            "re-sequencing in the spec's revision note)."
        )
    rows = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_stratum(
    rft_path: str,
    delivery_backend,
    out_labels: str,
    out_key: str,
    per_stratum_cap: int = 40,
    include_neither: int = 10,
) -> List[StratumItem]:
    """Build the grievance-vs-hostility stratum and emit a BLIND labelling sheet.

    Oversamples the contested region: the champion's own Q1/Q2 read is used only
    to *stratify* (so the rare grievance turns are actually present in useful
    numbers), never as the label. The human labels are the ground truth and land
    in a separate file, so the annotator never sees the model's answer — the same
    blind discipline as the batch01-03 hand-labelling sheets.

    A small `include_neither` sample is carried so the pooled comparison in §8.2
    step 4 is computable from the same sheet.

    On minimal pairs: §8.2 asks for them "where feasible" — same anger redirected
    employer->therapist with length and lexical intensity held fixed. Those cannot
    be *mined* reliably from rollouts, so this builder does stratified
    oversampling and records the stratum on each row. If you want true minimal
    pairs, author them into the §8.3 smoke-test set, where intensity is controlled
    by construction; the stratified gate here is the distributional test.
    """
    rows = load_rft_turns(rft_path)
    buckets: Dict[str, List[StratumItem]] = {
        STRATUM_GRIEVANCE: [], STRATUM_HOSTILITY: [], STRATUM_BOTH: [], STRATUM_NEITHER: [],
    }

    for i, row in enumerate(rows):
        turn, context, cell = _turn_context_cell(row)
        if not turn:
            continue
        decomp = _decompose(delivery_backend, turn, context, cell)
        stratum = _stratum_of(decomp.q1_hostility_toward_listener,
                              decomp.q2_grievance_toward_absent_party)
        cap = include_neither if stratum == STRATUM_NEITHER else per_stratum_cap
        if len(buckets[stratum]) >= cap:
            continue
        buckets[stratum].append(StratumItem(
            turn_id=f"s{i:05d}", cell=cell, context=context, turn=turn,
            model_q1=decomp.q1_hostility_toward_listener,
            model_q2=decomp.q2_grievance_toward_absent_party,
            stratum=stratum,
        ))

    items = [it for s in (*CONTESTED_STRATA, STRATUM_NEITHER) for it in buckets[s]]
    if not any(buckets[s] for s in CONTESTED_STRATA):
        raise RuntimeError(
            "No contested turns found in the RFT set — the champion reported neither "
            "listener-hostility nor employer-grievance anywhere. Either the RFT set is "
            "tiny or the delivery champion is answering degenerately; inspect it before "
            "treating this gate as passable."
        )

    _write_labels_sheet(items, out_labels)
    _write_key(items, out_key)
    return items


def _turn_context_cell(row: dict) -> tuple:
    """Accept both the SFT-record shape and the raw RFTExample shape."""
    if "messages" in row:
        msgs = row["messages"]
        turn = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
        context = next((m["content"] for m in msgs if m["role"] == "user"), "")
        return turn, context, row.get("cell", "")
    return row.get("completion", ""), row.get("context", ""), row.get("cell", "")


def _decompose(delivery_backend, turn: str, context: str, cell: str):
    """Get Q1/Q2 from the delivery backend, accepting an adapter or a bare core."""
    if hasattr(delivery_backend, "decompose"):
        return delivery_backend.decompose(turn, context, cell)
    core = getattr(delivery_backend, "core", delivery_backend)
    return dd.decomposition_from_labels(core.labels(turn, context, cell))


LABEL_COLUMNS = ["turn_id", "q1_hostility_toward_listener", "q2_grievance_toward_absent_party"]


def _write_labels_sheet(items: List[StratumItem], out_labels: str) -> None:
    """Blind sheet: context + turn + empty label columns, stratum NOT shown."""
    p = Path(out_labels)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "context", "turn",
                    "q1_hostility_toward_listener", "q2_grievance_toward_absent_party"])
        for it in items:
            w.writerow([it.turn_id, it.context, it.turn, "", ""])


def _write_key(items: List[StratumItem], out_key: str) -> None:
    p = Path(out_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "cell", "stratum", "model_q1", "model_q2", "turn"])
        for it in items:
            w.writerow([it.turn_id, it.cell, it.stratum,
                        str(it.model_q1).lower(), str(it.model_q2).lower(), it.turn])


def _as_bool(v) -> Optional[bool]:
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    return None


@dataclass
class StratifiedValidationResult:
    """§8.2 outcome: the contested-stratum gate plus the pooled comparison."""

    contested: Agreement
    pooled: Agreement
    by_stratum: Dict[str, dict] = field(default_factory=dict)
    by_cell: Dict[str, dict] = field(default_factory=dict)
    grievance_scored_hot: int = 0
    hot_missed: int = 0
    delivery_backend_identity: str = "unknown"
    n_labelled: int = 0
    scored_at: str = ""

    @property
    def passed(self) -> bool:
        return self.contested.passed

    @property
    def pooled_masks_failure(self) -> bool:
        """The §8.2 step 4 diagnosis: pooled clears, contested stratum does not.
        That gap IS the hole — not a near-miss."""
        return self.pooled.passed and not self.contested.passed

    def failing_cells(self) -> List[str]:
        return [c for c, d in self.by_cell.items() if d.get("passed") is False]

    def to_dict(self) -> dict:
        return {
            "gate": "delivery_stratified_validation (grpo_spec §8.2, C6-ii)",
            "passed": self.passed,
            "pooled_masks_failure": self.pooled_masks_failure,
            "delivery_backend_identity": self.delivery_backend_identity,
            "n_labelled": self.n_labelled,
            "scored_at": self.scored_at,
            "contested_stratum": self.contested.to_dict(),
            "pooled": self.pooled.to_dict(),
            "by_stratum": self.by_stratum,
            "by_cell": self.by_cell,
            "confusion": {
                "grievance_scored_hot": self.grievance_scored_hot,
                "hot_missed": self.hot_missed,
            },
            "failing_cells": self.failing_cells(),
        }


def score_stratum(
    labels_path: str,
    key_path: str,
    bar: float = KAPPA_BAR,
    delivery_backend_identity: str = "unknown",
    result_path: Optional[str] = DEFAULT_RESULT_PATH,
) -> StratifiedValidationResult:
    """Join hand labels to the key and compute the stratified gate.

    The gate statistic is agreement on **Q1 collapsed to hot/not_hot** — that is
    the label the reward actually consumes (`hot = Q1`). Q2 is carried through so
    the grievance->hot confusion count is reportable, but Q2 disagreements never
    move the gate, exactly as they never move the delivery label.
    """
    labels = _read_csv(labels_path)
    key = {r["turn_id"]: r for r in _read_csv(key_path)}

    human_hot: List[str] = []
    model_hot: List[str] = []
    strata: List[str] = []
    cells: List[str] = []
    grievance_scored_hot = hot_missed = 0

    for row in labels:
        tid = row.get("turn_id", "").strip()
        k = key.get(tid)
        if not k:
            continue
        q1 = _as_bool(row.get("q1_hostility_toward_listener"))
        if q1 is None:
            continue          # unlabelled row, dropped
        h = "hot" if q1 else "not_hot"
        m = "hot" if _as_bool(k.get("model_q1")) else "not_hot"
        human_hot.append(h)
        model_hot.append(m)
        strata.append(k.get("stratum", ""))
        cells.append(k.get("cell", ""))
        if h == "not_hot" and m == "hot":
            grievance_scored_hot += 1
        elif h == "hot" and m == "not_hot":
            hot_missed += 1

    if not human_hot:
        raise RuntimeError(
            f"No hand labels found in {labels_path}. §8.2 requires HUMAN labels on the "
            "stratum — the gate cannot be satisfied by the model labelling itself."
        )

    contested_idx = [i for i, s in enumerate(strata) if s in CONTESTED_STRATA]
    contested = agreement([human_hot[i] for i in contested_idx],
                          [model_hot[i] for i in contested_idx],
                          HOT_LABELS, bar=bar)
    pooled = agreement(human_hot, model_hot, HOT_LABELS, bar=bar)

    by_stratum = {}
    for s in (*CONTESTED_STRATA, STRATUM_NEITHER):
        idx = [i for i, v in enumerate(strata) if v == s]
        if idx:
            by_stratum[s] = agreement([human_hot[i] for i in idx],
                                      [model_hot[i] for i in idx],
                                      HOT_LABELS, bar=bar).to_dict()

    by_cell = {}
    for cell in sorted({c for c in cells if c}):
        idx = [i for i, c in enumerate(cells) if c == cell and strata[i] in CONTESTED_STRATA]
        if idx:
            by_cell[cell] = agreement([human_hot[i] for i in idx],
                                      [model_hot[i] for i in idx],
                                      HOT_LABELS, bar=bar).to_dict()

    result = StratifiedValidationResult(
        contested=contested, pooled=pooled, by_stratum=by_stratum, by_cell=by_cell,
        grievance_scored_hot=grievance_scored_hot, hot_missed=hot_missed,
        delivery_backend_identity=delivery_backend_identity,
        n_labelled=len(human_hot),
        scored_at=datetime.now(timezone.utc).isoformat(),
    )
    if result_path:
        p = Path(result_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result.to_dict(), indent=2))
    return result


def _read_csv(path: str) -> List[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


class StratifiedGateError(AssertionError):
    """Raised when C6-ii is not satisfied for the configured delivery backend."""


def assert_stratified_gate(
    delivery_backend_identity: str,
    result_path: str = DEFAULT_RESULT_PATH,
    bar: float = KAPPA_BAR,
) -> dict:
    """Hard, blocking assertion for the GRPO entry point (C6-ii).

    Reads the persisted §8.2 result rather than re-running it: the gate is a
    human-labelling outcome, so it cannot be recomputed at launch. Refuses to
    start when the file is missing, was produced for a DIFFERENT delivery
    backend, or fails the CI lower bound.
    """
    p = Path(result_path)
    if not p.exists():
        raise StratifiedGateError(
            f"C6-ii NOT SATISFIED: no §8.2 stratified validation result at {result_path}. "
            "GRPO must not start behind an unvalidated delivery champion. Run "
            "`grpo.run warmstart`, then `stratum-build`, hand-label the sheet, then "
            "`stratum-score`."
        )
    data = json.loads(p.read_text())

    recorded = data.get("delivery_backend_identity", "unknown")
    if recorded != delivery_backend_identity:
        raise StratifiedGateError(
            f"C6-ii STALE: the §8.2 result was produced for delivery backend {recorded!r} "
            f"but the configured backend is {delivery_backend_identity!r}. A validation of "
            "one champion says nothing about another — re-run §8.2 for this backend."
        )

    ci_low = (data.get("contested_stratum") or {}).get("kappa_ci", [None, None])[0]
    if ci_low is None or ci_low < bar:
        pooled_note = ""
        if data.get("pooled_masks_failure"):
            pooled_note = (
                " NOTE: the POOLED kappa clears the bar while the contested stratum does "
                "not — per §8.2 step 4 that gap is the hole, made precise, not a "
                "near-miss. Harden Q1-in-isolation or source a stronger delivery checker."
            )
        failing = data.get("failing_cells") or []
        cell_note = f" Failing cells: {failing}." if failing else ""
        raise StratifiedGateError(
            f"C6-ii FAILED: contested-stratum kappa CI lower bound = {ci_low} < {bar} for "
            f"delivery backend {recorded!r} "
            f"(grievance_scored_hot={data.get('confusion', {}).get('grievance_scored_hot')}). "
            "GRPO must NOT start — the grievance->hot confusion would become a gradient."
            f"{pooled_note}{cell_note}"
        )
    return data
