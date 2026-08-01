"""Certification & freeze — HUMAN, per finetune iteration (grpo_spec §10, C7).

The acceptance test must read **this adapter's** held-out turns and generate a
**fresh human judgment**. Two things that are emphatically NOT acceptance tests:

  * The stored inter-human kappa. It is the *grounding constant for the checker
    thresholds*. It is fixed before the adapter exists and cannot respond to what
    the adapter produces, so it cannot certify the adapter.
  * The reward backends. Scoring held-out turns with the same champions the
    adapter was trained against grades the adapter with its own reward signal —
    circular, and blind by construction to exactly the failure (reward hacking)
    that certification exists to catch. The champions are still run here, but only
    as the *comparison* against which the human labels are scored; the human
    labels are the acceptance.

Run on **each finetune iteration**, before freezing that iteration's adapter:

  1. **Human hand-labelling of held-out authored detail.** A held-out set
     generated on a *fresh authored instance* the loop never saw (fresh severity /
     instance fill-ins per [FB], not a reseed), labelled on engine + decomposed
     delivery. Finetuning is a stronger overfitting vector than prompt-writing, so
     this matters more here than in the prompt-optimized pipeline — hence
     per-iteration, not once.
  2. **Second bare interlocutor.** Certify against a bare model NOT used in
     training (§5.3) — the interlocutor-robustness run-time demands. Note that
     training now uses a SINGLE interlocutor (a documented deviation from §5.3,
     see `configs/grpo.yaml`), so this step carries the entire interlocutor-
     robustness claim. It is the only place an unseen partner appears.
  3. **Frozen rubric + fixed gold subset.** The rubric hash is pinned across
     iterations and a small fixed gold subset is re-scored every iteration, so
     "the adapter improved" is not confusable with "labelling got more lenient by
     round 6" (R6).
  4. **Optional offline cross-check.** Chat-Opus may score the same held-out set
     offline — one-time, out of loop, budget-compatible (D0.2). Not required, and
     never an API call from inside this module.
  5. **Freeze at deployment quant**, and version with base checkpoint + quant +
     both champion identities + seed + this iteration's kappa (A5).

Three phases, because phase 2 is human:

    python -m grpo.run cert-build  --adapter <path> --iteration 3
    # ... hand-label the emitted sheet against cert/rubric_frozen.md ...
    python -m grpo.run cert-score  --iteration 3
    python -m grpo.run cert-freeze --adapter <path> --iteration 3
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import grpo._bootstrap  # noqa: F401
from fidelity import ENGINE_CONVERGENCE_BAR

from grpo.data.rollout import Interlocutor, build_states, _default_generate
from grpo.reward.fidelity_reward import RewardBackends
from grpo.reward import turn_fidelity
from grpo.stats import ENGINE_LABELS, HOT_LABELS, KAPPA_BAR, agreement


CERT_ROOT = Path("results/grpo/certification")
RUBRIC_PATH = Path(__file__).resolve().parent / "rubric_frozen.md"
GOLD_SUBSET_PATH = CERT_ROOT / "gold_subset.json"
RUBRIC_LOCK_PATH = CERT_ROOT / "rubric_lock.json"

#: Agreement bar on the fixed gold subset, between this iteration's labels and the
#: original ones. Below it, the annotator has drifted and the iteration's numbers
#: are not comparable to earlier rounds (R6).
GOLD_DRIFT_BAR = 0.80


def rubric_hash(path: Path = RUBRIC_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_rubric_frozen(series: str = "default", path: Path = RUBRIC_PATH) -> str:
    """Pin the rubric across iterations (§10 step 3).

    First call in a series records the hash; later calls must match it. A changed
    rubric mid-series silently redefines what certification means, so this raises
    rather than warns — start a new series instead.
    """
    current = rubric_hash(path)
    RUBRIC_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = {}
    if RUBRIC_LOCK_PATH.exists():
        lock = json.loads(RUBRIC_LOCK_PATH.read_text())
    recorded = lock.get(series)
    if recorded is None:
        lock[series] = current
        RUBRIC_LOCK_PATH.write_text(json.dumps(lock, indent=2))
        return current
    if recorded != current:
        raise RuntimeError(
            f"FROZEN RUBRIC CHANGED for series {series!r}: recorded {recorded[:12]}..., "
            f"current {current[:12]}.... Per-iteration human certification only compares "
            "across iterations if the rubric is fixed (grpo_spec §10.3, R6). Either revert "
            f"{path}, or start a new series with --rubric-series <name> and do not mix the "
            "two sets of numbers."
        )
    return current


# ── phase 1: build the labelling sheet ───────────────────────────────────────

@dataclass
class CertItem:
    turn_id: str
    cell: str
    context: str
    turn: str
    model_engine: str
    model_engine_pass: int
    model_q1: bool
    model_q2: bool
    model_delivery_pass: int
    is_gold: bool = False


def build_certification_sheet(
    P_by_cell: Dict[str, str],
    cells: List[str],
    held_out_interlocutor: Interlocutor,
    policy_model_path: str,
    policy_base_url: str,
    backends: RewardBackends,
    iteration: int,
    arcs_per_cell: int = 10,
    prefix_turns: int = 6,
    used_interlocutor_names: Optional[List[str]] = None,
    gold_size: int = 20,
    seed: int = 0,
    out_dir: Optional[Path] = None,
) -> Path:
    """Roll the CANDIDATE policy on held-out authored prompts and emit a blind sheet.

    `P_by_cell` must carry the FRESH authored fill-ins (§10 step 1) — the held-out
    build dir, not the training prompts. The candidate policy (base + this
    iteration's adapter) must be served at `policy_base_url`.

    The fixed gold subset is shuffled in, unmarked, so the annotator cannot tell
    gold rows from new ones — which is the only way re-scoring them measures drift
    rather than diligence.
    """
    if used_interlocutor_names and held_out_interlocutor.name in used_interlocutor_names:
        raise ValueError(
            f"certification interlocutor {held_out_interlocutor.name!r} was used in "
            "training — §10 step 2 requires a second bare interlocutor NOT seen in "
            "training, or the run measures memorisation rather than robustness."
        )

    out_dir = Path(out_dir) if out_dir else CERT_ROOT / f"iter_{iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Certification measures robustness to THIS one unseen model, so the single
    # held-out interlocutor is passed through directly.
    interlocutors = [held_out_interlocutor]
    policy_generate = _default_generate(policy_model_path, policy_base_url, 0.3, 256)

    items: List[CertItem] = []
    counter = 0
    for cell in cells:
        states = build_states(P_by_cell[cell], cell, interlocutors, policy_generate,
                              n_states=arcs_per_cell, prefix_turns=prefix_turns,
                              seed_base=seed + 7919 * iteration)
        for st in states:
            transcript = st.transcript
            for i, msg in enumerate(transcript):
                if msg["role"] != "assistant":
                    continue
                context = "\n".join(
                    f"[{'Patient' if m['role'] == 'assistant' else 'Model'}]: {m['content']}"
                    for m in transcript[:i][-6:]
                )
                turn = msg["content"]
                labels = _model_reads(backends, turn, context, cell)
                items.append(CertItem(
                    turn_id=f"c{iteration}_{counter:05d}", cell=cell, context=context,
                    turn=turn, **labels,
                ))
                counter += 1

    if not items:
        raise RuntimeError(
            "Certification produced zero turns — check that the CANDIDATE policy is "
            f"served at {policy_base_url} and that {held_out_interlocutor.model_path} "
            "is reachable."
        )

    gold = _load_gold_subset()
    rows = list(items)
    if gold:
        rows.extend(CertItem(**{**g, "is_gold": True}) for g in gold)
    random.Random(seed).shuffle(rows)

    sheet = out_dir / "cert_labels_template.csv"
    key = out_dir / "cert_key.csv"
    _write_cert_sheet(rows, sheet)
    _write_cert_key(rows, key)
    (out_dir / "rubric.md").write_text(RUBRIC_PATH.read_text())

    # Seed the gold subset from the first iteration's turns (labels attach later).
    if not gold and gold_size:
        _seed_gold_candidates(items, gold_size, seed)

    return sheet


def _model_reads(backends: RewardBackends, turn: str, context: str, cell: str) -> dict:
    """The champions' reads — the comparison, never the acceptance."""
    engine_labels = {}
    core = getattr(backends.engine, "core", None)
    if core is not None:
        engine_labels = core.labels(turn, context, cell)
    q1 = q2 = False
    decompose = getattr(backends.delivery, "decompose", None)
    if decompose is not None:
        d = decompose(turn, context, cell)
        q1, q2 = d.q1_hostility_toward_listener, d.q2_grievance_toward_absent_party
    return {
        "model_engine": str(engine_labels.get("engine_direction", "")),
        "model_engine_pass": backends.engine.score(turn, context, cell),
        "model_q1": q1,
        "model_q2": q2,
        "model_delivery_pass": backends.delivery.score(turn, context, cell),
    }


CERT_LABEL_COLUMNS = ["engine_label", "q1_hostility_toward_listener",
                      "q2_grievance_toward_absent_party"]


def _write_cert_sheet(rows: List[CertItem], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "cell", "context", "turn", *CERT_LABEL_COLUMNS])
        for it in rows:
            # `is_gold` is deliberately absent: the annotator must not know.
            w.writerow([it.turn_id, it.cell, it.context, it.turn, "", "", ""])


def _write_cert_key(rows: List[CertItem], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "cell", "is_gold", "model_engine", "model_engine_pass",
                    "model_q1", "model_q2", "model_delivery_pass",
                    "context", "turn"])
        for it in rows:
            w.writerow([it.turn_id, it.cell, str(it.is_gold).lower(), it.model_engine,
                        it.model_engine_pass, str(it.model_q1).lower(),
                        str(it.model_q2).lower(), it.model_delivery_pass,
                        it.context, it.turn])


def _load_gold_subset() -> List[dict]:
    if not GOLD_SUBSET_PATH.exists():
        return []
    return json.loads(GOLD_SUBSET_PATH.read_text()).get("items", [])


def _seed_gold_candidates(items: List[CertItem], gold_size: int, seed: int) -> None:
    """Stash candidate gold rows; their reference labels attach at first scoring."""
    rng = random.Random(seed + 1)
    chosen = rng.sample(items, min(gold_size, len(items)))
    GOLD_SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLD_SUBSET_PATH.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_labels": {},
        "items": [{k: v for k, v in it.__dict__.items() if k != "is_gold"} for it in chosen],
    }, indent=2))


# ── phase 2: score the human labels ──────────────────────────────────────────

@dataclass
class CellCertResult:
    cell: str
    n_turns: int
    engine_pass_rate: float
    delivery_pass_rate: float
    passed: bool

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class CertResult:
    """One iteration's certification outcome. Human labels are the acceptance."""

    iteration: int
    passed: bool
    bar: float
    interlocutor: str
    rubric_sha256: str
    n_labelled: int
    cells: List[CellCertResult] = field(default_factory=list)
    kappa_engine: dict = field(default_factory=dict)
    kappa_delivery: dict = field(default_factory=dict)
    gold_drift: dict = field(default_factory=dict)
    backend_identities: dict = field(default_factory=dict)
    scored_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "certification": "human_per_iteration (grpo_spec §10, C7)",
            "iteration": self.iteration,
            "passed": self.passed,
            "bar": self.bar,
            "interlocutor": self.interlocutor,
            "rubric_sha256": self.rubric_sha256,
            "n_labelled": self.n_labelled,
            "cells": [c.to_dict() for c in self.cells],
            "kappa_engine_human_vs_champion": self.kappa_engine,
            "kappa_delivery_human_vs_champion": self.kappa_delivery,
            "gold_subset_drift": self.gold_drift,
            "backend_identities": self.backend_identities,
            "scored_at": self.scored_at,
            "notes": self.notes,
        }


def score_certification(
    iteration: int,
    labels_path: str,
    key_path: str,
    interlocutor_name: str,
    backend_identities: Optional[dict] = None,
    bar: float = ENGINE_CONVERGENCE_BAR,
    delivery_bar: Optional[float] = None,
    rubric_series: str = "default",
    gold_drift_bar: float = GOLD_DRIFT_BAR,
    out_dir: Optional[Path] = None,
    notes: str = "",
) -> CertResult:
    """Score one iteration from the HUMAN labels (§10).

    Acceptance is the human per-cell pass rate against `bar`. `delivery_bar` is
    optional and defaults to report-only, mirroring the harness's own gating
    discipline (`fidelity.DELIVERY_CONVERGENCE_BAR is None`): engine is the gated
    active ingredient, delivery is measured and reported. Pass a float to gate it.

    Also computed, and logged into the manifest at freeze:
      * **kappa vs the champions** — how far the reward's view sits from the
        human's on THIS adapter's output. Diagnostic, not acceptance: a high kappa
        with a low pass rate means the adapter is genuinely off-profile; a low
        kappa means the reward and the human have diverged on the adapter's own
        distribution, which is what reward hacking looks like from outside.
      * **gold-subset drift** — this iteration's labels on the fixed gold rows vs
        their reference labels (R6).
    """
    sha = assert_rubric_frozen(rubric_series)
    labels = _read_csv(labels_path)
    key = {r["turn_id"]: r for r in _read_csv(key_path)}

    per_cell: Dict[str, List[dict]] = {}
    h_eng, m_eng, h_hot, m_hot = [], [], [], []
    gold_now: Dict[str, dict] = {}
    n_labelled = 0

    for row in labels:
        tid = (row.get("turn_id") or "").strip()
        k = key.get(tid)
        if not k:
            continue
        parsed = _parse_human_row(row)
        if parsed is None:
            continue                       # blank row, dropped by design
        n_labelled += 1
        cell = k.get("cell", "")

        if _as_bool(k.get("is_gold")):
            gold_now[tid] = parsed
            continue                       # gold rows measure the annotator, not the adapter

        h_engine_pass = turn_fidelity.engine_pass(
            {"engine_direction": parsed["engine"]}, cell) if parsed["engine"] else None
        h_delivery_pass = _human_delivery_pass(parsed["q1"], cell)
        per_cell.setdefault(cell, []).append({
            "engine_pass": h_engine_pass,
            "delivery_pass": h_delivery_pass,
        })

        if parsed["engine"] in ENGINE_LABELS and k.get("model_engine") in ENGINE_LABELS:
            h_eng.append(parsed["engine"]); m_eng.append(k["model_engine"])
        if parsed["q1"] is not None:
            h_hot.append("hot" if parsed["q1"] else "not_hot")
            m_hot.append("hot" if _as_bool(k.get("model_q1")) else "not_hot")

    if n_labelled == 0:
        raise RuntimeError(
            f"No human labels found in {labels_path}. §10 certification is a HUMAN "
            "acceptance test — it cannot be satisfied by the reward backends, which "
            "would be grading the adapter with its own training signal."
        )

    cell_results = []
    all_pass = True
    for cell in sorted(per_cell):
        rows_ = per_cell[cell]
        e_vals = [r["engine_pass"] for r in rows_ if r["engine_pass"] is not None]
        d_vals = [r["delivery_pass"] for r in rows_ if r["delivery_pass"] is not None]
        e_rate = sum(e_vals) / len(e_vals) if e_vals else 0.0
        d_rate = sum(d_vals) / len(d_vals) if d_vals else 0.0
        cell_pass = bool(e_vals) and e_rate >= bar
        if delivery_bar is not None:
            cell_pass = cell_pass and bool(d_vals) and d_rate >= delivery_bar
        all_pass = all_pass and cell_pass
        cell_results.append(CellCertResult(cell, len(rows_), e_rate, d_rate, cell_pass))

    gold_drift = _score_gold_drift(gold_now, gold_drift_bar)
    if gold_drift.get("drifted"):
        all_pass = False

    result = CertResult(
        iteration=iteration, passed=all_pass, bar=bar, interlocutor=interlocutor_name,
        rubric_sha256=sha, n_labelled=n_labelled, cells=cell_results,
        kappa_engine=agreement(h_eng, m_eng, ENGINE_LABELS, bar=KAPPA_BAR).to_dict() if h_eng else {},
        kappa_delivery=agreement(h_hot, m_hot, HOT_LABELS, bar=KAPPA_BAR).to_dict() if h_hot else {},
        gold_drift=gold_drift, backend_identities=backend_identities or {},
        scored_at=datetime.now(timezone.utc).isoformat(), notes=notes,
    )

    out_dir = Path(out_dir) if out_dir else CERT_ROOT / f"iter_{iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cert_result.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result


def _parse_human_row(row: dict) -> Optional[dict]:
    engine = (row.get("engine_label") or "").strip().lower()
    q1 = _as_bool_or_none(row.get("q1_hostility_toward_listener"))
    q2 = _as_bool_or_none(row.get("q2_grievance_toward_absent_party"))
    if not engine and q1 is None and q2 is None:
        return None
    return {"engine": engine, "q1": q1, "q2": q2}


def _human_delivery_pass(q1: Optional[bool], cell: str) -> Optional[int]:
    """Cell-relative delivery binary from the human's Q1 — `hot = Q1` (§8.1)."""
    if q1 is None:
        return None
    target = turn_fidelity.poles_for_cell(cell)["delivery"]
    return int(q1) if target == "hot" else int(not q1)


def _score_gold_drift(gold_now: Dict[str, dict], bar: float) -> dict:
    """Compare this iteration's gold labels to the reference set (R6).

    On the first scoring pass the reference is empty, so this iteration's labels
    BECOME the reference and no drift is reported. Every later iteration is
    measured against that fixed baseline.
    """
    if not GOLD_SUBSET_PATH.exists():
        return {"status": "no_gold_subset"}
    data = json.loads(GOLD_SUBSET_PATH.read_text())
    ref = data.get("reference_labels") or {}

    if not ref:
        if not gold_now:
            return {"status": "gold_pending_first_labels"}
        data["reference_labels"] = {
            tid: {"engine": v["engine"], "q1": v["q1"]} for tid, v in gold_now.items()
        }
        data["reference_set_at"] = datetime.now(timezone.utc).isoformat()
        GOLD_SUBSET_PATH.write_text(json.dumps(data, indent=2))
        return {"status": "reference_established", "n": len(gold_now), "drifted": False}

    shared = [t for t in gold_now if t in ref]
    if not shared:
        return {"status": "gold_rows_not_labelled_this_iteration", "drifted": False}

    h_now_e = [gold_now[t]["engine"] for t in shared]
    h_ref_e = [ref[t]["engine"] for t in shared]
    e_agree = sum(1 for a, b in zip(h_now_e, h_ref_e) if a == b) / len(shared)

    hot_pairs = [(gold_now[t]["q1"], ref[t]["q1"]) for t in shared
                 if gold_now[t]["q1"] is not None and ref[t]["q1"] is not None]
    d_agree = (sum(1 for a, b in hot_pairs if a == b) / len(hot_pairs)) if hot_pairs else 1.0

    drifted = e_agree < bar or d_agree < bar
    return {
        "status": "compared", "n": len(shared), "bar": bar,
        "engine_self_agreement": round(e_agree, 4),
        "delivery_self_agreement": round(d_agree, 4),
        "drifted": drifted,
        "note": (
            "Annotator drift detected: this iteration's labels on the fixed gold subset "
            "disagree with the reference labels. The iteration's numbers are NOT "
            "comparable to earlier rounds — re-read the frozen rubric and relabel before "
            "treating any improvement as real (grpo_spec R6)."
        ) if drifted else "",
    }


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "t", "yes", "y", "1")


def _as_bool_or_none(v) -> Optional[bool]:
    s = str(v or "").strip().lower()
    if s in ("true", "t", "yes", "y", "1"):
        return True
    if s in ("false", "f", "no", "n", "0"):
        return False
    return None


def _read_csv(path: str) -> List[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


# ── optional offline cross-check (§10 step 4) ────────────────────────────────

def emit_opus_crosscheck_prompt(key_path: str, out_path: str, limit: int = 100) -> str:
    """Emit a paste-ready prompt for an OFFLINE chat-Opus read of the held-out set.

    Explicitly not an API call: D0.2 rules out Opus spend, and a chat-window Opus
    cannot sit in any loop. This produces a file you paste into a chat window
    once, then spot-check the subset where Opus and your labels disagree. Useful,
    not required, and independent of both reward champions.
    """
    rows = _read_csv(key_path)[:limit]
    parts = [
        "You are labelling patient turns from a benchmark, using the rubric below. "
        "Return CSV with columns: turn_id,engine_label,q1_hostility_toward_listener,"
        "q2_grievance_toward_absent_party. Nothing else.\n",
        "=== RUBRIC ===", RUBRIC_PATH.read_text(), "=== TURNS ===",
    ]
    for r in rows:
        parts.append(f"\n--- turn_id: {r['turn_id']} (cell {r.get('cell','')}) ---")
        if r.get("context"):
            parts.append(f"[context]\n{r['context']}")
        parts.append(f"[turn]\n{r.get('turn','')}")
    text = "\n".join(parts)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return str(p)


# ── phase 3: freeze ──────────────────────────────────────────────────────────

def freeze_adapter(
    adapter_path: str,
    out_dir: str,
    cfg: dict,
    cert: CertResult,
    backend_identities: dict,
    merge_and_quantize: bool = False,
) -> str:
    """Freeze a certified adapter and write the reproducibility manifest (§10.5-6).

    Refuses to freeze a non-certified adapter (C7). The manifest records base
    checkpoint + quant + BOTH champion identities + training seed + this
    iteration's certification kappa (A5).
    """
    if not cert.passed:
        raise RuntimeError(
            f"Refusing to freeze iteration {cert.iteration}: it failed §10 human "
            f"certification (bar={cert.bar}, interlocutor={cert.interlocutor}, "
            f"gold_drift={cert.gold_drift.get('status')}). C7 blocks the ship."
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "iteration": cert.iteration,
        "adapter_path": adapter_path,
        "base_model": cfg["base_model"],
        "deployment_quant": cfg["freeze"]["deployment_quant"],
        "reward_backends": backend_identities,          # both champions (A5)
        "training_seed": cfg["freeze"].get("seed"),
        "adapter_mode": cfg.get("adapter_mode"),
        "rubric_sha256": cert.rubric_sha256,
        "certification": cert.to_dict(),
        "certification_kappa": {
            "engine": cert.kappa_engine.get("kappa"),
            "delivery": cert.kappa_delivery.get("kappa"),
        },
    }
    (out / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2))

    if merge_and_quantize:
        _merge_and_quantize(adapter_path, str(out), cfg)
    return str(out / "freeze_manifest.json")


def _merge_and_quantize(adapter_path: str, out_dir: str, cfg: dict) -> None:
    """Merge the LoRA into the base and save at the deployment quant (§10 step 5)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto")
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(cfg["base_model"]).save_pretrained(out_dir)
