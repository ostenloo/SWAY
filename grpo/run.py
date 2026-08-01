#!/usr/bin/env python3
"""End-to-end GRPO driver (grpo_spec) — the glue from config to a launched run.

Pipeline:

    preflight-build -> [human labels] -> preflight-score   (C6-i, §0.1)
    smoketest                                              (optional, §8.3)
    warmstart                                              (§6)
    grpo                                                   (§7, asserts C6-i)
    cert-build      -> [human labels] -> cert-score        (§10, per iteration)
    cert-freeze                                            (C7)

Two steps are human by design and cannot be automated away: the §0.1 fork and
the §10 certification. Each emits a blind labelling sheet and ingests it back.

§8.2's stratified delivery gate has been REMOVED by researcher decision (it
worked by oversampling specific contested cases), so C6 is now the §0.1
pre-flight alone.

`preflight-*`, `smoketest`, `reward-sweep` and `cert-score` run on requests +
pyyaml + tools/requirements.txt; `warmstart` / `grpo` / `cert-freeze --merge`
pull in the GPU stack (lazily). Profile prompts load from results/build/ by
default; certification loads from the held-out build dir in the config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import grpo._bootstrap  # noqa: F401
from grpo.config import (
    load_config, build_reward_backends, build_cert_interlocutor,
    build_interlocutors, load_profile_prompts, reward_identities,
)


GATES_DIR = Path("results/grpo/gates")


# ── C6-i: §0.1 pre-flight non-convergence diagnostic ────────────────────────

def cmd_preflight_build(cfg, args) -> int:
    from grpo.gates.preflight_nonconvergence import build_diagnostic_sheet

    labels = args.out or str(GATES_DIR / "preflight_labels_template.csv")
    key = args.key or str(GATES_DIR / "preflight_key.csv")
    items = build_diagnostic_sheet(
        artifacts_dir=args.artifacts, cells=cfg["cells"],
        out_labels=labels, out_key=key, per_cell=args.per_cell, seed=args.seed,
    )
    print(f"✓ {len(items)} checker-flagged-off turns written to {labels}")
    print("  Hand-label engine_label + delivery_label, then run:\n"
          f"    python -m grpo.run preflight-score --labels {labels} "
          "--signed-off-by '<your name>'")
    return 0


def cmd_preflight_score(cfg, args) -> int:
    from grpo.gates.preflight_nonconvergence import score_diagnostic, VERDICT_POLICY

    key = args.key or str(GATES_DIR / "preflight_key.csv")
    result = score_diagnostic(
        labels_path=args.labels, key_path=key,
        signed_off_by=args.signed_off_by, notes=args.notes or "",
    )
    print(json.dumps(result.to_dict(), indent=2))
    if result.verdict != VERDICT_POLICY:
        print(f"\n✗ §0.1 verdict = {result.verdict}. GRPO is NOT cleared to start "
              "(C6-i). See the gate message for the fork.", file=sys.stderr)
        return 1
    if not result.signed_off_by:
        print("\n✗ computed but NOT signed off — pass --signed-off-by.", file=sys.stderr)
        return 1
    print("\n✓ C6-i cleared: policy capability ceiling, GRPO is the right tool.")
    return 0


# ── §8.3 optional smoke test (NOT the blocker) ──────────────────────────────

def cmd_smoketest(cfg, args) -> int:
    from grpo.gates.authored_pairs_smoketest import run_smoketest

    backends = build_reward_backends(cfg)
    result = run_smoketest(backends.delivery, bar=cfg["gate"]["delivery_kappa_bar"])
    print(json.dumps(result.to_dict(), indent=2))
    print("\nNote: §8.3 is an advisory smoke test on hand-authored pairs. It is "
          "not a gate and does not block training.")
    return 0 if result.accuracy >= 0.9 else 1


# ── §6 warm-start ───────────────────────────────────────────────────────────

def cmd_warmstart(cfg, args) -> int:
    from grpo.train.rft_warmstart import run_rft, rft_raw_path

    P = load_profile_prompts(cfg["cells"], args.build_dir)
    out = run_rft(cfg, P, adapter_out=args.out, dataset_out=args.dataset_out)
    print(f"✓ warm-start adapter written: {out}")
    print(f"✓ kept turns (raw): {rft_raw_path(args.dataset_out)}")
    print("  Next: `python -m grpo.run grpo`")
    return 0


# ── §7 GRPO ─────────────────────────────────────────────────────────────────

def cmd_reward_sweep(cfg, args) -> int:
    """Compare candidate reward shapes on real groups — no training, no GPU.

    Every shape is a pure function of the same three binaries, so one scoring
    pass settles the choice for all of them at once.
    """
    from grpo.analysis.reward_shapes import collect_groups, format_report, sweep
    from grpo.config import build_policy_generate

    cells = args.cells or cfg["cells"]
    P = load_profile_prompts(cells, args.build_dir)
    backends = build_reward_backends(cfg)
    groups = collect_groups(
        P_by_cell=P, cells=cells,
        interlocutors=build_interlocutors(cfg),
        policy_generate=build_policy_generate(cfg),
        backends=backends,
        n_states=args.states, group_size=args.group_size,
        prefix_turns=cfg["grpo"].get("prefix_turns", 4),
        seed_base=cfg["freeze"].get("seed", 42),
    )
    result = sweep(groups, out_path=args.out)
    print()
    print(format_report(result))
    print(f"\n✓ full report: {args.out}")
    return 0


def cmd_grpo(cfg, args) -> int:
    from grpo.train.grpo_loop import run_grpo

    P = load_profile_prompts(cfg["cells"], args.build_dir)
    out = run_grpo(cfg, P, adapter_in=args.adapter_in, adapter_out=args.out)
    print(f"✓ GRPO adapter written: {out}")
    return 0


# ── §10 certification (human, per iteration) ────────────────────────────────

def _cert_paths(cfg, iteration: int):
    root = Path("results/grpo/certification") / f"iter_{iteration}"
    return root / "cert_labels_template.csv", root / "cert_key.csv"


def cmd_cert_build(cfg, args) -> int:
    from grpo.cert.certify_and_freeze import build_certification_sheet

    backends = build_reward_backends(cfg)
    certcfg = cfg["certification"]
    # §10 step 1: fresh authored prompts the loop never saw.
    P_heldout = load_profile_prompts(cfg["cells"], certcfg["heldout_build_dir"])
    sheet = build_certification_sheet(
        P_by_cell=P_heldout, cells=cfg["cells"],
        held_out_interlocutor=build_cert_interlocutor(cfg),
        policy_model_path=cfg["policy"]["model_path"],   # serve the CANDIDATE here
        policy_base_url=cfg["policy"]["base_url"],
        backends=backends, iteration=args.iteration,
        arcs_per_cell=certcfg.get("arcs_per_cell", 10),
        prefix_turns=certcfg.get("prefix_turns", 6),
        used_interlocutor_names=[i.name for i in build_interlocutors(cfg)],
        gold_size=certcfg.get("gold_size", 20),
        seed=cfg["freeze"].get("seed", 42),
    )
    print(f"✓ certification sheet: {sheet}")
    print("  Label it against grpo/cert/rubric_frozen.md (the frozen rubric), then:\n"
          f"    python -m grpo.run cert-score --iteration {args.iteration}")
    return 0


def cmd_cert_score(cfg, args) -> int:
    from grpo.cert.certify_and_freeze import score_certification

    certcfg = cfg["certification"]
    default_labels, default_key = _cert_paths(cfg, args.iteration)
    result = score_certification(
        iteration=args.iteration,
        labels_path=args.labels or str(default_labels),
        key_path=args.key or str(default_key),
        interlocutor_name=certcfg["interlocutor"]["name"],
        backend_identities=reward_identities(cfg),
        delivery_bar=certcfg.get("delivery_bar"),
        rubric_series=certcfg.get("rubric_series", "default"),
        gold_drift_bar=certcfg.get("gold_drift_bar", 0.80),
        notes=args.notes or "",
    )
    print(json.dumps(result.to_dict(), indent=2))
    if not result.passed:
        print("\n✗ §10 certification FAILED — not eligible to freeze (C7).", file=sys.stderr)
        return 1
    print(f"\n✓ iteration {args.iteration} certified.")
    return 0


def cmd_cert_freeze(cfg, args) -> int:
    from grpo.cert.certify_and_freeze import CERT_ROOT, CertResult, freeze_adapter

    result_path = CERT_ROOT / f"iter_{args.iteration}" / "cert_result.json"
    if not result_path.exists():
        print(f"✗ no certification result at {result_path} — run cert-score first (C7).",
              file=sys.stderr)
        return 1
    data = json.loads(result_path.read_text())
    if not data.get("passed"):
        print(f"✗ iteration {args.iteration} did not pass §10 certification; "
              "refusing to freeze (C7).", file=sys.stderr)
        return 1

    cert = CertResult(
        iteration=data["iteration"], passed=True, bar=data["bar"],
        interlocutor=data["interlocutor"], rubric_sha256=data["rubric_sha256"],
        n_labelled=data["n_labelled"],
        kappa_engine=data.get("kappa_engine_human_vs_champion", {}),
        kappa_delivery=data.get("kappa_delivery_human_vs_champion", {}),
        gold_drift=data.get("gold_subset_drift", {}),
        backend_identities=data.get("backend_identities", {}),
        scored_at=data.get("scored_at", ""),
    )
    manifest = freeze_adapter(
        adapter_path=args.adapter,
        out_dir=f"{cfg['freeze']['out_dir']}/frozen/iter_{args.iteration}",
        cfg=cfg, cert=cert,
        backend_identities=reward_identities(cfg),
        merge_and_quantize=args.merge,
    )
    print(f"✓ frozen — manifest: {manifest}")
    return 0


def cmd_crosscheck(cfg, args) -> int:
    """§10 step 4 — emit the OFFLINE chat-Opus cross-check prompt (optional)."""
    from grpo.cert.certify_and_freeze import emit_opus_crosscheck_prompt

    _, key = _cert_paths(cfg, args.iteration)
    out = emit_opus_crosscheck_prompt(str(args.key or key), args.out, limit=args.limit)
    print(f"✓ paste-ready offline cross-check prompt: {out}")
    print("  This is NOT an API call and NOT part of the reward (D0.2) — paste it "
          "into a chat window once, then spot-check the disagreements.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SWAY GRPO driver")
    ap.add_argument("--config", default=None, help="path to grpo.yaml (default: configs/grpo.yaml)")
    ap.add_argument("--build-dir", default=None, help="dir with <cell>_prompt.txt (default: results/build)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preflight-build", help="§0.1 — emit the non-convergence diagnostic sheet")
    p.add_argument("--artifacts", default="results/build_artifacts")
    p.add_argument("--per-cell", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--key", default=None)

    p = sub.add_parser("preflight-score", help="§0.1 — score the diagnostic and sign off (C6-i)")
    p.add_argument("--labels", required=True)
    p.add_argument("--key", default=None)
    p.add_argument("--signed-off-by", default="")
    p.add_argument("--notes", default="")

    sub.add_parser("smoketest", help="§8.3 — authored-pair smoke test (NOT the blocker)")

    p = sub.add_parser("warmstart", help="§6 — reward-filtered SFT warm-start")
    p.add_argument("--out", default="results/grpo/adapters/rft")
    p.add_argument("--dataset-out", default="results/grpo/rft_dataset.jsonl")

    p = sub.add_parser("reward-sweep",
                       help="compare reward shapes on real groups (no training, no GPU)")
    p.add_argument("--cells", nargs="*", default=None, help="default: all cells")
    p.add_argument("--states", type=int, default=6, help="states per cell")
    p.add_argument("--group-size", type=int, default=8, help="G completions per state")
    p.add_argument("--out", default="results/grpo/reward_shape_sweep.json")

    p = sub.add_parser("grpo", help="§7 — GRPO on top of the warm start (asserts C6-i)")
    p.add_argument("--adapter-in", default="results/grpo/adapters/rft")
    p.add_argument("--out", default="results/grpo/adapters/grpo")

    p = sub.add_parser("cert-build", help="§10 — emit this iteration's certification sheet")
    p.add_argument("--iteration", type=int, required=True)

    p = sub.add_parser("cert-score", help="§10 — score the human certification labels")
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--key", default=None)
    p.add_argument("--notes", default="")

    p = sub.add_parser("cert-freeze", help="C7 — freeze a certified adapter at deployment quant")
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--merge", action="store_true", help="merge LoRA + save at deployment quant")

    p = sub.add_parser("crosscheck", help="§10.4 — emit the offline chat-Opus cross-check prompt")
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--key", default=None)
    p.add_argument("--out", default="results/grpo/certification/opus_crosscheck_prompt.txt")
    p.add_argument("--limit", type=int, default=100)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    return {
        "preflight-build": cmd_preflight_build,
        "preflight-score": cmd_preflight_score,
        "smoketest": cmd_smoketest,
        "reward-sweep": cmd_reward_sweep,
        "warmstart": cmd_warmstart,
        "grpo": cmd_grpo,
        "cert-build": cmd_cert_build,
        "cert-score": cmd_cert_score,
        "cert-freeze": cmd_cert_freeze,
        "crosscheck": cmd_crosscheck,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
