"""
Profile parser — extracts cell configurations from sway_profile_roster.md and layoff_fact_base.md.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from config import ROOT, PATHS


def _parse_hexaco_block(text: str) -> dict:
    result = {}
    patterns = {
        "H": r"H:\s*(.+?)(?:\s*\(|$)",
        "E": r"E:\s*(.+?)(?:\s*\(|$)",
        "X": r"X:\s*(.+?)(?:\s*\(|$)",
        "A": r"A:\s*(.+?)(?:\s*\(|$)",
        "C": r"C:\s*(.+?)(?:\s*\(|$)",
        "O": r"O:\s*(.+?)(?:\s*\(|$)",
    }
    for trait, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1).strip().rstrip("()").strip()
            val = val.split("(")[0].strip()
            result[trait] = val
    return result


def _parse_attribute_line(text: str) -> dict:
    result = {}
    lines = text.split("\n")
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            result[key] = val
    return result


def parse_cell(section_text: str) -> dict:
    cell = {}
    header_match = re.search(r"###\s*(B\d+|P\d+)\s*—\s*(.+)", section_text)
    if header_match:
        cell["id"] = header_match.group(1).lower()
        cell["name"] = header_match.group(2).strip()

    engine_match = re.search(r"engine:\s*(\S+)", section_text)
    delivery_match = re.search(r"delivery:\s*(\S+)", section_text)
    if engine_match:
        cell["engine"] = engine_match.group(1).strip()
    if delivery_match:
        cell["delivery"] = delivery_match.group(1).strip()

    dist_match = re.search(r"distortion_class:\s*(.+?)\s*\|", section_text)
    drift_match = re.search(r"drift_probed:\s*(\S+)", section_text)
    if dist_match:
        cell["distortion_class"] = dist_match.group(1).strip()
    if drift_match:
        cell["drift_probed"] = drift_match.group(1).strip()

    hexaco = _parse_hexaco_block(section_text)
    if hexaco:
        cell["hexaco"] = hexaco

    if "engine" in cell:
        cell["distortion_direction"] = cell.get("distortion_class", "unknown")

    agg_match = re.search(r"aggression flag (ON|OFF)", section_text)
    if agg_match:
        cell["aggression_flag"] = agg_match.group(1) == "ON"

    return cell


def load_roster() -> dict:
    roster_text = PATHS["roster"].read_text()
    result = {"baseline": {}, "backbone": [], "probes": []}

    # Baseline
    baseline_section = re.search(r"## Realism baseline.*?```(.*?)```", roster_text, re.DOTALL)
    if baseline_section:
        result["baseline"] = _parse_attribute_line(baseline_section.group(1))

    # Backbone (B1-B6)
    backbone_section = re.search(r"# Backbone \(6\)(.*?)(?=\n## )", roster_text, re.DOTALL)
    if backbone_section:
        text = backbone_section.group(1)
        parts = re.split(r"(?=###\s+B\d+)", text)
        for part in parts:
            if not part.strip() or "### B" not in part:
                continue
            cell = parse_cell(part)
            cell["type"] = "backbone"
            result["backbone"].append(cell)

    # Probes (P1-P3)
    probes_section = re.search(r"# Probe add-ons.*?(?=\n## )", roster_text, re.DOTALL)
    if probes_section:
        text = probes_section.group(0)
        parts = re.split(r"(?=###\s+P\d+\s+—)", text)
        for part in parts:
            if not part.strip() or "### P" not in part:
                continue
            cell = parse_cell(part)
            cell["type"] = "probe"
            base_match = re.search(r"\(base:\s*([^\)]+)\)", part)
            if base_match:
                cell["base_cell"] = base_match.group(1).strip()
            # Values may be hyphenated (e.g. "LOSES-THREAD"), so match [\w-]+.
            override_matches = re.findall(r"(\w+):\s*([\w-]+)\s*→\s*([\w-]+)", part)
            for orig_key, old_val, new_val in override_matches:
                cell[f"override_{orig_key.lower()}"] = new_val
            result["probes"].append(cell)

    return result


def load_fact_base() -> dict:
    text = PATHS["fact_base"].read_text()
    result = {
        "facts": [],
        "reservoir": [],
        "bait_map": {"entitlement": [], "dependency": [], "neutral": []},
        "premise": "",
    }

    premise_match = re.search(r"## 1\. Scenario premise.*?\n(.*?)(?=##|---)", text, re.DOTALL)
    if premise_match:
        result["premise"] = premise_match.group(1).strip()

    # Facts F1-F32
    facts_section = re.search(r"## 2\..*?(?:## 2A|## 3)", text, re.DOTALL)
    if facts_section:
        fact_rows = re.findall(r"\|\s*(F\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|", facts_section.group(0))
        for fid, ftex, hook in fact_rows:
            result["facts"].append({"id": fid, "text": ftex.strip(), "hook": hook.strip()})

    # 2A extended facts
    facts_2a = re.search(r"## 2A\..*?(?:## 3|## 4)", text, re.DOTALL)
    if facts_2a:
        fact_rows = re.findall(r"\|\s*(F\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|", facts_2a.group(0))
        for fid, ftex, hook in fact_rows:
            result["facts"].append({"id": fid, "text": ftex.strip(), "hook": hook.strip()})

    # Reservoir R1-R10
    reservoir_section = re.search(r"## 3\..*?(?:## 4|## 5)", text, re.DOTALL)
    if reservoir_section:
        res_rows = re.findall(r"\|\s*(R\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|", reservoir_section.group(0))
        for rid, question, why in res_rows:
            result["reservoir"].append({"id": rid, "question": question.strip(), "why_open": why.strip()})

    # Bait map — entitlement (4.1)
    ent_section = re.search(r"### 4\.1 Entitlement.*?(?:### 4\.2|## 5)", text, re.DOTALL)
    if ent_section:
        bait_rows = re.findall(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|", ent_section.group(0))
        for prop, reality, competent, drift in bait_rows:
            result["bait_map"]["entitlement"].append({
                "proposition": prop.strip(),
                "reality": reality.strip(),
                "competent_stance": competent.strip(),
                "drift": drift.strip(),
            })

    # Bait map — dependency (4.2)
    dep_section = re.search(r"### 4\.2 Dependency.*?(?:## 5)", text, re.DOTALL)
    if dep_section:
        bait_rows = re.findall(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|", dep_section.group(0))
        for prop, reality, competent, drift in bait_rows:
            result["bait_map"]["dependency"].append({
                "proposition": prop.strip(),
                "reality": reality.strip(),
                "competent_stance": competent.strip(),
                "drift": drift.strip(),
            })

    return result


def get_bait_text(engine: str) -> str:
    fact_base = load_fact_base()
    key = engine.lower()
    baits = fact_base["bait_map"].get(key, [])
    lines = []
    for bait in baits:
        lines.append(f"- Patient pushes: {bait['proposition']}")
        lines.append(f"  → Ledger says: {bait['reality']}")
        lines.append(f"  → Competent: {bait['competent_stance']}")
        lines.append("")
    return "\n".join(lines)


def get_profile(cell_id: str) -> dict:
    roster = load_roster()
    cell = None
    for c in roster["backbone"] + roster["probes"]:
        if c.get("id") == cell_id or cell_id in c.get("name", "").lower():
            cell = c
            break

    if not cell:
        name_map = {
            "b1": "dependency-warm",
            "b2": "dependency-hot",
            "b3": "entitlement-warm",
            "b4": "entitlement-hot",
            "b5": "neutral-warm",
            "b6": "neutral-hot",
            "p1": "guarded-distress",
            "p2": "loses-thread",
            "p3": "fluent-but-low-uptake",
        }
        mapped = name_map.get(cell_id.lower())
        if mapped:
            for c in roster["backbone"] + roster["probes"]:
                if mapped in c.get("name", "").lower():
                    cell = c
                    break

    if not cell:
        available = [c.get("name", c.get("id", "?")) for c in roster["backbone"] + roster["probes"]]
        raise ValueError(f"Unknown cell: {cell_id}. Available: {available}")

    profile = dict(roster["baseline"])
    # Probes pin ONE axis off a base cell ("all else inherits B1/B5"). Fold the
    # base cell in first so engine/delivery/comprehension/etc. carry over, THEN
    # apply the probe's own pinned overrides (override_<attr> -> <attr>).
    # Without this a probe silently collapses to baseline+engine=neutral, so its
    # authored manipulation is dropped and it is scored against the wrong poles.
    base_match = re.match(r"\s*(B\d+)", cell.get("base_cell", ""))
    if base_match:
        profile.update(get_profile(base_match.group(1).lower()))
    profile.update(cell)
    for key, val in cell.items():
        if key.startswith("override_"):
            profile[key[len("override_"):]] = val
    profile.setdefault("engine", "neutral")
    profile.setdefault("delivery", "warm")
    return profile


if __name__ == "__main__":
    import json
    roster = load_roster()
    print("Baseline:", json.dumps(roster["baseline"], indent=2))
    print(f"\nBackbone cells: {len(roster['backbone'])}")
    for c in roster["backbone"]:
        print(f"  {c.get('id', '?')}: {c.get('name', '?')}")
    print(f"\nProbe cells: {len(roster['probes'])}")
    for c in roster["probes"]:
        print(f"  {c.get('id', '?')}: {c.get('name', '?')}")

    print("\n--- Fact base ---")
    fb = load_fact_base()
    print(f"Facts: {len(fb['facts'])}")
    print(f"Reservoir: {len(fb['reservoir'])}")
    print(f"Bait entitlement: {len(fb['bait_map']['entitlement'])}")
    print(f"Bait dependency: {len(fb['bait_map']['dependency'])}")

    print("\n--- Profile B4 ---")
    p = get_profile("b4")
    print(json.dumps(p, indent=2))
