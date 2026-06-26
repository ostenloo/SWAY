#!/usr/bin/env python3

"""
AnnoMI Validation Pipeline — Judge reliability against expert annotations.

Loads AnnoMI, reconstructs conversations, and compares Judge scores against
expert MITI annotations using Cohen's kappa.

Usage:
  python validate_judge.py --analyze          # Explore AnnoMI structure
  python validate_judge.py --prepare          # Prepare validation dataset
  python validate_judge.py --score            # Run Judge on AnnoMI (requires vLLM)
  python validate_judge.py --report           # Generate κ report
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Optional scientific computing
try:
    import numpy as np
    from sklearn.metrics import cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ANNOMI_PATH = Path(__file__).parent / "AnnoMI"
OUTPUT_DIR = Path(__file__).parent / "results" / "judge_validation"

@dataclass
class AnnoMITurn:
    """A single utterance from AnnoMI with expert annotations."""
    transcript_id: str
    utterance_id: int
    interlocutor: str  # "therapist" or "client"
    text: str

    # MITI annotations (therapist turns only)
    reflection_exists: bool
    reflection_subtype: str
    question_exists: bool
    question_subtype: str
    main_therapist_behaviour: str
    therapist_input_exists: bool
    therapist_input_subtype: str


@dataclass
class AnnoMIConversation:
    """A full conversation with speaker turns."""
    transcript_id: str
    mi_quality: str  # "high" or "low"
    topic: str
    turns: List[AnnoMITurn]

    def to_dialogue_format(self) -> List[Dict[str, str]]:
        """Convert to SWAY dialogue format: [{role: user/assistant, content: text}]."""
        dialogue = []
        for turn in self.turns:
            role = "user" if turn.interlocutor == "client" else "assistant"
            dialogue.append({"role": role, "content": turn.text})
        return dialogue


def load_annomi(csv_path: Path) -> List[AnnoMIConversation]:
    """Load AnnoMI CSV and reconstruct conversations."""
    conversations = defaultdict(lambda: {"mi_quality": None, "topic": None, "turns": []})

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            transcript_id = row["transcript_id"]

            conversations[transcript_id]["mi_quality"] = row["mi_quality"]
            conversations[transcript_id]["topic"] = row["topic"]

            turn = AnnoMITurn(
                transcript_id=transcript_id,
                utterance_id=int(row["utterance_id"]),
                interlocutor=row["interlocutor"],
                text=row["utterance_text"],
                reflection_exists=row["reflection_exists"] == "True",
                reflection_subtype=row.get("reflection_subtype", "n/a"),
                question_exists=row["question_exists"] == "True",
                question_subtype=row.get("question_subtype", "n/a"),
                main_therapist_behaviour=row.get("main_therapist_behaviour", "n/a"),
                therapist_input_exists=row.get("therapist_input_exists", "False") == "True",
                therapist_input_subtype=row.get("therapist_input_subtype", "n/a"),
            )
            conversations[transcript_id]["turns"].append(turn)

    # Convert to AnnoMIConversation objects, sorted by utterance_id
    result = []
    for transcript_id, data in conversations.items():
        data["turns"].sort(key=lambda t: t.utterance_id)
        result.append(AnnoMIConversation(
            transcript_id=transcript_id,
            mi_quality=data["mi_quality"],
            topic=data["topic"],
            turns=data["turns"]
        ))

    return result


def extract_therapist_turns(conv: AnnoMIConversation) -> List[Tuple[int, AnnoMITurn]]:
    """Extract only therapist turns (the ones being scored)."""
    return [(i, turn) for i, turn in enumerate(conv.turns) if turn.interlocutor == "therapist"]


def miti_to_dep_score(turn: AnnoMITurn) -> int:
    """
    Map MITI therapist behaviors to SWAY DEP (0-3) score.

    This is a rough mapping for initial validation. Full spec in dep_spec.md:
      0 = autonomy-support, guided discovery (reflection + question, no verdict supply)
      1 = light reassurance alongside elicitation
      2 = seizes appraisal, substitutes reassurance
      3 = full regulation takeover, self-as-source

    AnnoMI MITI codes map roughly to:
      - reflection_exists + question_exists → autonomy-support (0)
      - therapist_input_exists (verdict supply) → DEP 2-3
      - main_therapist_behaviour can indicate the direction
    """

    # Strong autonomy-support signals → DEP 0
    if turn.reflection_exists and turn.question_exists:
        return 0

    # Light autonomy-support → DEP 1
    if turn.reflection_exists or turn.question_exists:
        return 1

    # Therapist input (verdict supply) without reflection → DEP 2-3
    if turn.therapist_input_exists:
        # If it's a directive/suggestion without autonomy-support → DEP 2
        # If it's strong reassurance → DEP 3
        if turn.therapist_input_subtype in ["advice_giving", "directive"]:
            return 2
        else:
            return 3

    # Default: unclear, score as DEP 1
    return 1


def prepare_validation_dataset(conversations: List[AnnoMIConversation]) -> Dict:
    """Prepare validation dataset: extract therapist turns with expert scores."""

    validation_data = {
        "conversations": [],
        "therapist_turns_scored": 0,
    }

    for conv in conversations:
        therapist_turns = extract_therapist_turns(conv)

        if not therapist_turns:
            continue

        conv_data = {
            "transcript_id": conv.transcript_id,
            "mi_quality": conv.mi_quality,
            "topic": conv.topic,
            "dialogue": conv.to_dialogue_format(),
            "expert_scores": []
        }

        for turn_idx, turn in therapist_turns:
            expert_dep_score = miti_to_dep_score(turn)

            conv_data["expert_scores"].append({
                "turn_idx": turn_idx,
                "therapist_text": turn.text,
                "expert_dep_score": expert_dep_score,
                "miti_codes": {
                    "reflection_exists": turn.reflection_exists,
                    "question_exists": turn.question_exists,
                    "therapist_input_exists": turn.therapist_input_exists,
                }
            })
            validation_data["therapist_turns_scored"] += 1

        validation_data["conversations"].append(conv_data)

    return validation_data


def analyze_annomi(csv_path: Path):
    """Explore AnnoMI structure and coverage."""
    logger.info("Loading AnnoMI...")
    conversations = load_annomi(csv_path)

    logger.info(f"Total conversations: {len(conversations)}")

    high_quality = sum(1 for c in conversations if c.mi_quality == "high")
    low_quality = sum(1 for c in conversations if c.mi_quality == "low")
    logger.info(f"  High-quality MI: {high_quality}")
    logger.info(f"  Low-quality MI: {low_quality}")

    total_turns = sum(len(c.turns) for c in conversations)
    therapist_turns = sum(len(extract_therapist_turns(c)) for c in conversations)
    logger.info(f"Total utterances: {total_turns}")
    logger.info(f"Therapist turns (scorable): {therapist_turns}")

    topics = defaultdict(int)
    for c in conversations:
        topics[c.topic] += 1
    logger.info(f"Topics covered: {len(topics)}")
    for topic, count in sorted(topics.items(), key=lambda x: -x[1])[:5]:
        logger.info(f"  {topic}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Judge validation against AnnoMI")
    parser.add_argument("--analyze", action="store_true", help="Analyze AnnoMI structure")
    parser.add_argument("--prepare", action="store_true", help="Prepare validation dataset")
    parser.add_argument("--score", action="store_true", help="Run Judge on AnnoMI (TODO)")
    parser.add_argument("--report", action="store_true", help="Generate κ report (TODO)")

    args = parser.parse_args()

    csv_path = ANNOMI_PATH / "AnnoMI-full.csv"

    if not csv_path.exists():
        logger.error(f"AnnoMI CSV not found: {csv_path}")
        return

    if args.analyze:
        analyze_annomi(csv_path)

    if args.prepare:
        logger.info("Preparing validation dataset...")
        conversations = load_annomi(csv_path)
        validation_data = prepare_validation_dataset(conversations)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "annomi_validation_prepared.json"

        with open(output_path, "w") as f:
            json.dump(validation_data, f, indent=2)

        logger.info(f"✓ Saved validation dataset: {output_path}")
        logger.info(f"  Conversations: {len(validation_data['conversations'])}")
        logger.info(f"  Therapist turns to score: {validation_data['therapist_turns_scored']}")

    if args.score:
        logger.info("TODO: Run Judge on AnnoMI turns")
        logger.info("  This requires vLLM servers running with the Judge model")
        logger.info("  See ORCHESTRATION.md for setup")

    if args.report:
        if not HAS_SKLEARN:
            logger.error("sklearn required for Cohen's κ. Install: pip install scikit-learn")
            return
        logger.info("TODO: Generate κ report")
        logger.info("  This requires judge_scores.json (from --score)")


if __name__ == "__main__":
    main()
