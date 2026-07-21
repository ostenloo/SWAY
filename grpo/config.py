"""Config loading for the GRPO pipeline (grpo_spec §11).

Thin loader over configs/grpo.yaml. PyYAML is imported lazily so the reward,
rollout, probe, monitor, and cert modules (and their tests) stay importable with
only `requests` + stdlib installed — the YAML dependency is only needed when
actually launching a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import grpo._bootstrap  # noqa: F401

GRPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = GRPO_ROOT / "configs" / "grpo.yaml"


def load_config(path: Optional[str] = None) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to load the GRPO config. `pip install pyyaml` "
            "(only needed to launch a run, not to import the reward modules)."
        ) from e
    p = Path(path) if path else DEFAULT_CONFIG
    with p.open() as f:
        return yaml.safe_load(f)


def build_reward_backends(cfg: dict):
    """Construct the reward backends selected in the config (grpo_spec §4/D0.2)."""
    from grpo.reward.backends import build_local_backends, build_opus_backends

    reward = cfg["reward"]
    if reward.get("backend", "local") == "opus":
        return build_opus_backends(reward.get("opus_model"))
    return build_local_backends(
        model_path=reward["local_model_path"],
        base_url=reward["local_base_url"],
    )


def build_interlocutors(cfg: dict):
    from grpo.data.rollout import Interlocutor
    return [Interlocutor(**spec) for spec in cfg["interlocutors"]]
