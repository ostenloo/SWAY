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
    """Construct the reward from the two local champions (grpo_spec §4, D0.2).

    Engine and delivery come from *different* local models, so their blind spots
    are axis-local and uncorrelated. There is no `backend: opus` switch: D0.2
    rules out API spend, and a chat-window Opus cannot sit in the gradient loop.
    C8 (family-disjointness from `base_model`) is asserted inside
    `build_champion_backends`.
    """
    from grpo.reward.backends import build_champion_backends

    reward = cfg["reward"]
    return build_champion_backends(
        engine_model=reward["backend_engine"],
        delivery_model=reward["backend_delivery"],
        base_url=reward["base_url"],
        base_model=cfg["base_model"],
    )


def reward_identities(cfg: dict) -> dict:
    """Both grader identities, without constructing the backends."""
    r = cfg["reward"]
    return {
        "engine": f"local:{r['backend_engine']}",
        "delivery": f"local:{r['backend_delivery']}",
    }


def build_interlocutors(cfg: dict):
    from grpo.data.rollout import Interlocutor
    return [Interlocutor(**spec) for spec in cfg["interlocutors"]]


def build_policy_generate(cfg: dict, max_tokens: int = 256):
    """A GenerateFn over the `policy` endpoint — the Simulator being rolled for
    warm-start arcs and GRPO history prefixes (distinct from the reward annotator)."""
    from grpo.data.rollout import _default_generate
    p = cfg["policy"]
    return _default_generate(p["model_path"], p["base_url"], p.get("temperature", 0.8), max_tokens)


def build_cert_interlocutor(cfg: dict):
    from grpo.data.rollout import Interlocutor
    return Interlocutor(**cfg["certification"]["interlocutor"])


def load_profile_prompts(cells, build_dir: Optional[str] = None) -> dict[str, str]:
    """Load frozen per-cell profile prompts (P_by_cell) from `<build_dir>/<cell>_prompt.txt`.

    Defaults to the harness build output (results/build/). Pass the held-out
    build dir for certification (§10 step 1 — fresh authored prompts)."""
    from pathlib import Path as _Path
    from config import BUILD_OUTPUT  # harness config (sway_harness on sys.path)

    base = _Path(build_dir) if build_dir else BUILD_OUTPUT
    prompts = {}
    for cell in cells:
        fp = base / f"{cell}_prompt.txt"
        if not fp.exists():
            raise FileNotFoundError(f"missing profile prompt for {cell}: {fp}")
        prompts[cell] = fp.read_text()
    return prompts
