"""Path bootstrap so the GRPO package can reuse the existing SWAY harness.

The harness modules (client.py, config.py, parser.py, fidelity.py, build.py) use
FLAT imports (`from client import ...`), i.e. they assume `sway_harness/` is on
sys.path rather than being an installed package. Importing any of them from the
`grpo` package therefore needs that directory on the path first.

Import this module (`import grpo._bootstrap`) before importing any harness module.
It is idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HARNESS = Path(__file__).resolve().parent.parent / "sway_harness"

if _HARNESS.is_dir() and str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))
