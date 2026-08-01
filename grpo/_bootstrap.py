"""Path bootstrap so the GRPO package can reuse the existing SWAY harness.

The harness modules (client.py, config.py, parser.py, fidelity.py, build.py) use
FLAT imports (`from client import ...`), i.e. they assume `sway_harness/` is on
sys.path rather than being an installed package. Importing any of them from the
`grpo` package therefore needs that directory on the path first.

The same applies to `tools/` — grpo_spec §8.2 requires the kappa + bootstrap-CI
machinery to be REUSED rather than reimplemented, and `tools/compute_kappa.py`
is likewise a flat module.

Import this module (`import grpo._bootstrap`) before importing any harness module.
It is idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_HARNESS = _ROOT / "sway_harness"
_TOOLS = _ROOT / "tools"

for _d in (_HARNESS, _TOOLS):
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
