"""Agreement statistics for the GRPO gates — a thin reuse layer.

The kappa + bootstrap-CI machinery is *reused* rather than reimplemented, so this
module wraps `tools/compute_kappa.py` (the same code that produced the batch01-03
human-vs-judge reports) instead of carrying a second implementation that could
drift from it. Read by the §0.1 diagnostic and §10 certification.

Requires `tools/requirements.txt` (numpy, pandas, scikit-learn). That dependency
is deliberate: the alternative — a stdlib reimplementation — is exactly the drift
the spec's reuse instruction is guarding against.

Why the CI lower bound rather than the point estimate: the strata these gates
score are small (tens of items), so a point estimate of 0.82 on n=40 routinely
covers a true value well below the bar. Every gate here reads
`bootstrap_ci(...)[0]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import grpo._bootstrap  # noqa: F401  (puts tools/ on sys.path)


HOT_LABELS = ["hot", "not_hot"]
ENGINE_LABELS = ["internalizing", "externalizing", "neutral"]
DELIVERY_LABELS = ["hot", "warm", "flat"]

#: Baig physician-vs-physician bar, the grounding constant for every gate here.
KAPPA_BAR = 0.80


def _machinery():
    """Import the shared kappa machinery, with an actionable error if absent."""
    try:
        from compute_kappa import safe_kappa, gwet_ac1, bootstrap_ci  # tools/
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "The GRPO gates reuse tools/compute_kappa.py for kappa + bootstrap CI "
            "Install its dependencies: "
            "`pip install -r tools/requirements.txt`."
        ) from e
    return safe_kappa, gwet_ac1, bootstrap_ci


@dataclass
class Agreement:
    """Agreement between a human labelling and a model labelling on one sample."""

    n: int
    kappa: float
    kappa_ci_low: float
    kappa_ci_high: float
    ac1: float
    ac1_ci_low: float
    ac1_ci_high: float
    raw_agreement: float
    bar: float
    labels: List[str] = field(default_factory=list)
    confusion: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Judge on the CI LOWER BOUND, never the point estimate."""
        return self.n > 0 and self.kappa_ci_low == self.kappa_ci_low and self.kappa_ci_low >= self.bar

    @property
    def point_estimate_clears(self) -> bool:
        """True when kappa clears the bar but the CI does not — the 'not enough
        data to certify' case, distinct from a genuine failure."""
        return self.kappa >= self.bar and not self.passed

    def to_dict(self) -> dict:
        def _r(x):
            return None if x != x else round(float(x), 4)  # NaN -> None
        return {
            "n": self.n,
            "kappa": _r(self.kappa),
            "kappa_ci": [_r(self.kappa_ci_low), _r(self.kappa_ci_high)],
            "ac1": _r(self.ac1),
            "ac1_ci": [_r(self.ac1_ci_low), _r(self.ac1_ci_high)],
            "raw_agreement": _r(self.raw_agreement),
            "bar": self.bar,
            "passed": self.passed,
            "point_estimate_clears_but_ci_does_not": self.point_estimate_clears,
            "confusion": self.confusion,
        }


def agreement(
    human: Sequence[str],
    model: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    bar: float = KAPPA_BAR,
    n_boot: int = 2000,
    seed: int = 0,
) -> Agreement:
    """Cohen's kappa + Gwet's AC1, each with a bootstrap CI.

    AC1 is reported alongside kappa for the same reason `tools/compute_kappa.py`
    reports both: these label sets are prevalence-skewed (hot turns are rare),
    which is the regime where kappa is depressed by the kappa paradox. Read them
    together — AC1 tells a real disagreement apart from a marginal artefact.
    """
    safe_kappa, gwet_ac1, bootstrap_ci = _machinery()
    import numpy as np

    labels = list(labels or HOT_LABELS)
    h = np.asarray(list(human))
    m = np.asarray(list(model))
    n = len(h)
    if n == 0:
        nan = float("nan")
        return Agreement(0, nan, nan, nan, nan, nan, nan, nan, bar, labels, {})

    k = float(safe_kappa(h, m, labels))
    k_lo, k_hi = bootstrap_ci(h, m, labels, n_boot=n_boot, seed=seed)
    a = float(gwet_ac1(h, m, labels))
    a_lo, a_hi = bootstrap_ci(h, m, labels, stat=gwet_ac1, n_boot=n_boot, seed=seed)
    raw = float((h == m).mean())

    confusion = {}
    for hl in labels:
        for ml in labels:
            count = int(((h == hl) & (m == ml)).sum())
            if count:
                confusion[f"human={hl},model={ml}"] = count

    return Agreement(
        n=n, kappa=k, kappa_ci_low=float(k_lo), kappa_ci_high=float(k_hi),
        ac1=a, ac1_ci_low=float(a_lo), ac1_ci_high=float(a_hi),
        raw_agreement=raw, bar=bar, labels=labels, confusion=confusion,
    )
