"""The v1 engine measurements, transcribed from [RATE §7].

**Why this file exists.** The rate-profile targets in §7 were measured on AnnoMI
with the frozen `command-r7b` grader at temperature 0, but the grader label cache
those percentiles were read off does not live in this repo — the pass ran on the
grading host. Re-running `annomi_calibrate rate-derive` against a restored cache
is the real derivation path and supersedes this file; until then the frozen v1
artifact is built from the numbers the spec records, shaped by the SAME
`rate_derive` code the live path uses, so the two cannot drift in how a band is
formed even though they differ in where the percentiles came from.

Every number below is a transcription. Nothing here is computed, and nothing
here may be edited to "improve" a band — that is a derivation change and belongs
in a re-run, not in a table.

**Two discrepancies in §7 are carried forward as disclosure, not silently fixed:**

1. **The groups overlap.** §7 reports n = 50 / 39 / 26 against "100 conversations
   meet `T >= 10`" — 115 memberships over 100 conversations, an overlap of 15.
   §6 step 4 says "partition", and its predicates are not disjoint as written
   (a conversation at `p_int = 0.08, p_ext = 0.02` is both internalizing-leaning
   and low-rate). `rate_derive.group_conversations` implements the disjoint
   partition with low-rate tested first; the first re-derivation will therefore
   NOT reproduce these edges exactly, and should not be expected to.

2. **§7 flags the §5.2 widening on one band and not the three others like it.**
   `EXTERNALIZING p_int in [0, 0.071]` is marked "widen to >= 0.10 span"; the
   equally sub-span `INTERNALIZING p_ext in [0, 0.089]` and both NEUTRAL bands
   are not. The widening is applied uniformly here and recorded per band.
"""

from __future__ import annotations

from grpo.calibration import rate_derive as R


#: §7, AnnoMI: 133 sessions, 3,221 substantive client turns, graded blind by
#: `command-r7b` at temperature 0. 100 conversations meet `T >= 10`.
#:
#:   group -> {marked label -> (p25, median, p75)}
SPEC_V1_ENGINE_PERCENTILES = {
    "internalizing": {
        "internalizing": (0.144, 0.200, 0.290),
        "externalizing": (0.000, 0.061, 0.089),
    },
    "externalizing": {
        "internalizing": (0.000, 0.059, 0.071),
        "externalizing": (0.106, 0.163, 0.232),
    },
    R.GROUP_LOW_RATE: {
        "internalizing": (0.000, 0.000, 0.077),
        "externalizing": (0.000, 0.019, 0.078),
    },
}

#: §7's reported group sizes. Recorded for provenance including the overlap.
SPEC_V1_GROUP_N = {"internalizing": 50, "externalizing": 39, R.GROUP_LOW_RATE: 26}

SPEC_V1_N_ELIGIBLE = 100
SPEC_V1_N_CONVERSATIONS = 133
SPEC_V1_N_TURNS = 3221
#: §7 names the engine grader as `command-r7b`; a backend's `identity` carries the
#: transport prefix it was served behind (`local:` for the ollama champions), and
#: C3 compares identities, so the full identity is what has to be recorded.
SPEC_V1_GRADER = "local:command-r7b:latest"

SPEC_V1_NOTE = (
    "Transcribed from RATE §7, not re-derived: the grader label cache those percentiles "
    "were read off is not in this repo. Two disclosures travel with these numbers. "
    f"(1) §7's group sizes sum to {sum(SPEC_V1_GROUP_N.values())} over "
    f"{SPEC_V1_N_ELIGIBLE} eligible conversations, so the source run's groups OVERLAP; "
    "§6 step 4 says 'partition' and rate_derive.group_conversations implements the "
    "disjoint version with low-rate tested first, so a re-derivation will shift these "
    "edges. (2) §7 flags the §5.2 widening on EXTERNALIZING p_int in [0, 0.071] but not "
    "on the equally sub-span INTERNALIZING p_ext in [0, 0.089] or either NEUTRAL band; "
    "it is applied uniformly here and recorded per band."
)


def engine_edges_from_spec(target_direction: str, *, T: int) -> list:
    """Both engine components for `target_direction`, from §7's table.

    Signature matches what `rate_derive.build_cells` injects, so the frozen build
    and the live build assemble the same document.
    """
    if target_direction == "neutral":
        group = R.GROUP_LOW_RATE
        roles = {"internalizing": R.OFF_DIRECTION, "externalizing": R.OFF_DIRECTION}
        order = ("internalizing", "externalizing")
    elif target_direction in ("internalizing", "externalizing"):
        group = target_direction
        other = "externalizing" if target_direction == "internalizing" else "internalizing"
        roles = {target_direction: R.ON_DIRECTION, other: R.OFF_DIRECTION}
        order = (target_direction, other)
    else:
        raise ValueError(f"{target_direction!r} is not an engine target")

    table = SPEC_V1_ENGINE_PERCENTILES[group]
    return [
        R.edges_from_percentiles(
            "engine", label, roles[label], group, *table[label], T=T,
            n_group=SPEC_V1_GROUP_N[group],
            n_eligible=SPEC_V1_N_ELIGIBLE,
            n_total=SPEC_V1_N_CONVERSATIONS,
        )
        for label in order
    ]
