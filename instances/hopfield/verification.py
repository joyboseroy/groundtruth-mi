"""
instances/hopfield/verification.py

Verification for the Hopfield instance, per Section 3's requirement that any
instance used as ground truth must first be checked to confirm its actual
(recall-time) behavior matches its specified mechanism, not merely that the
architecture was specified.

Two checks, matching the shorter checklist for Hopfield (no training phase
means no train-time drift is possible, unlike SDM):

  1. Stored patterns are genuine fixed points of the update rule.
  2. Corrupted queries converge to a stored pattern (or its global sign
     inverse, which is a known, expected attractor of the energy function,
     not a spurious one) rather than to an unintended spurious attractor.

Any query whose recall does not converge to a legitimate attractor must be
excluded from the benchmark's ground-truth query set, not silently included,
because for that query "the true circuit" is not well defined.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from build import HopfieldNetwork, corrupt, RecallResult


@dataclass
class FixedPointCheck:
    pattern_index: int
    is_fixed_point: bool
    n_flips_on_direct_recall: int


@dataclass
class SpuriousAttractorCheck:
    trial_index: int
    source_pattern_index: int
    n_flips_in_query: int
    converged: bool
    matches_stored_pattern: bool
    matches_global_sign_inverse: bool
    is_legitimate_attractor: bool  # True if it matches a pattern or its inverse


def verify_fixed_points(net: HopfieldNetwork) -> List[FixedPointCheck]:
    """
    Check 1: every stored pattern should be a fixed point, i.e. recall
    starting exactly at the pattern should produce zero flips.
    """
    results = []
    for idx, p in enumerate(net.stored_patterns):
        recall = net.recall_trace(p.copy())
        results.append(
            FixedPointCheck(
                pattern_index=idx,
                is_fixed_point=(len(recall.trace) == 0),
                n_flips_on_direct_recall=len(recall.trace),
            )
        )
    return results


def _matches(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def verify_no_spurious_attractors(
    net: HopfieldNetwork,
    n_trials_per_pattern: int = 20,
    corruption_levels: Optional[List[int]] = None,
    seed: int = 0,
) -> List[SpuriousAttractorCheck]:
    """
    Check 2: for a range of corruption levels, corrupted queries derived
    from each stored pattern should converge to that pattern or its global
    sign inverse (both are legitimate attractors of a Hopfield energy
    function; the inverse is NOT spurious, it is expected). A query that
    converges anywhere else indicates a spurious attractor at the tested
    corruption level, and must be excluded from the ground-truth query set
    used later in Section 6, since "the true circuit" is undefined for it.
    """
    rng = np.random.default_rng(seed)
    if corruption_levels is None:
        n = net.n_units
        corruption_levels = [max(1, n // 10), max(1, n // 5), max(1, n // 3)]

    results = []
    trial_idx = 0
    for pattern_idx, pattern in enumerate(net.stored_patterns):
        for level in corruption_levels:
            for _ in range(n_trials_per_pattern):
                query = corrupt(pattern, n_flips=level, rng=rng)
                recall = net.recall_trace(query)
                matches_pattern = _matches(recall.final_state, pattern)
                matches_inverse = _matches(recall.final_state, -pattern)
                results.append(
                    SpuriousAttractorCheck(
                        trial_index=trial_idx,
                        source_pattern_index=pattern_idx,
                        n_flips_in_query=level,
                        converged=recall.converged,
                        matches_stored_pattern=matches_pattern,
                        matches_global_sign_inverse=matches_inverse,
                        is_legitimate_attractor=(matches_pattern or matches_inverse),
                    )
                )
                trial_idx += 1
    return results


def summarize(
    fixed_point_results: List[FixedPointCheck],
    spurious_results: List[SpuriousAttractorCheck],
) -> str:
    fp_pass = sum(1 for r in fixed_point_results if r.is_fixed_point)
    fp_total = len(fixed_point_results)

    sa_pass = sum(1 for r in spurious_results if r.is_legitimate_attractor)
    sa_total = len(spurious_results)
    sa_failures = [r for r in spurious_results if not r.is_legitimate_attractor]

    lines = [
        f"Fixed-point check: {fp_pass}/{fp_total} stored patterns are exact fixed points.",
        f"Spurious-attractor check: {sa_pass}/{sa_total} trials converged to a legitimate "
        f"attractor (stored pattern or its global sign inverse).",
    ]
    if fp_pass < fp_total:
        lines.append(
            "WARNING: not all stored patterns are fixed points. This instance cannot be "
            "used as ground truth until this is understood (likely capacity overload if "
            "many patterns are stored relative to n_units)."
        )
    if sa_failures:
        lines.append(
            f"NOTE: {len(sa_failures)} trial(s) converged to a non-legitimate (spurious) "
            f"attractor. These specific queries must be EXCLUDED from the benchmark's "
            f"ground-truth query set in Section 6, not included with an assumed answer."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 20
    net = HopfieldNetwork(n_units=n)
    patterns = [rng.choice([-1, 1], size=n) for _ in range(3)]
    net.store(patterns)

    fp_results = verify_fixed_points(net)
    sa_results = verify_no_spurious_attractors(net, n_trials_per_pattern=10, seed=1)

    print(summarize(fp_results, sa_results))
