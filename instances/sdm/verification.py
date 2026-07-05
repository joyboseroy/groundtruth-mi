"""
instances/sdm/verification.py

Verification for the RankOrderSDM instance, per Section 3's requirement:
specifying an architecture does not guarantee the trained model's actual
behavior matches that specification (dead units, redundant pathways,
numerical drift are all real possibilities). Four checks, matching the
original SDM checklist:

  1. Address decoder activation is deterministic and matches direct
     recomputation from stored weights (no hidden non-determinism in
     top-N_w selection or rank-order re-encoding).
  2. Hard-location occupancy is reported honestly as a descriptive
     statistic, NOT graded pass/fail against 100% coverage. With T
     patterns activating N_w decoders each out of W total, expected
     coverage is roughly T*N_w/W under random hashing, not 100%. A
     failure here means occupancy is far below that expectation, which
     would indicate a real problem (e.g. address decoder weights not
     actually diverse), not merely "some locations never fire."
  3. The MAX-Hebbian data-store update rule is confirmed to actually take
     the max across writes, not silently behave like an additive/summing
     rule.
  4. Rank-order significance weights in re-encoded outputs are confirmed
     to track the ACTUAL magnitude-sorted order of raw activations, not
     the input's original firing order blindly carried through.

Requires the real sdm_library package (sequence-machine-revisited/src) to
be importable. Adjust SDM_REPO_SRC below to your local path, or set the
SDM_REPO_SRC environment variable, before running.
"""

import os
import sys

SDM_REPO_SRC = os.environ.get(
    "SDM_REPO_SRC",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "sequence-machine-revisited", "src"),
)
sys.path.insert(0, SDM_REPO_SRC)

from dataclasses import dataclass
from typing import List
import numpy as np

from sdm_library import RankOrderSDM, make_significance_vectors, topk_indices


# ---------------------------------------------------------------------------
# Check 1: Address decoder determinism and recomputation match
# ---------------------------------------------------------------------------

@dataclass
class DeterminismCheck:
    query_index: int
    matches_on_repeat: bool
    matches_manual_recompute: bool


def verify_address_decoder_determinism(sdm: RankOrderSDM, address_vecs: List[np.ndarray]) -> List[DeterminismCheck]:
    """
    For each query address, run _addr_forward twice (should be identical,
    no hidden randomness) and independently recompute the top-N_w selection
    directly from sdm.W_addr to confirm the internal method matches a
    from-scratch recomputation, not just itself.
    """
    results = []
    for idx, addr in enumerate(address_vecs):
        out1 = sdm._addr_forward(addr)
        out2 = sdm._addr_forward(addr)
        matches_repeat = bool(np.array_equal(out1, out2))

        # Manual recomputation, independent of _addr_forward's internals.
        activations = sdm.W_addr @ addr
        top = topk_indices(activations, sdm.N_w)
        top_sorted = top[np.argsort(activations[top])[::-1]]
        manual_out = np.zeros(sdm.W, dtype=np.float64)
        for rank, i in enumerate(top_sorted):
            manual_out[i] = sdm.alpha ** rank
        matches_manual = bool(np.allclose(out1, manual_out))

        results.append(DeterminismCheck(
            query_index=idx,
            matches_on_repeat=matches_repeat,
            matches_manual_recompute=matches_manual,
        ))
    return results


# ---------------------------------------------------------------------------
# Check 2: Hard-location occupancy (descriptive, not pass/fail against 100%)
# ---------------------------------------------------------------------------

@dataclass
class OccupancyReport:
    n_patterns: int
    n_hard_locations: int
    n_w: int
    naive_expected_fraction: float
    observed_fraction: float
    observed_count: int


def report_hard_location_occupancy(sdm: RankOrderSDM, address_vecs: List[np.ndarray]) -> OccupancyReport:
    touched = set()
    for addr in address_vecs:
        out = sdm._addr_forward(addr)
        touched.update(np.nonzero(out)[0].tolist())

    n = len(address_vecs)
    naive_expected = min(1.0, (n * sdm.N_w) / sdm.W)  # crude, ignores collisions
    observed = len(touched) / sdm.W

    return OccupancyReport(
        n_patterns=n,
        n_hard_locations=sdm.W,
        n_w=sdm.N_w,
        naive_expected_fraction=naive_expected,
        observed_fraction=observed,
        observed_count=len(touched),
    )


# ---------------------------------------------------------------------------
# Check 3: MAX-Hebbian rule actually takes max, not sum
# ---------------------------------------------------------------------------

@dataclass
class MaxRuleCheck:
    matches_max_semantics: bool
    would_have_matched_additive_semantics: bool


def verify_max_hebbian_rule(sdm: RankOrderSDM, addr_decoded_a: np.ndarray, addr_decoded_b: np.ndarray,
                             data_a: np.ndarray, data_b: np.ndarray) -> MaxRuleCheck:
    """
    Write two (addr_decoded, data) pairs to a fresh copy of the data store
    and confirm the resulting weights equal an explicit max() of the two
    outer products, not their sum. Operates on a throwaway copy of
    sdm.W_data so it doesn't disturb any real experiment state.
    """
    original = sdm.W_data.copy()
    sdm.W_data = np.zeros_like(sdm.W_data)

    sdm._data_write(addr_decoded_a, data_a)
    sdm._data_write(addr_decoded_b, data_b)
    actual = sdm.W_data.copy()

    expected_max = np.maximum(np.outer(data_a, addr_decoded_a), np.outer(data_b, addr_decoded_b))
    expected_sum = np.outer(data_a, addr_decoded_a) + np.outer(data_b, addr_decoded_b)

    sdm.W_data = original  # restore

    return MaxRuleCheck(
        matches_max_semantics=bool(np.allclose(actual, expected_max)),
        would_have_matched_additive_semantics=bool(np.allclose(actual, expected_sum)),
    )


# ---------------------------------------------------------------------------
# Check 4: Rank-order weights track actual magnitude order, not input order
# ---------------------------------------------------------------------------

@dataclass
class RankOrderCheck:
    query_index: int
    weights_match_magnitude_order: bool


def verify_rank_order_preserved(sdm: RankOrderSDM, address_vecs: List[np.ndarray]) -> List[RankOrderCheck]:
    """
    For each query, recompute raw activations, derive the expected
    alpha^rank assignment purely from magnitude order, and confirm the
    actual output's nonzero weights match that derivation. This is the
    check that would have caught the earlier known bug (re-encoding
    preserving raw input weights instead of reassigning by new rank).
    """
    results = []
    for idx, addr in enumerate(address_vecs):
        out = sdm._addr_forward(addr)
        activations = sdm.W_addr @ addr
        nonzero_idx = np.nonzero(out)[0]

        # Expected order: sort the ACTIVE indices by their raw activation,
        # descending, and check assigned weight matches alpha^rank of that.
        active_sorted = nonzero_idx[np.argsort(activations[nonzero_idx])[::-1]]
        expected = {i: sdm.alpha ** rank for rank, i in enumerate(active_sorted)}
        actual = {i: out[i] for i in nonzero_idx}

        matches = all(
            abs(expected[i] - actual[i]) < 1e-9 for i in nonzero_idx
        ) and set(expected.keys()) == set(actual.keys())

        results.append(RankOrderCheck(query_index=idx, weights_match_magnitude_order=matches))
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(determinism: List[DeterminismCheck], occupancy: OccupancyReport,
              max_rule: MaxRuleCheck, rank_order: List[RankOrderCheck]) -> str:
    det_pass = sum(1 for r in determinism if r.matches_on_repeat and r.matches_manual_recompute)
    rank_pass = sum(1 for r in rank_order if r.weights_match_magnitude_order)

    lines = [
        f"Check 1 (determinism/recomputation): {det_pass}/{len(determinism)} queries pass "
        f"(deterministic AND matches manual recomputation from W_addr).",
        f"Check 2 (occupancy, descriptive): {occupancy.observed_count}/{occupancy.n_hard_locations} "
        f"hard locations touched ({100*occupancy.observed_fraction:.1f}%) across {occupancy.n_patterns} "
        f"patterns; naive expectation ~{100*occupancy.naive_expected_fraction:.1f}% ignoring collisions.",
        f"Check 3 (MAX-Hebbian rule): matches max semantics = {max_rule.matches_max_semantics}, "
        f"matches additive semantics = {max_rule.would_have_matched_additive_semantics}.",
        f"Check 4 (rank-order preserved through re-encoding): {rank_pass}/{len(rank_order)} queries pass.",
    ]

    if det_pass < len(determinism):
        lines.append("WARNING: address decoder output is non-deterministic or does not match "
                     "direct recomputation. Instance cannot be used as ground truth until resolved.")
    if not max_rule.matches_max_semantics:
        lines.append("WARNING: data store does NOT exhibit MAX-Hebbian semantics as specified. "
                     "This is a serious mismatch between specification and actual behavior.")
    if rank_pass < len(rank_order):
        lines.append("WARNING: rank-order significance weights do not track actual magnitude order "
                     "for at least one query. Re-encoding may be preserving stale weights.")

    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sdm = RankOrderSDM(D=64, N_i=6, N_a=20, W=256, N_w=8, N_d=6, alpha=0.99, seed=0)

    address_vecs = list(make_significance_vectors(20, 64, 6, 0.99, rng))

    determinism_results = verify_address_decoder_determinism(sdm, address_vecs)
    occupancy_result = report_hard_location_occupancy(sdm, address_vecs)

    addr_decoded_a = sdm._addr_forward(address_vecs[0])
    addr_decoded_b = sdm._addr_forward(address_vecs[1])
    data_a = make_significance_vectors(1, 64, 6, 0.99, rng)[0]
    data_b = make_significance_vectors(1, 64, 6, 0.99, rng)[0]
    max_rule_result = verify_max_hebbian_rule(sdm, addr_decoded_a, addr_decoded_b, data_a, data_b)

    rank_order_results = verify_rank_order_preserved(sdm, address_vecs)

    print(summarize(determinism_results, occupancy_result, max_rule_result, rank_order_results))
