"""
instances/sdm/run_discovery.py

The real Section 6 pipeline for the SDM address-decoder instance, mirroring
instances/hopfield/run_discovery.py's structure:

  1. Verification (verification.py) — confirm the instance's actual behavior
     matches its specification before using it as ground truth.
  2. Select a query, get its target (unablated top-N_w decoder selection).
  3. Ground truth via marginal single-edge causal necessity
     (discovery_adapter.identify_necessary_edges).
  4. Real discovery via the generic cumulative ACDC-style procedure
     (discovery/acdc_runner.py).
  5. Score discovered circuit against ground truth (protocol/metrics.py).

Scope: address-decoder layer only. The data-store (MAX-Hebbian) layer is a
separate follow-up, not attempted here.
"""

import os
import sys

SDM_REPO_SRC = os.environ.get(
    "SDM_REPO_SRC",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "sequence-machine-revisited", "src"),
)
sys.path.insert(0, SDM_REPO_SRC)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "protocol"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "discovery"))

import numpy as np

from sdm_library import RankOrderSDM, make_significance_vectors
from verification import (
    verify_address_decoder_determinism,
    report_hard_location_occupancy,
    verify_max_hebbian_rule,
    verify_rank_order_preserved,
    summarize as summarize_verification,
)
from discovery_adapter import SDMAddressDecoderAdapter, identify_necessary_edges, build_circuit_from_necessary_edges
from metrics import compare_circuits
from acdc_runner import discover_circuit, DiscoveryConfig


def main():
    rng = np.random.default_rng(0)
    sdm = RankOrderSDM(D=64, N_i=6, N_a=20, W=256, N_w=8, N_d=6, alpha=0.99, seed=0)
    address_vecs = list(make_significance_vectors(20, 64, 6, 0.99, rng))

    print("=== Step 1: Verification ===")
    determinism_results = verify_address_decoder_determinism(sdm, address_vecs)
    occupancy_result = report_hard_location_occupancy(sdm, address_vecs)
    addr_decoded_a = sdm._addr_forward(address_vecs[0])
    addr_decoded_b = sdm._addr_forward(address_vecs[1])
    data_a = make_significance_vectors(1, 64, 6, 0.99, rng)[0]
    data_b = make_significance_vectors(1, 64, 6, 0.99, rng)[0]
    max_rule_result = verify_max_hebbian_rule(sdm, addr_decoded_a, addr_decoded_b, data_a, data_b)
    rank_order_results = verify_rank_order_preserved(sdm, address_vecs)
    print(summarize_verification(determinism_results, occupancy_result, max_rule_result, rank_order_results))

    if any(not (r.matches_on_repeat and r.matches_manual_recompute) for r in determinism_results):
        print("Aborting: determinism check failed.")
        return
    print()

    print("=== Step 2: Select query and target ===")
    query = address_vecs[0]
    target_out = sdm._addr_forward(query)
    target_set = frozenset(np.nonzero(target_out)[0].tolist())
    print(f"Target decoder set (top-N_w={sdm.N_w}): {sorted(target_set)}")
    print()

    print("=== Step 3: Ground truth via single-edge causal necessity ===")
    adapter = SDMAddressDecoderAdapter(sdm, query_target_pairs=[(query, target_set)])
    full = adapter.full_graph()
    print(f"Candidate edges (full structural graph): {len(full.edges)}")
    necessary_edges = identify_necessary_edges(adapter, (query, target_set), full.edges, tolerance=0.0)
    true_circuit = build_circuit_from_necessary_edges(necessary_edges)
    print(f"Necessary edges found: {len(necessary_edges)} out of {len(full.edges)} candidates")
    print()

    print("=== Step 4: Real discovery (generic ACDC-style CUMULATIVE procedure) ===")
    config = DiscoveryConfig(threshold=0.0)  # exact match required, matches necessity criterion's tolerance
    discovered_circuit, discovery_trace = discover_circuit(
        adapter, queries=[(query, target_set)], config=config
    )
    print(f"Threshold used: {config.threshold}")
    print(f"Baseline metric (no ablation): {discovery_trace.baseline_metric:.3f}")
    print(f"Edges pruned (deemed non-load-bearing): {len(discovery_trace.pruned_edges_in_order)}")
    print(f"Edges kept (deemed load-bearing): {len(discovery_trace.kept_edges_in_order)}")
    print()

    print("=== Step 5: Score discovered circuit against ground truth ===")
    result = compare_circuits(true_circuit, discovered_circuit)
    print(result.summary())


if __name__ == "__main__":
    main()
