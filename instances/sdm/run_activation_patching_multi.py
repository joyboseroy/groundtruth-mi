"""
instances/sdm/run_activation_patching_multi.py

Runs activation patching (discovery/activation_patching.py) against the SDM
address-decoder instance, across all 20 training queries, directly
comparable to run_discovery_multi.py's ACDC-style results.

Expectation, stated before running rather than after: since this method's
scoring is structurally close to the marginal-necessity ground-truth
criterion itself, high agreement here is expected, not a surprising
success. The comparison that matters is whether activation patching's
EDGE RECALL is higher than ACDC's (mean 0.574), which would support the
paper's existing explanation that ACDC's edge-recall cap is an artifact
of greedy, cumulative pruning order specifically, not a general property
of circuit discovery on this architecture.
"""

import os
import sys
import csv

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
from activation_patching import discover_circuit_activation_patching, ActivationPatchingConfig


def main():
    rng = np.random.default_rng(0)
    sdm = RankOrderSDM(D=64, N_i=6, N_a=20, W=256, N_w=8, N_d=6, alpha=0.99, seed=0)
    address_vecs = list(make_significance_vectors(20, 64, 6, 0.99, rng))

    print("=== Verification (run once, applies to all queries) ===")
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

    config = ActivationPatchingConfig(threshold=0.0)
    rows = []

    print("=== Per-query results: Activation Patching vs. Ground Truth ===")
    print(f"{'id':>3} {'n_cand':>7} {'n_necessary':>11} {'n_disc_ap':>10} "
          f"{'node_p':>7} {'node_r':>7} {'edge_p':>7} {'edge_r':>7}")

    for idx, query in enumerate(address_vecs):
        target_out = sdm._addr_forward(query)
        target_set = frozenset(np.nonzero(target_out)[0].tolist())

        adapter = SDMAddressDecoderAdapter(sdm, query_target_pairs=[(query, target_set)])
        full = adapter.full_graph()

        necessary_edges = identify_necessary_edges(adapter, (query, target_set), full.edges, tolerance=0.0)
        true_circuit = build_circuit_from_necessary_edges(necessary_edges)

        discovered_circuit, trace = discover_circuit_activation_patching(
            adapter, queries=[(query, target_set)], config=config
        )

        result = compare_circuits(true_circuit, discovered_circuit)
        node_p, node_r = result.node_metrics.precision, result.node_metrics.recall
        edge_p, edge_r = result.edge_metrics.precision, result.edge_metrics.recall

        rows.append({
            "id": idx, "n_cand": len(full.edges), "n_necessary": len(necessary_edges),
            "n_disc_ap": len(discovered_circuit.edges),
            "node_p": node_p, "node_r": node_r, "edge_p": edge_p, "edge_r": edge_r,
        })

        def fmt(x):
            return "n/a" if x is None else f"{x:.3f}"

        print(f"{idx:>3} {len(full.edges):>7} {len(necessary_edges):>11} "
              f"{len(discovered_circuit.edges):>10} {fmt(node_p):>7} {fmt(node_r):>7} "
              f"{fmt(edge_p):>7} {fmt(edge_r):>7}")

    print()
    print("=== Summary: Activation Patching on SDM (compare to ACDC: node_p=0.949, node_r=0.921, edge_p=0.912, edge_r=0.574) ===")

    def mean_of(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    for key, label in [("node_p", "Node precision"), ("node_r", "Node recall"),
                        ("edge_p", "Edge precision"), ("edge_r", "Edge recall")]:
        m = mean_of(key)
        n_defined = sum(1 for r in rows if r[key] is not None)
        print(f"{label}: mean={('n/a' if m is None else f'{m:.3f}')} (defined for {n_defined}/{len(rows)})")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "sdm")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "activation_patching_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "n_candidates", "n_necessary", "n_discovered_ap",
                                                 "node_precision", "node_recall", "edge_precision", "edge_recall"])
        writer.writeheader()
        for r in rows:
            def fmt2(x):
                return "" if x is None else f"{x:.3f}"
            writer.writerow({
                "query_id": r["id"], "n_candidates": r["n_cand"], "n_necessary": r["n_necessary"],
                "n_discovered_ap": r["n_disc_ap"],
                "node_precision": fmt2(r["node_p"]), "node_recall": fmt2(r["node_r"]),
                "edge_precision": fmt2(r["edge_p"]), "edge_recall": fmt2(r["edge_r"]),
            })
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
