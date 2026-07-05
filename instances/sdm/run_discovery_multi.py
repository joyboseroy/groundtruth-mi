"""
instances/sdm/run_discovery_multi.py

Same pipeline as run_discovery.py, looped over all T=20 training address
patterns, to check whether the single-query precision/recall profile
(~0.92 node precision, ~0.85 node recall, ~0.89 edge precision, ~0.57 edge
recall) holds generally or was specific to one favorable query.
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
from acdc_runner import discover_circuit, DiscoveryConfig


def main():
    rng = np.random.default_rng(0)
    sdm = RankOrderSDM(D=64, N_i=6, N_a=20, W=256, N_w=8, N_d=6, alpha=0.99, seed=0)
    address_vecs = list(make_significance_vectors(20, 64, 6, 0.99, rng))

    print("=== Verification (run once, applies to all queries: W_addr is fixed) ===")
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

    config = DiscoveryConfig(threshold=0.0)
    rows = []

    print("=== Per-query results ===")
    print(f"{'id':>3} {'n_cand':>7} {'n_necessary':>11} {'n_discovered':>12} "
          f"{'node_p':>7} {'node_r':>7} {'edge_p':>7} {'edge_r':>7}")

    for idx, query in enumerate(address_vecs):
        target_out = sdm._addr_forward(query)
        target_set = frozenset(np.nonzero(target_out)[0].tolist())

        adapter = SDMAddressDecoderAdapter(sdm, query_target_pairs=[(query, target_set)])
        full = adapter.full_graph()

        necessary_edges = identify_necessary_edges(adapter, (query, target_set), full.edges, tolerance=0.0)
        true_circuit = build_circuit_from_necessary_edges(necessary_edges)

        discovered_circuit, _trace = discover_circuit(adapter, queries=[(query, target_set)], config=config)

        result = compare_circuits(true_circuit, discovered_circuit)

        node_p = result.node_metrics.precision
        node_r = result.node_metrics.recall
        edge_p = result.edge_metrics.precision
        edge_r = result.edge_metrics.recall

        rows.append({
            "id": idx, "n_cand": len(full.edges), "n_necessary": len(necessary_edges),
            "n_discovered": len(discovered_circuit.edges),
            "node_p": node_p, "node_r": node_r, "edge_p": edge_p, "edge_r": edge_r,
        })

        def fmt(x):
            return "n/a" if x is None else f"{x:.3f}"

        print(f"{idx:>3} {len(full.edges):>7} {len(necessary_edges):>11} "
              f"{len(discovered_circuit.edges):>12} {fmt(node_p):>7} {fmt(node_r):>7} "
              f"{fmt(edge_p):>7} {fmt(edge_r):>7}")

    print()
    print("=== Summary across queries ===")
    n_total = len(rows)
    n_empty_gt = sum(1 for r in rows if r["n_necessary"] == 0)
    print(f"Queries with nonempty ground truth: {n_total - n_empty_gt} / {n_total}")
    if n_empty_gt:
        print(f"Queries with EMPTY ground truth (0 necessary edges): {n_empty_gt} / {n_total} "
              f"(unlike Hopfield, expected to be rare or zero here given the discrete cutoff)")

    def mean_of(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    for key, label in [("node_p", "Node precision"), ("node_r", "Node recall"),
                        ("edge_p", "Edge precision"), ("edge_r", "Edge recall")]:
        m = mean_of(key)
        n_defined = sum(1 for r in rows if r[key] is not None)
        print(f"{label}: mean={('n/a' if m is None else f'{m:.3f}')} "
              f"(defined for {n_defined}/{n_total} queries)")

    print()
    print(f"Necessary-edge count across queries: min={min(r['n_necessary'] for r in rows)}, "
          f"max={max(r['n_necessary'] for r in rows)}, "
          f"mean={sum(r['n_necessary'] for r in rows)/n_total:.1f}")
    print(f"Discovered-edge count across queries: min={min(r['n_discovered'] for r in rows)}, "
          f"max={max(r['n_discovered'] for r in rows)}, "
          f"mean={sum(r['n_discovered'] for r in rows)/n_total:.1f}")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "sdm")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "multi_query_results.csv")
    fieldnames = ["query_id", "n_candidates", "n_necessary", "n_discovered",
                  "node_precision", "node_recall", "edge_precision", "edge_recall"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            def fmt(x):
                return "" if x is None else f"{x:.3f}"
            writer.writerow({
                "query_id": r["id"],
                "n_candidates": r["n_cand"],
                "n_necessary": r["n_necessary"],
                "n_discovered": r["n_discovered"],
                "node_precision": fmt(r["node_p"]),
                "node_recall": fmt(r["node_r"]),
                "edge_precision": fmt(r["edge_p"]),
                "edge_recall": fmt(r["edge_r"]),
            })
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
