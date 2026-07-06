"""
instances/hopfield/run_activation_patching_multi.py

Runs the activation-patching discovery method (discovery/activation_patching.py)
against the Hopfield instance, across the same query set used in
run_discovery_multi.py for the ACDC-style method, so the two methods are
directly comparable on identical queries.

The real question this script answers: on an instance where marginal-necessity
ground truth is empty on every query (see run_discovery_multi.py's result),
does an independent/parallel scoring method also report empty circuits, or
does it, like ACDC's cumulative method, still find spurious structure?
"""

import sys
import os
import csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "protocol"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "discovery"))

import numpy as np

from build import HopfieldNetwork, corrupt
from verification import verify_fixed_points, summarize, verify_no_spurious_attractors
from discovery_adapter import HopfieldAblatableAdapter, identify_necessary_edges
from activation_patching import discover_circuit_activation_patching, ActivationPatchingConfig


def is_legitimate_attractor(final_state, patterns):
    return any(
        np.array_equal(final_state, p) or np.array_equal(final_state, -p)
        for p in patterns
    )


def main():
    rng = np.random.default_rng(0)
    n = 20
    net = HopfieldNetwork(n_units=n)
    patterns = [rng.choice([-1, 1], size=n) for _ in range(3)]
    net.store(patterns)

    print("=== Verification ===")
    fp_results = verify_fixed_points(net)
    sa_results = verify_no_spurious_attractors(net, n_trials_per_pattern=10, seed=1)
    print(summarize(fp_results, sa_results))
    if not all(r.is_fixed_point for r in fp_results):
        print("Aborting: stored patterns are not all fixed points.")
        return
    print()

    corruption_levels = [2, 3, 4]
    trials_per_level = 3
    config = ActivationPatchingConfig(threshold=0.0)

    rows = []
    query_id = 0
    for pattern_idx, pattern in enumerate(patterns):
        for level in corruption_levels:
            for _ in range(trials_per_level):
                query_id += 1
                query = corrupt(pattern, n_flips=level, rng=rng)
                recall = net.recall_trace(query.copy())

                if not is_legitimate_attractor(recall.final_state, patterns):
                    rows.append({"query_id": query_id, "excluded": True})
                    continue

                target = recall.final_state
                adapter = HopfieldAblatableAdapter(net, query_target_pairs=[(query, target)])
                full = adapter.full_graph()

                necessary = identify_necessary_edges(adapter, (query, target), full.edges, tolerance=0.0)
                discovered_circuit, trace = discover_circuit_activation_patching(
                    adapter, queries=[(query, target)], config=config
                )

                rows.append({
                    "query_id": query_id, "excluded": False,
                    "n_candidates": len(full.edges),
                    "n_necessary": len(necessary),
                    "n_discovered_ap": len(discovered_circuit.edges),
                })

    print("=== Per-query results: Activation Patching vs. Ground Truth ===")
    print(f"{'id':>3} {'cands':>6} {'necessary':>9} {'discovered (AP)':>16}")
    for r in rows:
        if r.get("excluded"):
            print(f"{r['query_id']:>3} EXCLUDED (spurious attractor)")
            continue
        print(f"{r['query_id']:>3} {r['n_candidates']:>6} {r['n_necessary']:>9} {r['n_discovered_ap']:>16}")

    included = [r for r in rows if not r.get("excluded")]
    n_empty_gt = sum(1 for r in included if r["n_necessary"] == 0)
    n_ap_also_empty = sum(1 for r in included if r["n_necessary"] == 0 and r["n_discovered_ap"] == 0)
    n_ap_nonempty_given_empty_gt = n_empty_gt - n_ap_also_empty

    print()
    print("=== Summary: Activation Patching on Hopfield ===")
    print(f"Included queries: {len(included)}")
    print(f"Queries with empty ground truth: {n_empty_gt} / {len(included)}")
    print(f"Of those, activation patching ALSO reported empty: {n_ap_also_empty} / {n_empty_gt}")
    print(f"Of those, activation patching reported nonempty (like ACDC did): "
          f"{n_ap_nonempty_given_empty_gt} / {n_empty_gt}")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "hopfield")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "activation_patching_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "excluded", "n_candidates", "n_necessary", "n_discovered_ap"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "query_id": r["query_id"],
                "excluded": r.get("excluded", False),
                "n_candidates": r.get("n_candidates", ""),
                "n_necessary": r.get("n_necessary", ""),
                "n_discovered_ap": r.get("n_discovered_ap", ""),
            })
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
