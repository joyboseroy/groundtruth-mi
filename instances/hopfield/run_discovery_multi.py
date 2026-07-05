"""
instances/hopfield/run_discovery_multi.py

Same pipeline as run_discovery.py, but looped over several queries (varying
source pattern and corruption level) to check whether the single-query
finding replicates: does marginal (single-edge) necessity stay empty while
cumulative discovery still reports edges, across different queries, or was
that specific to one query?

Only queries that individually converge to a legitimate attractor (a stored
pattern or its global sign inverse) are included, per the same exclusion
rule as verification.py.
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
from acdc_runner import discover_circuit, DiscoveryConfig


def is_legitimate_attractor(final_state: np.ndarray, patterns) -> bool:
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
    config = DiscoveryConfig(threshold=0.02)

    rows = []
    query_id = 0
    for pattern_idx, pattern in enumerate(patterns):
        for level in corruption_levels:
            for _ in range(trials_per_level):
                query_id += 1
                query = corrupt(pattern, n_flips=level, rng=rng)
                recall = net.recall_trace(query.copy())

                if not is_legitimate_attractor(recall.final_state, patterns):
                    rows.append({
                        "query_id": query_id, "pattern": pattern_idx, "level": level,
                        "excluded": True,
                    })
                    continue

                target = recall.final_state
                adapter = HopfieldAblatableAdapter(net, query_target_pairs=[(query, target)])
                full = adapter.full_graph()

                necessary = identify_necessary_edges(adapter, (query, target), full.edges, tolerance=0.0)
                discovered_circuit, _trace = discover_circuit(adapter, queries=[(query, target)], config=config)

                rows.append({
                    "query_id": query_id, "pattern": pattern_idx, "level": level,
                    "excluded": False,
                    "n_flips_in_recall": len(recall.trace),
                    "n_candidates": len(full.edges),
                    "n_necessary": len(necessary),
                    "n_discovered": len(discovered_circuit.edges),
                })

    print("=== Per-query results ===")
    print(f"{'id':>3} {'pattern':>7} {'level':>5} {'flips':>5} {'cands':>5} "
          f"{'necessary':>9} {'discovered':>10}")
    for r in rows:
        if r.get("excluded"):
            print(f"{r['query_id']:>3} {r['pattern']:>7} {r['level']:>5} "
                  f"{'EXCLUDED (spurious attractor)':>30}")
            continue
        print(f"{r['query_id']:>3} {r['pattern']:>7} {r['level']:>5} "
              f"{r['n_flips_in_recall']:>5} {r['n_candidates']:>5} "
              f"{r['n_necessary']:>9} {r['n_discovered']:>10}")

    included = [r for r in rows if not r.get("excluded")]
    n_empty_ground_truth = sum(1 for r in included if r["n_necessary"] == 0)
    n_nonempty_discovery_given_empty_gt = sum(
        1 for r in included if r["n_necessary"] == 0 and r["n_discovered"] > 0
    )

    print()
    print("=== Summary across queries ===")
    print(f"Total queries run: {len(rows)}, excluded (spurious): "
          f"{sum(1 for r in rows if r.get('excluded'))}, included: {len(included)}")
    print(f"Queries with EMPTY marginal ground truth (0 necessary edges): "
          f"{n_empty_ground_truth} / {len(included)}")
    print(f"Of those, queries where cumulative discovery STILL found edges anyway: "
          f"{n_nonempty_discovery_given_empty_gt} / {n_empty_ground_truth if n_empty_ground_truth else 0}")
    if n_empty_ground_truth == len(included) and n_nonempty_discovery_given_empty_gt > 0:
        print()
        print("The single-query finding replicates: across every included query, no single "
              "edge is individually necessary, yet cumulative sequential ablation still "
              "reports a nonempty circuit. This supports treating the divergence as a "
              "general property of this instance's redundant computation, not an artifact "
              "of one specific query.")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "hopfield")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "multi_query_results.csv")
    fieldnames = ["query_id", "pattern", "corruption_level", "excluded_spurious",
                  "n_flips_in_recall", "n_candidates", "n_necessary", "n_discovered"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            if r.get("excluded"):
                writer.writerow({
                    "query_id": r["query_id"], "pattern": r["pattern"],
                    "corruption_level": r["level"], "excluded_spurious": True,
                    "n_flips_in_recall": "", "n_candidates": "", "n_necessary": "", "n_discovered": "",
                })
            else:
                writer.writerow({
                    "query_id": r["query_id"], "pattern": r["pattern"],
                    "corruption_level": r["level"], "excluded_spurious": False,
                    "n_flips_in_recall": r["n_flips_in_recall"],
                    "n_candidates": r["n_candidates"],
                    "n_necessary": r["n_necessary"],
                    "n_discovered": r["n_discovered"],
                })
    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
