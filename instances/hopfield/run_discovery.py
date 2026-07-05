"""
instances/hopfield/run_discovery.py

The real Section 6 pipeline for the Hopfield instance, replacing the
synthetic stand-in in demo_end_to_end.py with an actual discovery run via
discovery/acdc_runner.py.

Pipeline:
  1. Build network, store patterns (build.py)
  2. Verify: fixed points + no spurious attractors (verification.py)
  3. Select a query, confirm it individually converges to a legitimate
     attractor (not just checked in aggregate during step 2)
  4. Extract ground-truth Circuit from its recall trace (ground_truth.py)
  5. Run the generic ACDC-style discovery procedure against the SAME query,
     using discovery_adapter.py to expose the network to acdc_runner.py
  6. Score the discovered circuit against ground truth (protocol/metrics.py)
  7. Report the discovery trace (which edges were pruned/kept) alongside
     the score, for transparency

This script's output is the first real (non-synthetic) result for the
Hopfield instance and is what Section 6 of the paper should describe,
not demo_end_to_end.py's stand-in numbers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "protocol"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "discovery"))

import numpy as np

from build import HopfieldNetwork, corrupt
from verification import verify_fixed_points, verify_no_spurious_attractors, summarize
from discovery_adapter import HopfieldAblatableAdapter, identify_necessary_edges, build_circuit_from_necessary_edges
from metrics import compare_circuits
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

    print("=== Step 1-2: Verification ===")
    fp_results = verify_fixed_points(net)
    sa_results = verify_no_spurious_attractors(net, n_trials_per_pattern=10, seed=1)
    print(summarize(fp_results, sa_results))
    if not all(r.is_fixed_point for r in fp_results):
        print("Aborting: stored patterns are not all fixed points.")
        return
    print()

    print("=== Step 3: Select and confirm a single query ===")
    query = corrupt(patterns[0], n_flips=3, rng=rng)
    recall = net.recall_trace(query.copy())
    if not is_legitimate_attractor(recall.final_state, patterns):
        print("Aborting: selected query did not converge to a legitimate attractor.")
        return
    print(f"Query converged in {recall.sweeps_run} sweep(s), {len(recall.trace)} flip(s).")
    print()

    print("=== Step 4: Build adapter with FULL candidate graph (all contributors) ===")
    target = recall.final_state
    adapter = HopfieldAblatableAdapter(net, query_target_pairs=[(query, target)])
    full = adapter.full_graph()
    print(f"Candidate nodes: {len(full.nodes)}, candidate edges: {len(full.edges)} "
          f"(all contributors, not top-K)")
    print()

    print("=== Step 5a: Ground truth via single-edge causal necessity (Option B) ===")
    necessary_edges = identify_necessary_edges(adapter, (query, target), full.edges, tolerance=0.0)
    true_circuit = build_circuit_from_necessary_edges(necessary_edges)
    print(f"Necessary edges found: {len(necessary_edges)} out of {len(full.edges)} candidates")
    if necessary_edges:
        print(f"Necessary edges: {sorted(necessary_edges)}")
    else:
        print("No single edge is individually necessary at this pattern load: "
              "this is a real finding, not a pipeline failure. It means this "
              "Hopfield instance's recall is robust to any one contributor's "
              "removal, consistent with distributed, majority-vote computation "
              "rather than a small set of individually critical connections.")
    print()

    print("=== Step 5b: Run real discovery (generic ACDC-style CUMULATIVE procedure) ===")
    config = DiscoveryConfig(threshold=0.02)
    discovered_circuit, discovery_trace = discover_circuit(
        adapter, queries=[(query, target)], config=config
    )
    print(f"Threshold used: {config.threshold}")
    print(f"Baseline metric (no ablation): {discovery_trace.baseline_metric:.3f}")
    print(f"Edges pruned (deemed non-load-bearing): {len(discovery_trace.pruned_edges_in_order)}")
    print(f"Edges kept (deemed load-bearing): {len(discovery_trace.kept_edges_in_order)}")
    print()

    print("=== Step 6: Score discovered circuit against ground truth ===")
    result = compare_circuits(true_circuit, discovered_circuit)
    print(result.summary())
    print()

    print("=== Step 7: Full discovery trace, for transparency ===")
    print(f"Pruned edges: {sorted(discovery_trace.pruned_edges_in_order)}")
    print(f"Kept edges:   {sorted(discovery_trace.kept_edges_in_order)}")


if __name__ == "__main__":
    main()
