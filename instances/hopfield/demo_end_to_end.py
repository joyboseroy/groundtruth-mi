"""
instances/hopfield/demo_end_to_end.py

Demonstrates the full pipeline for the Hopfield instance, end to end:

  1. Build network, store patterns (build.py)
  2. Verify: fixed points + no spurious attractors (verification.py)
  3. Select ONLY a query that passed verification
  4. Extract ground-truth Circuit from its recall trace (ground_truth.py)
  5. Score a "discovered" circuit against it (protocol/metrics.py)

Step 5 currently uses a synthetic stand-in for a discovery method's output,
NOT a real ACDC run. This is intentional and must remain clearly labeled as
such until discovery/acdc_runner.py exists and is actually wired to a real
discovery method. Do not read the numbers below as a result; they exist only
to prove the pipeline (ground truth -> comparison -> interpretation) works
mechanically before any real method is plugged in.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "protocol"))

import numpy as np

from build import HopfieldNetwork, corrupt
from verification import verify_fixed_points, verify_no_spurious_attractors, summarize
from ground_truth import build_circuit_from_trace
from metrics import Circuit, compare_circuits


def make_stand_in_discovered_circuit(true_circuit: Circuit, drop_fraction: float, seed: int) -> Circuit:
    """
    Synthetic stand-in for a discovery method's output: drops a fraction of
    true nodes/edges (simulating a conservative method) and adds a couple of
    extraneous ones (simulating phantom attribution). This exists ONLY to
    exercise the pipeline before a real discovery method is integrated.
    """
    rng = np.random.default_rng(seed)
    nodes = set(true_circuit.nodes)
    edges = set(true_circuit.edges)

    keep_nodes = {n for n in nodes if rng.random() > drop_fraction}
    keep_edges = {e for e in edges if e[0] in keep_nodes and e[1] in keep_nodes}

    # Simulate a couple of phantom edges/nodes not in ground truth.
    phantom_nodes = {f"unit_{rng.integers(100, 200)}" for _ in range(2)}
    phantom_edges = {(pn, list(keep_nodes)[0]) for pn in phantom_nodes} if keep_nodes else set()

    return Circuit.from_sets(
        nodes=keep_nodes | phantom_nodes,
        edges=keep_edges | phantom_edges,
        paths=true_circuit.paths,  # left unchanged for this stand-in
    )


def main():
    rng = np.random.default_rng(0)
    n = 20
    net = HopfieldNetwork(n_units=n)
    patterns = [rng.choice([-1, 1], size=n) for _ in range(3)]
    net.store(patterns)

    fp_results = verify_fixed_points(net)
    sa_results = verify_no_spurious_attractors(net, n_trials_per_pattern=10, seed=1)
    print("--- Verification ---")
    print(summarize(fp_results, sa_results))
    print()

    if not all(r.is_fixed_point for r in fp_results):
        print("Aborting: not all stored patterns are fixed points. Fix before proceeding.")
        return

    # Select a query, but only proceed if it actually converges to a
    # legitimate attractor. This is the "exclude, don't assume" step.
    query = corrupt(patterns[0], n_flips=3, rng=rng)
    recall = net.recall_trace(query)
    is_legitimate = np.array_equal(recall.final_state, patterns[0]) or np.array_equal(
        recall.final_state, -patterns[0]
    )
    if not is_legitimate:
        print("Aborting: selected query did not converge to a legitimate attractor. "
              "Pick a different query or lower corruption level.")
        return

    true_circuit = build_circuit_from_trace(recall, top_k_contributions=3)
    print(f"--- Ground truth (query converged to a legitimate attractor) ---")
    print(f"Nodes: {len(true_circuit.nodes)}, Edges: {len(true_circuit.edges)}, "
          f"Path length: {len(next(iter(true_circuit.paths), ()))}")
    print()

    discovered_circuit = make_stand_in_discovered_circuit(true_circuit, drop_fraction=0.2, seed=2)

    print("--- Comparison (STAND-IN discovery output, not a real method) ---")
    result = compare_circuits(true_circuit, discovered_circuit)
    print(result.summary())


if __name__ == "__main__":
    main()
