"""
instances/hopfield/ground_truth.py

Builds a `Circuit` (protocol/metrics.py) from a Hopfield recall trace.

Modeling decision, stated explicitly rather than left implicit: a Hopfield
network is fully connected, so "the true circuit" cannot mean every nonzero
weight, that would just be the entire network for every query. Instead, the
true circuit for a specific recall is defined as:

  - nodes: units that flipped during convergence (the units whose state
    actually changed on the path from query to attractor)
  - edges: for each flip, the top-K contributing units by |contribution|
    (W[i,j] * state[j] at the moment of the flip), i.e. the units that
    dominated the local field driving that flip
  - paths: the single temporal sequence of flipped units, in the order
    they flipped (a flip-history path, not a spatial path)

K (top_k_contributions) is a stated parameter of this ground-truth
definition, not a hidden default. Changing it changes what counts as
"the true circuit," and this must be reported alongside any benchmark
result that uses it.

Only recall traces that passed verification.py's spurious-attractor check
should be passed to build_circuit_from_trace. This module does not check
that itself; the caller is responsible for filtering (see demo_end_to_end.py
for the intended usage pattern).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "protocol"))

from typing import List
from build import RecallResult, FlipEvent
from metrics import Circuit


def _unit_name(i: int) -> str:
    return f"unit_{i}"


def build_circuit_from_trace(recall: RecallResult, top_k_contributions: int = 3) -> Circuit:
    """
    Convert a RecallResult's flip trace into a Circuit for use as ground
    truth in protocol/metrics.py.
    """
    nodes = set()
    edges = set()
    path: List[str] = []

    for event in recall.trace:
        flipped_node = _unit_name(event.unit)
        nodes.add(flipped_node)
        path.append(flipped_node)

        # Rank contributors by absolute contribution magnitude, take top-K.
        ranked = sorted(
            event.contributions.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        for other_unit, _contribution in ranked[:top_k_contributions]:
            contributor_node = _unit_name(other_unit)
            nodes.add(contributor_node)
            edges.add((contributor_node, flipped_node))

    return Circuit.from_sets(
        nodes=nodes,
        edges=edges,
        paths=[tuple(path)] if path else [],
    )


if __name__ == "__main__":
    import numpy as np
    from build import HopfieldNetwork, corrupt

    rng = np.random.default_rng(0)
    n = 20
    net = HopfieldNetwork(n_units=n)
    patterns = [rng.choice([-1, 1], size=n) for _ in range(3)]
    net.store(patterns)

    query = corrupt(patterns[0], n_flips=4, rng=rng)
    recall = net.recall_trace(query)

    circuit = build_circuit_from_trace(recall, top_k_contributions=3)
    print(f"Ground-truth nodes: {sorted(circuit.nodes)}")
    print(f"Ground-truth edges: {sorted(circuit.edges)}")
    print(f"Ground-truth path: {circuit.paths}")
