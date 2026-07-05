"""
instances/hopfield/discovery_adapter.py

Implements the AblatableSystem interface (discovery/acdc_runner.py) for the
Hopfield instance. This is what turns discover_circuit() from a generic
algorithm into something that can actually run against a real network.

Key design decisions, stated explicitly:

  - The candidate graph (full_graph) is NOT the complete N^2 connectivity of
    the Hopfield network. It is restricted to edges that actually appear in
    the unablated recall trace(s) of the query set being tested (i.e. the
    same top-K-contribution edges used in ground_truth.py). This mirrors how
    the original ACDC operates on a bounded, known computational graph (a
    transformer's fixed set of components), not an unbounded search space.
    It also keeps runtime tractable. This restriction is a real scoping
    decision and must be reported as such, since it means the discovery
    procedure is only ever asked to prune among edges the ground-truth
    extractor already considered "candidate" edges, not the full network.

  - The task metric for a given query is agreement (fraction of matching
    signs) between the recall's final state and the TARGET attractor, i.e.
    the attractor the unablated network legitimately converges to for that
    query. The target must be pre-verified as legitimate (see verification.py)
    before being used here; this module does not re-verify it.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "protocol"))

from typing import FrozenSet, List, Tuple
import numpy as np

from metrics import Circuit, Edge
from build import HopfieldNetwork


QueryTargetPair = Tuple[np.ndarray, np.ndarray]


def _unit_name(i: int) -> str:
    return f"unit_{i}"


def _parse_edge(edge: Edge) -> Tuple[int, int]:
    """edge is (contributor_node, flipped_node); return (contributor_idx, flipped_idx)."""
    src, dst = edge
    return int(src.split("_")[1]), int(dst.split("_")[1])


class HopfieldAblatableAdapter:
    """
    Wraps a trained, verified HopfieldNetwork so discover_circuit() can run
    against it. Constructed with a list of (query, target) pairs, where each
    target is the legitimate attractor that query's unablated recall
    converges to (already checked against verification.py's spurious-
    attractor test before being passed here).
    """

    def __init__(self, net: HopfieldNetwork, query_target_pairs: List[QueryTargetPair],
                 max_sweeps: int = 20):
        """
        Note: the earlier top_k_contributions restriction has been removed.
        Under Option B (single-edge causal necessity as ground truth), the
        candidate graph must include EVERY contributor to a flipped unit's
        local field, not just the top-K by magnitude. Restricting the
        candidate set to top-K was the source of the earlier degenerate
        result (ablating 3 of 19 contributors could never flip a field
        whose sign was determined by the full 19-way sum).
        """
        self.net = net
        self.query_target_pairs = query_target_pairs
        self.max_sweeps = max_sweeps
        self._candidate_nodes, self._candidate_edges = self._build_candidate_graph()

    def _build_candidate_graph(self):
        nodes = set()
        edges = set()
        for query, _target in self.query_target_pairs:
            recall = self.net.recall_trace(query.copy())
            for event in recall.trace:
                flipped = _unit_name(event.unit)
                nodes.add(flipped)
                for other_unit in event.contributions:  # ALL contributors, not top-K
                    contributor = _unit_name(other_unit)
                    nodes.add(contributor)
                    edges.add((contributor, flipped))
        return nodes, edges

    def full_graph(self) -> Circuit:
        return Circuit.from_sets(nodes=self._candidate_nodes, edges=self._candidate_edges)

    def _recall_with_ablation(self, query: np.ndarray, ablated_edges: FrozenSet[Edge]) -> np.ndarray:
        """
        Re-run recall from `query`, but zero the contribution of any
        ablated (contributor -> flipped) pair when computing the flipped
        unit's local field. This is a separate loop from
        HopfieldNetwork.recall_trace because that method has no ablation
        hook; duplicating the sweep logic here keeps build.py free of
        discovery-specific concerns.
        """
        ablate_pairs = {_parse_edge(e) for e in ablated_edges}
        n = self.net.n_units
        state = query.copy().astype(np.float64)

        for _sweep in range(self.max_sweeps):
            flips = 0
            for j in range(n):
                row = self.net.W[j]
                h = 0.0
                for i in range(n):
                    if row[i] == 0.0:
                        continue
                    if (i, j) in ablate_pairs:
                        continue
                    h += row[i] * state[i]
                new_val = 1.0 if h > 0 else (-1.0 if h < 0 else state[j])
                if new_val != state[j]:
                    state[j] = new_val
                    flips += 1
            if flips == 0:
                break
        return state

    def run_metric_with_ablation(self, query_target_pair: QueryTargetPair,
                                  ablated_edges: FrozenSet[Edge]) -> float:
        query, target = query_target_pair
        final_state = self._recall_with_ablation(query, ablated_edges)
        matches = float(np.sum(final_state == target))
        return matches / len(target)


def build_circuit_from_necessary_edges(necessary_edges: FrozenSet[Edge]) -> Circuit:
    """
    Wrap a set of causally-necessary edges (from identify_necessary_edges)
    into a Circuit object, matching the interface protocol/metrics.py
    expects. No path is set here; paths would need a separate, explicit
    definition under a marginal-necessity criterion and are left empty
    rather than guessed at.
    """
    nodes = {n for e in necessary_edges for n in e}
    return Circuit.from_sets(nodes=nodes, edges=necessary_edges, paths=[])


def identify_necessary_edges(
    adapter: "HopfieldAblatableAdapter",
    query_target_pair: QueryTargetPair,
    candidate_edges,
    tolerance: float = 0.0,
) -> FrozenSet[Edge]:
    """
    Option B ground-truth definition: an edge is "necessary" if ablating it
    ALONE (nothing else ablated) changes the final converged state relative
    to the target, i.e. run_metric_with_ablation drops below 1.0 - tolerance
    when only that single edge is removed.

    This tests MARGINAL necessity (one edge at a time, all else intact),
    which is a different question from what discover_circuit() in
    acdc_runner.py tests (CUMULATIVE, order-dependent pruning). The two can
    legitimately disagree when the true mechanism has redundancy: no single
    edge may be individually necessary even though some subset of edges
    jointly is. If discovered and necessary-edge sets diverge, that
    divergence itself is a reportable finding, not evidence of a bug in
    either procedure.
    """
    necessary = set()
    for edge in candidate_edges:
        metric = adapter.run_metric_with_ablation(query_target_pair, frozenset({edge}))
        if metric < 1.0 - tolerance:
            necessary.add(edge)
    return frozenset(necessary)


if __name__ == "__main__":
    # Standalone sanity check: confirm ablating an unrelated edge doesn't
    # hurt the metric, and confirm the metric is 1.0 with no ablation at all
    # for a query that already converges correctly.
    from build import corrupt

    rng = np.random.default_rng(0)
    n = 20
    net = HopfieldNetwork(n_units=n)
    patterns = [rng.choice([-1, 1], size=n) for _ in range(3)]
    net.store(patterns)

    query = corrupt(patterns[0], n_flips=3, rng=rng)
    recall = net.recall_trace(query.copy())
    target = recall.final_state

    adapter = HopfieldAblatableAdapter(net, query_target_pairs=[(query, target)])
    full = adapter.full_graph()
    print(f"Candidate nodes: {len(full.nodes)}, candidate edges: {len(full.edges)}")

    baseline_metric = adapter.run_metric_with_ablation((query, target), frozenset())
    print(f"Baseline metric (no ablation): {baseline_metric:.3f}")
    assert baseline_metric == 1.0, "Baseline with no ablation should exactly match its own target."

    some_edge = next(iter(full.edges))
    ablated_metric = adapter.run_metric_with_ablation((query, target), frozenset({some_edge}))
    print(f"Metric after ablating one candidate edge {some_edge}: {ablated_metric:.3f}")
    print("Self-test passed.")
