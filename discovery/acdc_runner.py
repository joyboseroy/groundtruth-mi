"""
discovery/acdc_runner.py

A generic, architecture-agnostic reimplementation of the CORE ALGORITHM behind
Automatic Circuit Discovery (ACDC; Conmy et al. 2023): iteratively ablate
candidate edges and keep only those whose removal changes a task metric by
more than a chosen threshold. Edges that can be removed without hurting the
metric are pruned; what remains is the "discovered circuit."

IMPORTANT, do not gloss over this in the paper: this is NOT the official
Automatic-Circuit-Discovery library (github.com/ArthurConmy/Automatic-Circuit-
Discovery), which is built specifically for transformer computational graphs
(attention heads, MLPs, residual stream edges) and expects hooks into that
specific graph structure. Neither Hopfield networks nor SDM expose that
structure. This module reimplements ACDC's central idea, greedy edge ablation
against a metric-preservation threshold, generically, so it can be applied to
any system that exposes the AblatableSystem interface below.

In any writeup, this should be described precisely as "an ACDC-style
greedy edge-ablation procedure, implemented generically for non-transformer
architectures," not as "we ran ACDC." A reviewer who knows the original
library will immediately ask which one this is; answer honestly in the text,
not just in this docstring.

Reference (for the algorithm this reimplements): Conmy, A., Mavor-Parker, A.,
Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards Automated
Circuit Discovery for Mechanistic Interpretability. arXiv:2304.14997.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, List, Protocol, Sequence

from metrics import Circuit, Edge, Node


# ---------------------------------------------------------------------------
# Interface every instance (Hopfield, SDM, future additions) must implement
# to be usable with this generic discovery procedure.
# ---------------------------------------------------------------------------

class AblatableSystem(Protocol):
    """
    Any system this discovery procedure runs against must be able to answer
    two questions for a given query: (1) the full candidate graph before any
    pruning, and (2) what a task metric evaluates to when a specified set of
    edges is ablated (their contribution zeroed out) during computation.

    Concrete adapters (e.g. instances/hopfield/discovery_adapter.py,
    instances/sdm/discovery_adapter.py — not yet built) implement this
    Protocol by wrapping their own recall/forward-pass logic.
    """

    def full_graph(self) -> Circuit:
        """All candidate nodes and edges before any pruning is applied."""
        ...

    def run_metric_with_ablation(self, query, ablated_edges: FrozenSet[Edge]) -> float:
        """
        Run the system on `query` with the given edges ablated (their
        contribution to computation removed), and return a scalar task
        metric. Higher must mean better/more faithful to the correct output
        (e.g. 1.0 minus normalized Hamming distance to the correct attractor
        for Hopfield, or retrieval accuracy for SDM). The metric with an
        empty ablation set is the system's unablated baseline performance.
        """
        ...


# ---------------------------------------------------------------------------
# Discovery configuration and procedure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscoveryConfig:
    threshold: float = 0.02
    # Edges are tried for pruning in the order returned by full_graph().edges,
    # converted to a sorted list for determinism. This is a simplification
    # relative to the original ACDC, which processes edges in a specific
    # reverse-topological order tied to the transformer's layer structure.
    # State this explicitly as a limitation if edge order turns out to
    # matter for a given instance's results.


@dataclass(frozen=True)
class DiscoveryTrace:
    """
    Record of the pruning process, kept for transparency/debugging, not just
    the final discovered circuit. Useful when writing up Section 6 to show
    *how* the discovered circuit was arrived at, not just what it ended up
    being.
    """
    baseline_metric: float
    pruned_edges_in_order: List[Edge]
    kept_edges_in_order: List[Edge]
    metric_after_each_prune_attempt: List[float]


def discover_circuit(
    system: AblatableSystem,
    queries: Sequence,
    config: DiscoveryConfig = DiscoveryConfig(),
) -> "tuple[Circuit, DiscoveryTrace]":
    """
    Generic ACDC-style discovery: greedily ablate each candidate edge in
    turn; if ablating it (on top of everything already pruned) changes the
    metric, averaged over `queries`, by no more than `config.threshold`
    relative to the current (partially pruned) baseline, keep it pruned.
    Otherwise restore it. What remains after one full pass is the
    discovered circuit.

    Multiple queries are supported and averaged, since a single query's
    recall may not exercise every edge in the full candidate graph; a
    discovered circuit should reflect behavior across the query set the
    benchmark actually cares about, not one instance.
    """
    full = system.full_graph()
    candidate_edges = sorted(full.edges)  # deterministic order, see DiscoveryConfig note

    pruned: set = set()

    def avg_metric(ablated: FrozenSet[Edge]) -> float:
        return sum(system.run_metric_with_ablation(q, ablated) for q in queries) / len(queries)

    baseline = avg_metric(frozenset(pruned))

    trace_pruned: List[Edge] = []
    trace_kept: List[Edge] = []
    trace_metrics: List[float] = []

    current_baseline = baseline
    for edge in candidate_edges:
        trial_pruned = pruned | {edge}
        metric_trial = avg_metric(frozenset(trial_pruned))
        trace_metrics.append(metric_trial)

        if abs(current_baseline - metric_trial) <= config.threshold:
            # Removing this edge didn't hurt beyond threshold: prune it.
            pruned = trial_pruned
            current_baseline = metric_trial
            trace_pruned.append(edge)
        else:
            # Removing this edge hurt the metric too much: keep it.
            trace_kept.append(edge)

    discovered_edges = set(candidate_edges) - pruned
    discovered_nodes = {n for e in discovered_edges for n in e}

    discovered_circuit = Circuit.from_sets(
        nodes=discovered_nodes,
        edges=discovered_edges,
        paths=[],  # this generic procedure does not reconstruct paths;
                   # an instance-specific adapter may add path reconstruction
                   # on top of the discovered edge set if needed for Section 6
    )

    discovery_trace = DiscoveryTrace(
        baseline_metric=baseline,
        pruned_edges_in_order=trace_pruned,
        kept_edges_in_order=trace_kept,
        metric_after_each_prune_attempt=trace_metrics,
    )

    return discovered_circuit, discovery_trace


# ---------------------------------------------------------------------------
# Self-test with a hand-built toy AblatableSystem (no instance dependency).
#
# This proves the generic algorithm behaves correctly in isolation: given a
# graph where some edges are load-bearing and some are decorative, it should
# prune the decorative ones and keep the load-bearing ones. This is a
# sanity check on the discovery procedure itself, not a real experiment.
# ---------------------------------------------------------------------------

class _ToyAblatableSystem:
    """
    Toy system: output = sum of contributions from 'load-bearing' edges only.
    Decorative edges contribute nothing to the metric regardless of ablation.
    A correct discovery procedure should prune all decorative edges and keep
    all load-bearing ones, for any reasonable threshold.
    """

    def __init__(self):
        self.load_bearing = {("a", "out"), ("b", "out")}
        self.decorative = {("c", "out"), ("d", "b")}
        self._graph = Circuit.from_sets(
            nodes={"a", "b", "c", "d", "out"},
            edges=self.load_bearing | self.decorative,
        )

    def full_graph(self) -> Circuit:
        return self._graph

    def run_metric_with_ablation(self, query, ablated_edges: FrozenSet[Edge]) -> float:
        # Metric = 1.0 if all load-bearing edges survive ablation, degraded
        # proportionally for each load-bearing edge that's been ablated.
        # Decorative edges never affect this metric.
        surviving_load_bearing = self.load_bearing - set(ablated_edges)
        return len(surviving_load_bearing) / len(self.load_bearing)


if __name__ == "__main__":
    toy = _ToyAblatableSystem()
    discovered, trace = discover_circuit(toy, queries=[None], config=DiscoveryConfig(threshold=0.01))

    print(f"True load-bearing edges: {sorted(toy.load_bearing)}")
    print(f"Discovered edges:        {sorted(discovered.edges)}")
    print(f"Pruned (correctly, expected = decorative): {sorted(trace.pruned_edges_in_order)}")
    print(f"Baseline metric: {trace.baseline_metric}")

    assert discovered.edges == toy.load_bearing, (
        "Sanity check failed: discovery procedure did not recover exactly "
        "the load-bearing edges on this toy system."
    )
    print("Self-test passed: discovered circuit matches known load-bearing edges.")
