"""
protocol/metrics.py

Generic ground-truth comparison protocol for the GroundTruthMI benchmark.

This module is deliberately architecture-agnostic and discovery-method-agnostic.
It knows nothing about SDM, Hopfield networks, or ACDC. It only knows how to
compare two `Circuit` objects: one representing a known, verified ground-truth
mechanism, and one representing the output of some discovery method applied
to that same system treated as a black box.

Usage pattern:

    true_circuit = Circuit(nodes={...}, edges={...}, paths={...})
    discovered_circuit = Circuit(nodes={...}, edges={...}, paths={...})
    result = compare_circuits(true_circuit, discovered_circuit)
    print(result.summary())

Any instance (SDM, Hopfield, future additions) is responsible for producing
a `Circuit` object for its ground truth, and for wrapping a discovery method's
output into a `Circuit` object as well. This module does the scoring only.
"""

from dataclasses import dataclass, field
from typing import FrozenSet, Set, Tuple, Optional


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

Node = str  # node identifiers are opaque strings, e.g. "hard_location_42"
Edge = Tuple[Node, Node]  # directed edge (src, dst)
Path = Tuple[Node, ...]  # ordered sequence of nodes representing a full pathway


@dataclass(frozen=True)
class Circuit:
    """
    A circuit is a set of nodes, a set of directed edges, and (optionally) a
    set of full causal paths. This is the common representation both ground
    truth and discovery output must be expressed in before comparison.

    Paths are optional because not every discovery method or every instance
    naturally produces full end-to-end pathways; node- and edge-level scoring
    remain meaningful without them.
    """
    nodes: FrozenSet[Node] = field(default_factory=frozenset)
    edges: FrozenSet[Edge] = field(default_factory=frozenset)
    paths: FrozenSet[Path] = field(default_factory=frozenset)

    @staticmethod
    def from_sets(nodes=None, edges=None, paths=None) -> "Circuit":
        return Circuit(
            nodes=frozenset(nodes or []),
            edges=frozenset(edges or []),
            paths=frozenset(paths or []),
        )


# ---------------------------------------------------------------------------
# Generic precision / recall / F1
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PRF1:
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    true_positives: int
    false_positives: int
    false_negatives: int

    def summary(self, label: str) -> str:
        def fmt(x):
            return "n/a" if x is None else f"{x:.3f}"
        return (
            f"{label}: precision={fmt(self.precision)} "
            f"recall={fmt(self.recall)} f1={fmt(self.f1)} "
            f"(tp={self.true_positives}, fp={self.false_positives}, "
            f"fn={self.false_negatives})"
        )


def _prf1_from_sets(true_set: Set, discovered_set: Set) -> PRF1:
    """
    Generic set-based precision/recall/F1. Used identically for nodes and
    edges; the caller decides what the elements of the sets are.
    """
    tp = len(true_set & discovered_set)
    fp = len(discovered_set - true_set)
    fn = len(true_set - discovered_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None

    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return PRF1(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


# ---------------------------------------------------------------------------
# Path-level alignment (Jaccard over path sets)
# ---------------------------------------------------------------------------

def _jaccard(true_set: Set, discovered_set: Set) -> Optional[float]:
    union = true_set | discovered_set
    if not union:
        return None
    return len(true_set & discovered_set) / len(union)


# ---------------------------------------------------------------------------
# Aggregate comparison result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CircuitComparisonResult:
    node_metrics: PRF1
    edge_metrics: PRF1
    path_jaccard: Optional[float]

    def summary(self) -> str:
        lines = [
            self.node_metrics.summary("Node"),
            self.edge_metrics.summary("Edge"),
            f"Path Jaccard: {'n/a' if self.path_jaccard is None else f'{self.path_jaccard:.3f}'}",
            "",
            interpret(self),
        ]
        return "\n".join(lines)


def compare_circuits(true_circuit: Circuit, discovered_circuit: Circuit) -> CircuitComparisonResult:
    """
    Score a discovered circuit against a verified ground-truth circuit.

    Ground truth (`true_circuit`) must already have passed the instance's own
    verification procedure (see instances/*/verification.py) before being
    passed here. This function does not check that; it assumes it.
    """
    node_metrics = _prf1_from_sets(set(true_circuit.nodes), set(discovered_circuit.nodes))
    edge_metrics = _prf1_from_sets(set(true_circuit.edges), set(discovered_circuit.edges))
    path_jaccard = _jaccard(set(true_circuit.paths), set(discovered_circuit.paths))

    return CircuitComparisonResult(
        node_metrics=node_metrics,
        edge_metrics=edge_metrics,
        path_jaccard=path_jaccard,
    )


# ---------------------------------------------------------------------------
# Score interpretation
#
# Raw numbers don't tell a reader what went wrong. This function translates
# precision/recall patterns into the qualitative failure-mode categories the
# paper needs to discuss in Section 4 ("interpreting scores") and Section 6
# ("validation results"). It is intentionally conservative: it flags patterns,
# it does not assert a causal explanation for them. Any causal claim (e.g.
# "this drop is due to the MAX-Hebbian non-linearity") must be argued
# separately from the actual architecture and data, not read off these
# thresholds alone.
# ---------------------------------------------------------------------------

_HIGH = 0.85
_LOW = 0.40


def _band(value: Optional[float]) -> str:
    if value is None:
        return "undefined"
    if value >= _HIGH:
        return "high"
    if value <= _LOW:
        return "low"
    return "moderate"


def interpret(result: CircuitComparisonResult) -> str:
    """
    Produce a plain-language interpretation of a comparison result, using
    the qualitative categories below. Thresholds (_HIGH, _LOW) are reported
    explicitly so they can be scrutinized and adjusted per benchmark instance
    rather than treated as fixed universal cutoffs.
    """
    true_positives = result.node_metrics.true_positives
    false_negatives = result.node_metrics.false_negatives
    ground_truth_was_empty = (true_positives + false_negatives) == 0

    if ground_truth_was_empty:
        discovered_count = result.node_metrics.false_positives
        if discovered_count == 0:
            return (
                "Ground truth was empty (no nodes/edges met the ground-truth "
                "criterion for this query) and discovery found nothing either: "
                "consistent agreement on 'no discoverable circuit exists here.'"
            )
        return (
            f"Ground truth was empty (no nodes/edges met the ground-truth criterion "
            f"for this query), but discovery reported {discovered_count} node(s) anyway. "
            f"This does not mean discovery 'failed' in the ordinary precision/recall "
            f"sense; it means the discovery method is answering a different question "
            f"than the ground-truth criterion asks (e.g. cumulative/order-dependent "
            f"necessity vs. marginal single-edge necessity). Report this divergence "
            f"explicitly rather than reading the 0.000 precision as a simple error rate."
        )

    p_band = _band(result.node_metrics.precision)
    r_band = _band(result.node_metrics.recall)

    if p_band == "high" and r_band == "high":
        pattern = "Discovery method recovers the ground-truth circuit closely."
    elif p_band == "high" and r_band == "low":
        pattern = (
            "Discovery method is conservative: what it finds is mostly correct, "
            "but it misses a substantial share of the true circuit. Consistent with "
            "a pruning threshold set too aggressively for this architecture."
        )
    elif p_band == "low" and r_band == "high":
        pattern = (
            "Discovery method over-attributes: it recovers most of the true circuit "
            "but also reports many nodes/edges that are not part of the true mechanism "
            "(phantom pathways)."
        )
    elif p_band == "low" and r_band == "low":
        pattern = (
            "Discovery method's output diverges substantially from the true mechanism "
            "on both precision and recall. This pattern is worth investigating for a "
            "structural mismatch between the method's assumptions and the architecture's "
            "actual computation, rather than assuming the method is simply miscalibrated."
        )
    else:
        pattern = "Mixed or moderate agreement; inspect precision/recall values directly."

    return (
        f"Node precision band: {p_band}, node recall band: {r_band} "
        f"(thresholds: high>={_HIGH}, low<={_LOW}).\n{pattern}"
    )


# ---------------------------------------------------------------------------
# Minimal self-test (not a substitute for real instance tests)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    true_c = Circuit.from_sets(
        nodes={"h1", "h2", "h3"},
        edges={("in1", "h1"), ("in2", "h2"), ("h1", "out"), ("h2", "out")},
        paths={("in1", "h1", "out"), ("in2", "h2", "out")},
    )
    discovered_c = Circuit.from_sets(
        nodes={"h1", "h2", "h4"},
        edges={("in1", "h1"), ("in2", "h2"), ("h1", "out")},
        paths={("in1", "h1", "out")},
    )
    result = compare_circuits(true_c, discovered_c)
    print(result.summary())
