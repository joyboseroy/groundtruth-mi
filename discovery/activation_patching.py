"""
discovery/activation_patching.py

A second discovery method, structurally distinct from discovery/acdc_runner.py's
cumulative, sequential greedy pruning, so the benchmark protocol can be shown
to compare methods, not just describe one.

ACDC-style discovery (acdc_runner.py) prunes edges one at a time against an
EVER-SHRINKING baseline: whether edge N is kept depends on every decision
made about edges 1..N-1. This is what makes it "cumulative."

Activation patching, as implemented here, scores every candidate edge
INDEPENDENTLY: each edge is ablated alone, in isolation, and its effect is
measured against the SAME original, fully-unablated baseline every time. No
edge's score depends on any other edge's ablation status. This is the
"parallel" counterpart to ACDC's "sequential" search.

HONESTY NOTE, stated explicitly rather than left implicit: this scoring
procedure is structurally very close to the marginal-necessity ground-truth
criterion itself (identify_necessary_edges in each instance's
discovery_adapter.py). Both ablate one edge alone against the unablated
baseline. The difference is that ground truth uses an exact, binary
criterion (any deviation from the target counts as necessary), while this
method uses a continuous score thresholded at a chosen cutoff (default 0.0,
matching the ground-truth tolerance for a fair comparison).

This means: on any instance where marginal necessity is well-defined and
non-degenerate (e.g. SDM), this method is EXPECTED to recover ground truth
closely, close to by construction, not as a surprising empirical success.
The genuinely informative comparison is what this method does on an
instance where ground truth is EMPTY (Hopfield): does an independent,
parallel scoring method also correctly report an empty or near-empty
circuit, unlike ACDC's cumulative approach, which was shown to report a
nonempty circuit on every tested query? That result is not guaranteed in
advance by the method's structure, and is the real test this module exists
to run.
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List

from metrics import Circuit, Edge


@dataclass(frozen=True)
class ActivationPatchingConfig:
    threshold: float = 0.0
    # An edge is included in the discovered circuit if its independent
    # ablation effect exceeds this threshold. 0.0 matches the "any
    # deviation counts" tolerance used for the marginal-necessity ground
    # truth criterion, so the two are directly, fairly comparable.


@dataclass(frozen=True)
class ActivationPatchingTrace:
    baseline_metric: float
    scores: Dict[Edge, float]  # independent ablation effect per candidate edge


def discover_circuit_activation_patching(
    system,
    queries: List,
    config: ActivationPatchingConfig = ActivationPatchingConfig(),
):
    """
    system: any AblatableSystem-compatible object (see discovery/acdc_runner.py
            for the interface: full_graph(), run_metric_with_ablation(...)).
    queries: list of query objects passed through to run_metric_with_ablation,
             same convention as acdc_runner.discover_circuit.
    """
    full = system.full_graph()
    candidate_edges = sorted(full.edges)

    def avg_metric(ablated: FrozenSet[Edge]) -> float:
        return sum(system.run_metric_with_ablation(q, ablated) for q in queries) / len(queries)

    baseline = avg_metric(frozenset())

    scores: Dict[Edge, float] = {}
    for edge in candidate_edges:
        metric_ablated = avg_metric(frozenset({edge}))
        scores[edge] = abs(baseline - metric_ablated)

    discovered_edges = {e for e, s in scores.items() if s > config.threshold}
    discovered_nodes = {n for e in discovered_edges for n in e}

    discovered_circuit = Circuit.from_sets(
        nodes=discovered_nodes,
        edges=discovered_edges,
        paths=[],
    )
    trace = ActivationPatchingTrace(baseline_metric=baseline, scores=scores)
    return discovered_circuit, trace
