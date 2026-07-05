"""
instances/sdm/discovery_adapter.py

AblatableSystem adapter for the RankOrderSDM address-decoder layer (input
dimensions -> hard locations), scoped deliberately to this one layer first;
the data-store layer (MAX-Hebbian write + read) is a separate follow-up
experiment, not attempted here.

Candidate graph: ALL structurally-existing (input_dim, decoder) edges, i.e.
every nonzero entry of W_addr. With D=64, W=256, N_a=20 this is exactly
W*N_a = 5120 edges. This is deliberately NOT truncated to a smaller
candidate set up front (e.g. "only decoders near the cutoff boundary"),
because the earlier Hopfield instance's degenerate first result came
precisely from scoping candidates too narrowly before checking whether that
scoping was justified. Every structurally-existing edge is a candidate here;
if this turns out to be computationally expensive at larger W, narrowing the
candidate set can be revisited with actual timing data as justification, not
assumed in advance.

Ground truth definition (mirrors Hopfield's Option B): an edge (input_dim,
decoder) is marginally necessary if ablating it ALONE (zeroing only that one
weight, everything else intact) changes the top-N_w decoder selection
relative to the unablated (target) selection for that query.

Metric: fraction overlap between the ablated top-N_w decoder set and the
target top-N_w decoder set (1.0 = identical set, N_w/N_w; lower values mean
some decoders entered or left the selection).
"""

import os
import sys

SDM_REPO_SRC = os.environ.get(
    "SDM_REPO_SRC",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "sequence-machine-revisited", "src"),
)
sys.path.insert(0, SDM_REPO_SRC)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "protocol"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "discovery"))

from typing import FrozenSet, List, Tuple, Set
import numpy as np

from sdm_library import RankOrderSDM
from metrics import Circuit, Edge


QueryTargetPair = Tuple[np.ndarray, FrozenSet[int]]  # (address_vec, target decoder indices)


def _input_name(i: int) -> str:
    return f"input_{i}"


def _decoder_name(j: int) -> str:
    return f"decoder_{j}"


def _parse_edge(edge: Edge) -> Tuple[int, int]:
    src, dst = edge
    return int(src.split("_")[1]), int(dst.split("_")[1])


class SDMAddressDecoderAdapter:
    """
    Wraps a RankOrderSDM instance so discover_circuit() (discovery/
    acdc_runner.py) can run against its address-decoder layer.
    """

    def __init__(self, sdm: RankOrderSDM, query_target_pairs: List[QueryTargetPair]):
        self.sdm = sdm
        self.query_target_pairs = query_target_pairs
        self._candidate_nodes, self._candidate_edges = self._build_candidate_graph()

    def _build_candidate_graph(self):
        """
        Full structural edge set: every nonzero (input_dim, decoder) weight
        in W_addr, independent of any specific query. This is the same for
        every query since W_addr is fixed after initialization; it is not
        re-derived per query the way Hopfield's per-query trace was.
        """
        nodes: Set[str] = set()
        edges: Set[Edge] = set()
        nonzero_rows, nonzero_cols = np.nonzero(self.sdm.W_addr)
        for decoder_idx, input_idx in zip(nonzero_rows, nonzero_cols):
            src = _input_name(int(input_idx))
            dst = _decoder_name(int(decoder_idx))
            nodes.add(src)
            nodes.add(dst)
            edges.add((src, dst))
        return nodes, edges

    def full_graph(self) -> Circuit:
        return Circuit.from_sets(nodes=self._candidate_nodes, edges=self._candidate_edges)

    def _top_set_with_ablation(self, address_vec: np.ndarray, ablated_edges: FrozenSet[Edge]) -> FrozenSet[int]:
        """
        Recompute the top-N_w decoder set with specified (input, decoder)
        weight entries zeroed out, on a throwaway copy of W_addr.
        """
        W_ablated = self.sdm.W_addr.copy()
        for edge in ablated_edges:
            input_idx, decoder_idx = _parse_edge(edge)
            W_ablated[decoder_idx, input_idx] = 0.0

        activations = W_ablated @ address_vec
        # Same top-k selection logic as the real _addr_forward (base.py's
        # topk_indices via argpartition, tie-break by whatever order
        # argpartition returns; this matches production behavior exactly).
        if self.sdm.N_w >= len(activations):
            top = np.arange(len(activations))
        else:
            top = np.argpartition(activations, -self.sdm.N_w)[-self.sdm.N_w:]
        return frozenset(int(i) for i in top)

    def run_metric_with_ablation(self, query_target_pair: QueryTargetPair,
                                  ablated_edges: FrozenSet[Edge]) -> float:
        address_vec, target_set = query_target_pair
        ablated_set = self._top_set_with_ablation(address_vec, ablated_edges)
        overlap = len(ablated_set & target_set)
        return overlap / len(target_set) if target_set else 1.0


def identify_necessary_edges(
    adapter: SDMAddressDecoderAdapter,
    query_target_pair: QueryTargetPair,
    candidate_edges,
    tolerance: float = 0.0,
) -> FrozenSet[Edge]:
    """
    Marginal necessity: an edge is necessary if ablating it ALONE changes
    the metric (top-N_w set overlap with target) below 1.0 - tolerance.
    Mirrors instances/hopfield/discovery_adapter.py's identify_necessary_edges,
    same semantics, different underlying system.
    """
    necessary = set()
    for edge in candidate_edges:
        metric = adapter.run_metric_with_ablation(query_target_pair, frozenset({edge}))
        if metric < 1.0 - tolerance:
            necessary.add(edge)
    return frozenset(necessary)


def build_circuit_from_necessary_edges(necessary_edges: FrozenSet[Edge]) -> Circuit:
    nodes = {n for e in necessary_edges for n in e}
    return Circuit.from_sets(nodes=nodes, edges=necessary_edges, paths=[])


if __name__ == "__main__":
    from sdm_library import make_significance_vectors

    rng = np.random.default_rng(0)
    sdm = RankOrderSDM(D=64, N_i=6, N_a=20, W=256, N_w=8, N_d=6, alpha=0.99, seed=0)

    address_vecs = list(make_significance_vectors(20, 64, 6, 0.99, rng))
    query = address_vecs[0]
    target_out = sdm._addr_forward(query)
    target_set = frozenset(np.nonzero(target_out)[0].tolist())

    adapter = SDMAddressDecoderAdapter(sdm, query_target_pairs=[(query, target_set)])
    full = adapter.full_graph()
    print(f"Candidate nodes: {len(full.nodes)}, candidate edges: {len(full.edges)}")
    print(f"Target decoder set (top-N_w): {sorted(target_set)}")

    baseline = adapter.run_metric_with_ablation((query, target_set), frozenset())
    print(f"Baseline metric (no ablation): {baseline:.3f}")
    assert baseline == 1.0, "Baseline with no ablation should exactly match its own target."

    some_edge = next(iter(full.edges))
    ablated_metric = adapter.run_metric_with_ablation((query, target_set), frozenset({some_edge}))
    print(f"Metric after ablating one candidate edge {some_edge}: {ablated_metric:.3f}")
    print("Self-test passed.")
