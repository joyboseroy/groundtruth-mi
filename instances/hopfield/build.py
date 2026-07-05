"""
instances/hopfield/build.py

Classical (discrete, bipolar) Hopfield network.

This is the fast, low-build-cost second instance for GroundTruthMI. There is
no training loop: the weight matrix is computed directly from the stored
patterns via Hebbian outer-product storage. The "specified mechanism" is the
energy-descent update rule itself, which is exact and closed-form, not
learned. This is precisely why the instance is fast to build and also why it
is the more toy-like of the two instances in the benchmark (no equivalent of
SDM's train-time drift is possible here, because nothing is trained).

Reference: Hopfield, J. J. (1982). Neural networks and physical systems with
emergent collective computational abilities. PNAS 79(8), 2554-2558.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np


@dataclass
class FlipEvent:
    """
    A single unit flip during asynchronous recall. Records enough detail to
    later reconstruct which other units causally contributed to the flip
    (see ground_truth.py), not just that a flip happened.
    """
    unit: int
    sweep: int
    old_value: int
    new_value: int
    local_field: float
    contributions: Dict[int, float]  # {other_unit_index: W[unit, other] * state[other]}


@dataclass
class RecallResult:
    final_state: np.ndarray
    trace: List[FlipEvent]
    converged: bool
    sweeps_run: int


class HopfieldNetwork:
    """
    Bipolar (+1/-1) Hopfield network with Hebbian storage and asynchronous
    (sequential sweep) update. Deterministic update order is used rather than
    random order, so recall traces are reproducible across runs, which matters
    for using a specific trace as ground truth.
    """

    def __init__(self, n_units: int):
        self.n_units = n_units
        self.W = np.zeros((n_units, n_units), dtype=np.float64)
        self.stored_patterns: List[np.ndarray] = []

    def store(self, patterns: List[np.ndarray]) -> None:
        """
        Store patterns via Hebbian outer-product summation, zero diagonal.
        Patterns must be bipolar (+1/-1) arrays of length n_units.
        """
        for p in patterns:
            assert p.shape == (self.n_units,), "pattern length must match n_units"
            assert set(np.unique(p)).issubset({-1, 1}), "patterns must be bipolar (+1/-1)"
        self.stored_patterns = list(patterns)
        W = np.zeros((self.n_units, self.n_units), dtype=np.float64)
        for p in patterns:
            W += np.outer(p, p)
        np.fill_diagonal(W, 0.0)
        self.W = W

    def energy(self, state: np.ndarray) -> float:
        return -0.5 * state @ self.W @ state

    def local_field(self, state: np.ndarray, unit: int) -> float:
        return float(self.W[unit] @ state)

    def recall_trace(
        self,
        query: np.ndarray,
        update_order: Optional[List[int]] = None,
        max_sweeps: int = 20,
    ) -> RecallResult:
        """
        Run asynchronous (sequential sweep) recall from `query`, recording
        every unit flip as a FlipEvent. A sweep with zero flips signals
        convergence. update_order defaults to sequential unit indices for
        reproducibility; pass an explicit order if a different deterministic
        order is needed.
        """
        state = query.copy().astype(np.float64)
        order = update_order if update_order is not None else list(range(self.n_units))
        trace: List[FlipEvent] = []
        converged = False

        for sweep in range(max_sweeps):
            flips_this_sweep = 0
            for i in order:
                h = self.local_field(state, i)
                new_val = 1.0 if h > 0 else (-1.0 if h < 0 else state[i])
                if new_val != state[i]:
                    contributions = {
                        j: float(self.W[i, j] * state[j])
                        for j in range(self.n_units)
                        if self.W[i, j] != 0.0
                    }
                    trace.append(
                        FlipEvent(
                            unit=i,
                            sweep=sweep,
                            old_value=int(state[i]),
                            new_value=int(new_val),
                            local_field=h,
                            contributions=contributions,
                        )
                    )
                    state[i] = new_val
                    flips_this_sweep += 1
            if flips_this_sweep == 0:
                converged = True
                break

        return RecallResult(
            final_state=state,
            trace=trace,
            converged=converged,
            sweeps_run=sweep + 1,
        )


def corrupt(pattern: np.ndarray, n_flips: int, rng: np.random.Generator) -> np.ndarray:
    """Flip n_flips randomly chosen bits of a bipolar pattern."""
    corrupted = pattern.copy()
    idx = rng.choice(len(pattern), size=n_flips, replace=False)
    corrupted[idx] *= -1
    return corrupted


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 20
    net = HopfieldNetwork(n_units=n)

    patterns = [rng.choice([-1, 1], size=n) for _ in range(3)]
    net.store(patterns)

    query = corrupt(patterns[0], n_flips=4, rng=rng)
    result = net.recall_trace(query)

    print(f"Converged: {result.converged} in {result.sweeps_run} sweeps")
    print(f"Flips recorded: {len(result.trace)}")
    print(f"Final state matches pattern[0]: {np.array_equal(result.final_state, patterns[0])}")
