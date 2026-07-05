# GroundTruthMI

A benchmark protocol for evaluating mechanistic interpretability methods against architectures whose computational mechanism is specified independently of learning, and whose actual behavior has been verified to match that specification before being used as ground truth.

This repository accompanies the paper *"Ground Truth for Mechanistic Interpretability: A Benchmark Protocol Using Architectures with Explicit Computational Semantics."* It is independent of, and does not depend on, the companion repository [`sequence-machine-revisited`](https://github.com/joyboseroy/sequence-machine-revisited), which this repo reuses only as a source library for one of its two benchmark instances (see below).

## Why this exists

Mechanistic interpretability methods (circuit discovery, activation patching, causal abstraction) are currently validated by qualitative plausibility, performance on hand-designed toy tasks, or agreement between methods that may share the same blind spots. None of these is an independently specified ground truth. This repository provides two architectures where the correct mechanism is stated before training or construction begins, verifies that each instance's actual behavior matches that specification, and applies a generic circuit-discovery algorithm against both to show what an independently verified ground truth reveals that toy-task validation would not.

## Repository structure

```
groundtruth-mi/
├── protocol/
│   └── metrics.py                  Generic node/edge/path precision, recall, F1.
│                                    Architecture-agnostic and discovery-method-
│                                    agnostic; knows nothing about Hopfield, SDM,
│                                    or ACDC specifically.
│
├── discovery/
│   └── acdc_runner.py               A generic reimplementation of the CORE
│                                     ALGORITHM behind Automatic Circuit Discovery
│                                     (Conmy et al., 2023): greedy, threshold-based
│                                     edge ablation. This is NOT the official
│                                     Automatic-Circuit-Discovery library, which is
│                                     built for transformer computational graphs.
│                                     See the module docstring for the exact
│                                     distinction; do not cite this as "ACDC"
│                                     without that qualification.
│
├── instances/
│   ├── hopfield/                    Instance 1: classical Hopfield network
│   │   ├── build.py                 Network construction, Hebbian storage,
│   │   │                            asynchronous recall (no training loop).
│   │   ├── verification.py          Two checks: fixed-point confirmation and
│   │   │                            spurious-attractor screening. Any query that
│   │   │                            fails the spurious-attractor check is
│   │   │                            excluded from the benchmark's query set.
│   │   ├── discovery_adapter.py     AblatableSystem adapter exposing the network
│   │   │                            to discovery/acdc_runner.py, plus
│   │   │                            identify_necessary_edges() implementing the
│   │   │                            marginal (single-edge) necessity ground-truth
│   │   │                            criterion.
│   │   ├── run_discovery.py         Single-query real pipeline: verify -> ground
│   │   │                            truth -> discovery -> score.
│   │   └── run_discovery_multi.py   Same pipeline across 27 queries (25 included
│   │                                 after exclusions), replicating the finding.
│   │
│   └── sdm/                         Instance 2: rank-order SDM address-decoder
│       │                            layer (data-store layer not in scope here)
│       ├── verification.py          Four checks: determinism/recomputation match,
│       │                            hard-location occupancy (reported
│       │                            descriptively, not graded against 100%),
│       │                            MAX-Hebbian rule confirmation, rank-order
│       │                            weight preservation through re-encoding.
│       ├── discovery_adapter.py     AblatableSystem adapter over the FULL
│       │                            structural edge set (every nonzero address-
│       │                            decoder weight, not truncated in advance),
│       │                            plus identify_necessary_edges().
│       ├── run_discovery.py         Single-query real pipeline.
│       └── run_discovery_multi.py   Same pipeline across all 20 training queries.
│
├── external/
│   └── sequence-machine-revisited/  Git submodule (pinned commit) providing
│                                     sdm_library (RankOrderSDM, base.py encoding
│                                     utilities). Not copied or vendored; see
│                                     "Shared dependency" below.
│
├── paper/
│   └── ground_truth_mi_draft.md     Current paper draft.
│
└── results/
    Raw terminal output / CSVs from actual runs, kept as supplementary evidence
    for the numbers reported in the paper.
```

## Shared dependency: `sdm_library`

The SDM instance imports `RankOrderSDM` and `make_significance_vectors` from `sdm_library`, which lives in the separate [`sequence-machine-revisited`](https://github.com/joyboseroy/sequence-machine-revisited) repository (the thesis-faithful reimplementation and CALM/SpikingMamba comparison paper). That repository is included here as a **pinned git submodule**, not copied, so this repository's results stay reproducible against the exact library version that produced them, independent of any later changes made there for other purposes.

To set up:

```bash
git submodule update --init --recursive
```

If you are not using the submodule, set the `SDM_REPO_SRC` environment variable to point at your local checkout of `sequence-machine-revisited/src` instead:

```bash
export SDM_REPO_SRC=/path/to/sequence-machine-revisited/src
```

## Running the benchmarks

Each instance is self-contained once its dependencies are on the path.

**Hopfield** (no external dependency beyond `numpy`):

```bash
cd instances/hopfield
python3 build.py                 # sanity check: network builds, converges
python3 verification.py          # fixed-point + spurious-attractor checks
python3 run_discovery.py         # single-query real result
python3 run_discovery_multi.py   # 27-query replication (25 included)
```

**SDM** (requires `sdm_library` on the path, see above):

```bash
cd instances/sdm
export SDM_REPO_SRC=/path/to/sequence-machine-revisited/src   # or use the submodule
python3 verification.py          # four-item checklist
python3 run_discovery.py         # single-query real result
python3 run_discovery_multi.py   # 20-query replication
```

Both instance folders assume `protocol/` and `discovery/` are two directories up; this is handled automatically via `sys.path` insertion at the top of each script, no manual `PYTHONPATH` setup needed beyond `SDM_REPO_SRC` for the SDM instance.

## Headline results

Full detail and interpretation are in `paper/ground_truth_mi_draft.md`, Section 6. Summary:

| | Hopfield (25 queries) | SDM (20 queries) |
|---|---|---|
| Ground truth ever empty (0 marginally-necessary edges) | 25 / 25 | 0 / 20 |
| Node precision | undefined | 0.949 (0.846–1.000) |
| Node recall | undefined | 0.921 (0.846–1.000) |
| Edge precision | undefined | 0.912 (0.778–1.000) |
| Edge recall | undefined | 0.574 (0.483–0.643) |

The same generic, cumulative circuit-discovery algorithm produces a completely empty-vs-nonempty divergence on the Hopfield instance (a fully redundant, majority-vote mechanism) and a stable, informative precision/recall profile on the SDM instance (a sparse, discrete-cutoff mechanism). This contrast, not either result in isolation, is the paper's central finding: precision and recall in circuit discovery are properties of the relationship between a method and an architecture, not of the method alone, and that relationship is only visible once ground truth has been independently specified and verified.

## Relationship to other repositories

- [`sequence-machine-revisited`](https://github.com/joyboseroy/sequence-machine-revisited): thesis-faithful SDM reimplementation and the companion CALM/SpikingMamba comparison paper. This repository depends on it (via submodule) for one instance's underlying architecture, but is otherwise independent; neither paper's claims depend on the other's acceptance or results.

## Status and scope

This is a first, two-instance version of a proposed standing benchmark suite (working name within the paper: GroundTruthMI). Section 7 of the paper sketches natural extensions not yet implemented here: the SDM data-store (MAX-Hebbian) layer, Hierarchical Temporal Memory, small hand-specified spiking neural networks, and additional discovery methods beyond the generic ACDC-style procedure used here (activation patching, causal abstraction, edge attribution). Contributions extending the suite along these lines are the intended next steps, not currently implemented.

## License

MIT, matching `sequence-machine-revisited`.
