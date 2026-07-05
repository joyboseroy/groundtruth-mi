# Ground Truth for Mechanistic Interpretability: A Benchmark Protocol Using Architectures with Explicit Computational Semantics

## Abstract

Mechanistic interpretability methods aim to recover the causal structure a neural network uses to compute its outputs. These methods are currently validated in three ways: qualitative plausibility of a recovered circuit, performance on toy tasks with hand-designed circuits, or agreement between different discovery methods. None of these is an independently specified ground truth, so it remains difficult to know whether a discovery method has found the correct mechanism or merely a plausible one.

We propose evaluating mechanistic analysis methods against architectures with explicit computational semantics: systems where the intended mechanism is specified as part of the architecture and can be independently verified against the trained model's actual behavior, rather than inferred from it. We define criteria for this category, a general benchmark protocol (ground truth, discovery output, comparison metric, score) instantiated as node-, edge-, and path-level precision, recall, and F1, and a verification procedure to confirm an instance's behavior matches its specified mechanism before it is used as ground truth.

We apply this protocol to two instances: a classical Hopfield network and a rank-order Sparse Distributed Memory (SDM) address-decoder layer, using a generic reimplementation of the core ACDC (Automatic Circuit Discovery) algorithm as the discovery method under test. The two instances produce sharply different, and equally informative, results. On the Hopfield instance, across 25 verified queries, no single edge is ever individually (marginally) necessary for correct recall, yet the same cumulative, order-dependent discovery procedure reports a nonempty "circuit" on every query; ground truth and discovery output diverge completely, a direct consequence of the architecture's fully redundant, majority-vote computation. On the SDM instance, across all 20 training queries, ground truth is never empty (mean 29.5 necessary edges out of 5120 structural candidates), and discovery achieves high, stable node precision (mean 0.949) and recall (mean 0.921), with consistently moderate edge recall (mean 0.574), a stable, structural gap rather than noise. The contrast between the two instances is itself the paper's central empirical finding: identical discovery machinery, given a verified ground truth, behaves in ways that directly reflect whether the underlying architecture computes through redundancy or through sparse, discrete selection. We argue this protocol generalizes beyond these two instances and propose it as a standing calibration suite for mechanistic analysis methods more broadly.

---

## 1. Introduction

Progress in empirical fields depends on benchmarks with independently verifiable ground truth. Physics has standard masses. Computer vision has labeled datasets with known ground truth. Machine translation has reference translations. Mechanistic interpretability, despite growing quickly as a subfield, has no comparable standard. A circuit-discovery method is judged by whether its output looks right to a human, whether it succeeds on a toy task the researcher designed, or whether it agrees with a different method that shares the same blind spots. None of these tells you whether the method found the actual mechanism, because in a trained transformer, the actual mechanism is unknown.

We argue that a certain class of architectures avoids this problem by construction, though with an important caveat that determines whether the argument holds at all. In these architectures, the computational mechanism, which units interact, in what order, under what rule, is specified explicitly as part of the architecture, prior to any training. Sparse distributed memory (Kanerva, 1988) and rank-order N-of-M coding (Thorpe, 1990) are two examples. Classical Hopfield networks (Hopfield, 1982) are another. In each case, the interpretation of what the model computes does not need to be extracted after training, because it is fixed by the architecture's own specification.

**The contribution of this paper is methodological rather than architectural.** Specifying a mechanism does not, on its own, guarantee that a trained or constructed instance's actual behavior matches that specification: dead units, redundant pathways, numerical drift, and unexpected interactions are all real possibilities. An instance can only serve as ground truth after its actual behavior has been independently checked against its specification, not merely asserted to match it by virtue of being "interpretable-by-construction" in name. This distinction, between an architecture whose mechanism is *specified* and an instance whose behavior has been *verified* to match that specification, is the difference between a defensible benchmark and an unearned one, and it structures every part of this paper's method.

This paper makes four contributions:

1. Criteria for architectures with explicit computational semantics, and a verification procedure that must be run against any specific instance before it is used as ground truth (Section 3).
2. A general benchmark protocol, ground truth → discovery output → comparison metric → score, instantiated here as node-, edge-, and path-level precision, recall, and F1, together with guidance for interpreting what different precision/recall patterns mean (Section 4).
3. Verified descriptions of two instances, a classical Hopfield network and a rank-order SDM address-decoder layer, each passing an explicit, reported verification procedure before being used as ground truth (Section 5).
4. An empirical application of a generic, architecture-agnostic reimplementation of ACDC's core algorithm against both instances, replicated across many queries per instance, showing that ground truth and discovery output can diverge completely on one architecture while producing a stable, informative precision/recall profile on another (Section 6).

One obstacle to evaluating mechanistic interpretability methods is the scarcity of benchmarks with independently specified, independently verified ground truth. This paper does not claim to resolve that scarcity; it proposes one way to build toward it, and reports two concrete, replicated instances as a first step toward a broader standing suite (Section 7).

---

## 2. Related Work

**Mechanistic interpretability and its current validation modes.** Circuit-discovery methods including Automatic Circuit Discovery (ACDC; Conmy et al., 2023), causal abstraction (Geiger et al., 2021; 2022; 2023), and activation patching (Wang et al., 2023, on the IOI circuit in GPT-2) are validated through a combination of qualitative plausibility, agreement on hand-constructed toy circuits (e.g. modular addition, indirect object identification), and cross-method agreement. Nanda et al. (2023) study grokking through a circuit lens; Elhage et al. (2022) study superposition. None of these establishes ground truth independent of the discovery process itself.

**Probing versus mechanistic interpretability.** Probing methods (Hewitt and Manning, 2019; Hewitt and Liang, 2019) ask what information is linearly recoverable from a representation, a different and generally easier question than how a model's computation arrives at its output. Ignat et al. (2023) survey open NLP research directions and name mechanistic interpretability explicitly as an area not solved by current large language models, distinguishing it from probing along similar lines; we adopt their framing of the open problem as motivation for this paper's method.

**Architectures with specified computational mechanisms.** Sparse distributed memory (Kanerva, 1988) specifies an address-decoding and data-storage mechanism directly, independent of any learning procedure applied on top of it. Rank-order N-of-M coding (Thorpe, 1990) specifies meaning via firing order rather than learned weight magnitude. Classical Hopfield networks (Hopfield, 1982) specify an energy-descent update rule with no training loop at all, weights are computed directly from stored patterns. Hierarchical Temporal Memory (Hawkins) and small hand-specified spiking neural networks are further candidates in this class, not explored empirically in this paper but named here as natural extensions (Section 7).

---

## 3. Architectures with Explicit Computational Semantics: Definition, Criteria, and Verification

### 3.1 Definition and criteria

An architecture qualifies as having explicit computational semantics if:

1. Its update rule (which units interact, under what function, in what order) is stated in closed form as part of the architecture's definition, independent of any training procedure.
2. For any given input, the architecture's specification determines, without appeal to the trained weights' emergent behavior, what computation *should* occur.
3. The interpretation of the model's behavior is therefore defined independently of any post-hoc analysis performed on a trained instance; a discovery method run against the instance is being checked against a preexisting, independently stated answer, not asked to produce the only available account of what happened.

This differs from calling an architecture "interpretable-by-construction," a phrase we deliberately avoid. That phrase implies a guarantee, that a specified architecture's trained behavior automatically matches its specification, which does not follow from specification alone. Dead units, redundant pathways, and emergent shortcuts are all real possibilities in an architecture that is fully specified on paper. The category defined here claims only that the mechanism *can be specified and independently checked*, not that checking is unnecessary.

### 3.2 Verification requirement

Before any specific instance of a qualifying architecture is used as ground truth, its actual behavior must be checked against its specification. We report this verification explicitly for both instances used in this paper (Section 5), rather than treating architectural specification as sufficient on its own.

### 3.3 Scope limitation

This guarantee holds at the level of the base, verified architecture. Hybridizing toward higher task performance, stacking learned components on top of a specified mechanism, reintroduces the opacity this category is meant to avoid, and would require its own, separate verification pass. Neither instance in this paper is hybridized in this way; both are used in their base, fully specified form.

---

## 4. Benchmark Protocol

### 4.1 General structure

The protocol has four stages, applicable to any qualifying, verified architecture and any discovery method: **ground truth → discovery output → comparison metric → score**. Node-, edge-, and path-level precision, recall, and F1 (below) are one instantiation of the comparison-metric stage, not the whole of it; other comparison metrics (e.g. weighted or probabilistic variants) could be substituted without changing the overall structure.

### 4.2 Node, edge, and path metrics

For a ground-truth circuit and a discovered circuit, each expressed as a set of nodes, a set of directed edges, and (optionally) a set of full causal paths:

- **Node precision/recall/F1**: standard set-based precision, recall, and F1 over the node sets.
- **Edge precision/recall/F1**: the same, over the directed edge sets.
- **Path alignment**: Jaccard similarity between discovered and ground-truth path sets, where paths are recorded.

### 4.3 Interpreting scores

Raw precision/recall numbers do not, by themselves, tell a reader what a discovery method got right or wrong. We distinguish the following patterns, used throughout Section 6:

- **High precision, high recall**: discovery closely recovers the ground-truth circuit.
- **High precision, low recall**: discovery is conservative, what it finds is largely correct, but it misses much of the true circuit, consistent with an overly aggressive pruning threshold.
- **Low precision, high recall**: discovery over-attributes, recovering most of the true circuit but also reporting substantial phantom structure.
- **Low precision, low recall**: discovery diverges from the true mechanism on both axes, worth investigating for a structural mismatch between the method's assumptions and the architecture's actual computation, rather than assuming simple miscalibration.
- **Ground truth is empty**: a degenerate but informative case (Section 6.1). If discovery nonetheless reports structure, this does not mean discovery "failed" in the ordinary sense; it means the discovery method is answering a different question than the ground-truth criterion asks. We distinguish two notions of causal necessity throughout this paper for exactly this reason:
  - **Marginal necessity**: does ablating this one edge alone, with everything else intact, change the outcome?
  - **Cumulative necessity**: given a specific greedy pruning order, does ablating this edge (on top of everything already pruned) change the outcome?
  These can diverge sharply when the true mechanism has redundancy, no single edge may be marginally necessary even though, once other redundant edges are removed, some remaining edge becomes cumulatively necessary. Section 6 reports a case where this divergence is total (Hopfield) and a case where it is partial and structurally stable (SDM).

---

## 5. Instances: Verification and Description

### 5.1 Instance 1: Classical Hopfield Network

**Architecture.** A bipolar (+1/−1), fully connected Hopfield network with Hebbian outer-product storage (Hopfield, 1982). N = 20 units; three random bipolar patterns stored. No training loop: the weight matrix is computed directly from the stored patterns. Recall proceeds by asynchronous, sequential-sweep energy descent until convergence.

**Non-triviality.** Recall under corruption is a genuine associative-memory task, not a toy circuit constructed for this paper; the network's attractor dynamics and capacity limits are well studied and not fixed in advance to produce a particular result.

**Verification (Section 3.2 requirement).** Two checks, matching the architecture's lack of a training phase (there is no learned-versus-specified gap to check for, since nothing is learned):

1. **Fixed-point check**: all three stored patterns are confirmed to be exact fixed points of the update rule (zero flips on direct recall). Result: 3/3 pass.
2. **Spurious-attractor check**: across 90 corrupted-query trials (3 patterns × 3 corruption levels × 10 trials), recall is confirmed to converge to a legitimate attractor (a stored pattern or its global sign inverse, both are valid energy minima, not spurious). Result: 81/90 trials converge to a legitimate attractor; the remaining 9 are excluded from the benchmark's query set rather than assumed to have a well-defined ground truth.

Both checks passed on the instance used; the query set used in Section 6 excludes the 9 flagged trials.

### 5.2 Instance 2: Rank-Order Sparse Distributed Memory (Address-Decoder Layer)

**Architecture.** RankOrderSDM (Kanerva, 1988; rank-order coding per Thorpe, 1990), following a three-layer thesis-faithful implementation with four separate N-of-M parameters (input code sparsity N_i, address-decoder weight sparsity N_a, address-decoder output sparsity N_w, data code sparsity N_d), rank-order significance weighting (α = 0.99), and a MAX-Hebbian data-store update rule. Parameters: D = 64, N_i = 6, N_a = 20, W = 256, N_w = 8, N_d = 6, T = 20 training patterns. This paper scopes its benchmark to the address-decoder layer only (input dimensions → hard locations); the data-store layer's MAX-Hebbian nonlinearity is a natural follow-up, not attempted here.

**Non-triviality.** The address-decoder layer performs a genuine top-k selection over a significance-weighted inner product across 256 hard locations, the same mechanism used in the associative-recall results reported for this architecture elsewhere (Section 4-equivalent capacity experiments, not repeated here), not a mechanism constructed solely to be benchmarked.

**Verification (Section 3.2 requirement).** Four checks, reflecting both a genuine train/specification gap (the data store is written incrementally) and points where the earlier draft of this checklist would have missed a real issue:

1. **Determinism and recomputation match**: for 20 test queries, the address decoder's output is confirmed deterministic across repeated calls and confirmed to match an independent recomputation of the top-N_w selection directly from the stored weight matrix. Result: 20/20 pass.
2. **Hard-location occupancy**: reported descriptively, not graded pass/fail against 100% coverage, since with T·N_w = 160 activations distributed across W = 256 locations, full coverage is not the correct expectation even in a healthy instance. Result: 116/256 locations touched (45.3%), against a naive collision-free expectation of 62.5%; the gap is consistent with ordinary hashing collisions at this scale, not evidence of a defect.
3. **MAX-Hebbian rule confirmation**: the data-store update is confirmed to take the elementwise maximum across writes, not an additive sum, by writing two known pairs to a throwaway copy of the weight matrix and checking the result against both possible semantics explicitly. Result: matches max semantics exactly; does not match additive semantics.
4. **Rank-order weight preservation through re-encoding**: for 20 queries, the significance weights assigned to re-encoded outputs are confirmed to track the actual magnitude-sorted order of raw activations, not a stale or blindly-copied input order. Result: 20/20 pass.

All four checks passed on the instance used.

---

## 6. Validation: Applying Discovery to Verified Ground Truth

The discovery method under test is a generic, architecture-agnostic reimplementation of ACDC's (Conmy et al., 2023) core algorithm: greedy, sequential ablation of candidate edges, pruning an edge if its removal (cumulative with everything already pruned) changes a task metric by no more than a fixed threshold. This is explicitly **not** the official Automatic-Circuit-Discovery library, which is built for transformer computational graphs; it is a generic reimplementation of the same underlying idea, applied here to non-transformer architectures via an instance-specific ablation adapter. We report results per instance.

### 6.1 Hopfield: Total Divergence Between Marginal and Cumulative Necessity

For each of 25 verified queries (3 source patterns × 3 corruption levels × several trials, excluding the 9 flagged spurious-attractor trials from Section 5.1), we computed:

- **Ground truth (marginal necessity)**: for every candidate contributor edge to every unit that flips during recall, does ablating that single edge alone change the final converged state relative to the correct target attractor?
- **Discovery (cumulative, ACDC-style)**: the generic discovery procedure applied to the same candidate graph and the same task metric.

**Result, replicated across all 25 included queries without exception**: zero edges are ever marginally necessary (candidate graphs ranged from 38 to 76 edges depending on corruption level and number of flips; necessary count was 0 in every case), while cumulative discovery reported a nonempty circuit on every query (2 to 5 edges). In the single-query case examined in detail, the three edges cumulative discovery retained were all incoming edges to one specific unit, load-bearing only because, by the time the greedy procedure reached them, every other contributor had already been pruned away; no edge, tested in isolation against the full unablated network, individually determined the outcome.

**Interpretation.** This is not a discovery failure and not a benchmark malfunction; it is a direct, replicated demonstration that marginal and cumulative necessity are different questions, and that a fully redundant, majority-vote architecture can produce a completely empty answer to the first while still producing a nonempty answer to the second. Precision and recall are undefined in the ordinary sense here (ground truth has no positives), and reporting them as simple error rates would misstate what happened; the divergence itself is the finding.

### 6.2 SDM: A Stable, Informative Precision/Recall Profile

For each of the 20 training queries, we computed the same two quantities against the address-decoder layer's full structural candidate graph (5120 edges: every nonzero entry of the address-decoder weight matrix, not truncated to a smaller candidate set in advance).

**Ground truth was never empty.** Necessary-edge counts ranged from 27 to 35 across the 20 queries (mean 29.5), confirming that this architecture's discrete top-k selection produces genuinely, individually necessary edges, in direct contrast to Hopfield's fully redundant computation.

**Discovery results, aggregated across all 20 queries:**

| Metric | Mean | Range |
|---|---|---|
| Node precision | 0.949 | 0.846–1.000 |
| Node recall | 0.921 | 0.846–1.000 |
| Edge precision | 0.912 | 0.778–1.000 |
| Edge recall | 0.574 | 0.483–0.643 |

Node-level precision and recall are both high and tightly clustered, several queries reach exactly 1.000 on both. Edge recall is the one metric that is consistently and stably lower, never approaching the node-level scores, across every query, not the product of noise or a single unlucky trial.

**Interpretation.** Discovery reliably identifies *which hard locations* participate in the true circuit (high node precision and recall) but recovers only around half of the *specific edges* feeding into those locations (stable edge recall near 0.57). A plausible mechanism, not asserted here as confirmed but consistent with the greedy, order-dependent pruning procedure: once one edge into a given hard location is pruned early because a redundant alternative edge into the same location keeps the metric intact, that pruned edge is never re-tested against the true, fully-unablated baseline, so genuinely necessary edges can be permanently and systematically missed whenever more than one edge feeds the same necessary location. This is a specific, structural, reproducible limitation of cumulative greedy discovery, not a general failure of the method, and it is exactly the kind of finding an independently verified ground truth benchmark is positioned to surface.

### 6.3 Cross-Instance Comparison

The same discovery machinery, applied to a verified ground truth, produces two qualitatively different outcomes depending on the underlying architecture's computational character:

| | Hopfield (redundant, majority-vote) | SDM (sparse, discrete cutoff) |
|---|---|---|
| Ground truth ever empty | 25/25 queries | 0/20 queries |
| Node precision | undefined | 0.949 |
| Node recall | undefined | 0.921 |
| Edge precision | undefined | 0.912 |
| Edge recall | undefined | 0.574 |

This is the paper's central empirical point made concrete: precision and recall are not properties of a discovery method in isolation, they are properties of the *relationship* between a discovery method and the architecture it is applied to, and that relationship is only visible once ground truth has been independently specified and verified.

---

## 7. Toward a Standing Benchmark Suite

The two instances reported here, a Hopfield network and an SDM address-decoder layer, are proposed as the first two entries in a broader suite (working name: GroundTruthMI), rather than the paper's complete contribution. Natural extensions, not attempted here: the SDM data-store (MAX-Hebbian) layer, as a follow-up to the address-decoder result in Section 6.2; Hierarchical Temporal Memory; small, hand-specified spiking neural networks. Additional discovery methods beyond the generic ACDC-style procedure used here, activation patching, causal abstraction, edge attribution, could be evaluated against the same verified ground truths without requiring new architecture instances. In spirit, a matured version of this suite would function analogously to GLUE, ImageNet, or HELM for their respective fields: a standing, reusable calibration resource rather than a single paper's result.

---

## 8. Discussion

**What this buys.** An independently verifiable calibration standard for mechanistic analysis methods, filling a gap that exists because circuit discovery currently checks itself against toy tasks and internal agreement rather than against ground truth specified and verified independently of the discovery process.

**What this does not buy.** No claim is made that either instance's transparency scales to transformer-level interpretability, or that hybridizing either architecture toward higher task performance would preserve the verification guarantees reported in Section 5. The value of this protocol is calibration, checking whether a discovery method's assumptions match a given class of mechanism, not a solution to interpretability at scale.

**Relation to the stated open problem.** Ignat et al. (2023) name mechanistic interpretability as an area not solved by current large language models. This paper does not claim to establish why that is true; it argues that one obstacle to evaluating progress on that open problem is the scarcity of benchmarks with independently specified and verified ground truth, and offers two such benchmarks, with the cross-instance comparison in Section 6.3 as evidence of what independently verified ground truth reveals that toy-task validation alone would not.

---

## 9. Limitations

- Two discovery methods, not several: only a generic ACDC-style procedure was evaluated. Activation patching, causal abstraction, and edge attribution were named as intended future targets (Section 7) but not run here.
- Two architecture instances, not a full suite: the SDM instance is scoped to its address-decoder layer only; the data-store layer's MAX-Hebbian nonlinearity was deliberately excluded from this paper's scope.
- Query counts, while replicated (25 and 20 respectively), remain modest; larger-scale replication would strengthen confidence that the reported means and ranges are representative rather than incidental to the specific pattern sets used.
- No claim of relevance to transformer-scale interpretability beyond providing a calibration tool for methods later applied there.

---

## 10. Conclusion

This paper argues that mechanistic interpretability should complement post-hoc circuit discovery with benchmark architectures whose mechanisms are specified independently of learning and whose actual behavior has been independently verified to match that specification, a distinction this paper treats as load-bearing throughout, not a formality. A classical Hopfield network and a rank-order SDM address-decoder layer provide two such verified benchmarks, and a generic reimplementation of ACDC's core algorithm, evaluated against both, produces a result neither instance alone would have made visible: identical discovery machinery can diverge completely from ground truth on one architecture while producing a stable, interpretable precision/recall profile on another. Future work extends this suite to additional architectures and additional discovery methods.

---

## References

- Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. arXiv:2304.14997.
- Elhage, N., et al. (2022). Toy Models of Superposition. arXiv:2209.10652.
- Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal Abstractions of Neural Networks. NeurIPS.
- Geiger, A., et al. (2022). Inducing Causal Structure for Interpretable Neural Networks. ICML.
- Geiger, A., Potts, C., & Icard, T. (2023). Causal Abstraction for Faithful Model Interpretation. arXiv:2301.04709.
- Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. EMNLP-IJCNLP.
- Hewitt, J., & Manning, C. D. (2019). A Structural Probe for Finding Syntax in Word Representations. NAACL-HLT.
- Hopfield, J. J. (1982). Neural Networks and Physical Systems with Emergent Collective Computational Abilities. PNAS, 79(8), 2554-2558.
- Ignat, O., Jin, Z., et al. (2023). Has It All Been Solved? Open NLP Research Questions Not Solved by Large Language Models. arXiv:2305.12544.
- Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.
- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress Measures for Grokking via Mechanistic Interpretability. arXiv:2301.05217.
- Thorpe, S. (1990). Spike Arrival Times: A Highly Efficient Coding Scheme for Neural Networks.
- Wang, K. R., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2023). Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small. ICLR.
