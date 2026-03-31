# Track B Brief: Strengthening Evidence for Graph-Aligned Benefit

## Objective
Upgrade the Track A empirical revision into a full-venue-ready submission by strengthening statistical support and expanding evidence for when graph regularization is beneficial, while preserving the paper's disciplined, topology-dependent framing.

## Hypotheses
- H1: In graph-aligned settings, moderate graph regularization preserves recovery within a small F1 margin while materially improving structural connectivity and smoothness.
- H2: The recovery–smoothness tradeoff is topology-dependent and stable across seeds, with SBM showing stronger recovery penalties than chain under the same regularization grid.
- H3: With sufficient seeds and significance testing, the graph-aligned benefit is defensible against the strongest sparse baseline (A-ManPG).

## Required Experiments
- Increase seed counts for all main comparisons and sweeps (target: 20+; minimum: 10 if compute-limited).
- Add paired significance tests across seeds for support F1, smoothness, and connectivity metrics.
- Report coupling gap statistics at termination (mean ± std) for all methods in the main comparison.
- Report runtime table (mean ± std) for main comparison and graph-aligned setting.
- Add empirical convergence plots (objective and coupling gap vs. iteration) for representative runs.
- Optional: add one additional comparator (graph+sparsity method) if feasible and clearly documented.

## Phase 1 Experiment Matrix
- Fixed: `p=200`, support size fixed (same as Track A), same baselines as Track A.
- Vary: seed, `lambda2`, decoy intensity.
- `lambda2` grid: {0, 0.05, 0.1, 0.2, 0.5}.
- Decoy intensity levels (fixed counts, not scaled with p):
- Decoy intensity low: `decoy_count=6`, `decoy_variance_factor=1.5`.
- Decoy intensity medium: `decoy_count=12`, `decoy_variance_factor=2.0`.
- Decoy intensity high: `decoy_count=18`, `decoy_variance_factor=2.5`.
- Control rule: sample decoys only from non-support nodes.
- Control rule: decoy locations fixed across methods within a seed.
- Control rule: only `lambda2` and decoy intensity vary within Phase 1.

## Phase 1 Acceptance Criteria
- There exists a low-to-moderate `lambda2` band where Proposed is competitive with A-ManPG/SparseNoGraph on F1.
- The advantage becomes more visible as decoy intensity increases.
- Connectivity and smoothness improve in a materially meaningful way.
- Effects are stable across seeds and supported by significance testing once seed counts are increased.

## Phase 1 Non-Goals
- Support-size sweeps or geometry changes beyond the decoy-intensity axis.
- Real-data validation or broader graph families before Phase 1 is complete.

## Acceptance Criteria
- Graph-aligned setting shows a statistically significant recovery advantage over the strongest sparse baseline, or a statistically supported "competitive recovery + structural gain" outcome with clear connectivity improvement.
- Multi-seed sweeps show stable topology-dependent tradeoffs with uncertainty bands and A-ManPG references.
- All new metrics and claims are supported by statistical tests and error bars.
- Manuscript claims remain bounded to tested settings; no universal superiority language.

## Non-Goals
- New theoretical convergence proofs.
- Broad real-data benchmarks without clear alignment to the graph prior.
- Major algorithm redesign or new optimization framework.
