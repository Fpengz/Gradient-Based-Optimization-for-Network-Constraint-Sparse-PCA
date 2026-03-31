# Graph-Smooth Sparse Orthogonal PCA — Program Guide and Execution Index

## Purpose

This document is the **program-level guide** for the research project on **graph-smooth sparse orthogonal PCA**. It is written to give human collaborators and coding/research agents a stable context for building the research pipeline, running iterations, and updating the manuscript without drifting from scope.

This is **not** the final paper.  
This is the **execution map**.

The current project goal is to produce a **credible base paper** built around a narrow, defensible claim:

> We study sparse, graph-smooth, explicitly orthogonal multi-component PCA, and evaluate when this improves structure and interpretability relative to strong baselines.

---

# Index Map

This file should act as the root index. Other documents should be linked from here.

## Core Documents

- `00_program_guide.md`  
  This file. Master context, execution logic, guardrails, pipeline, and iteration rules.

- `01_problem_scope.md`  
  Frozen paper claim, exact gaps tackled, non-goals, and scope boundaries.

- `02_related_work_map.md`  
  Literature map, paper clusters, counterexamples, claim boundaries, and citation positioning.

- `03_formulation.md`  
  Exact mathematical objective(s), notation, assumptions, and design choices.

- `04_algorithm.md`  
  Solver design, update rules, implementation notes, convergence targets, and pseudocode.

- `05_baselines.md`  
  Baseline matrix, inclusion criteria, implementation status, and comparison policy.

- `06_synthetic_protocol.md`  
  Synthetic data generation plan, graph families, support regimes, metrics, and sweeps.

- `07_ablation_plan.md`  
  Mandatory ablations, what each ablation tests, and interpretation rules.

- `08_real_data_plan.md`  
  Real datasets, graph sources, preprocessing plans, risks, and inclusion criteria.

- `09_results_schema.md`  
  Standardized output format for tables, figures, logs, experiment summaries, and manuscript insertion.

- `10_theory_roadmap.md`  
  Theorem ladder, proof priorities, assumptions, and what not to overclaim.

- `11_manuscript_map.md`  
  Paper section map, contribution structure, what evidence supports each claim.

- `12_iteration_log.md`  
  Chronological record of major design decisions, failures, updates, and next actions.

---

# 1. Executive Summary

This project starts from the observation that current graph-aware PCA methods, including graph-regularized PCA, do not yet fully solve the specific problem we care about:

- sparse loadings,
- smoothness over a known feature graph,
- explicit simultaneous orthogonality for multiple components,
- a clean optimization framing suitable for a modern, defensible paper.

The base paper should **not** claim:
- that graph-aware PCA is new,
- that graph+sparsity PCA is new,
- that nonsmooth manifold optimization is new,
- that this is the first time graphs, sparsity, and manifolds have ever been combined.

Instead, the base paper should claim:

> the specific PCA intersection of sparse loadings, feature-graph smoothness, and explicit simultaneous orthogonality remains comparatively underexplored, especially under a clean optimization and empirical framework.

---

# 2. Harness Engineering Principles for This Project

This project should be run using **harness engineering** rather than freeform wandering.

That means every work cycle must follow:

1. **clear goal**
2. **clear scope**
3. **explicit acceptance criteria**
4. **stable checkpoint**
5. **validation**
6. **gap analysis**
7. **next iteration**

Agents should not “explore forever.”  
Agents should work inside a bounded task harness.

## 2.1 Task Contract Template

Every substantial task should start with a contract:

### Goal
What concrete output should exist after this task?

### Scope
What is included and excluded?

### Inputs
Which docs, formulas, code modules, and papers are the source of truth?

### Deliverables
What files, sections, plots, tables, or code should be produced?

### Acceptance Criteria
How do we decide whether the task is done?

### Validation
How will the result be checked?

### Risks
What could go wrong or drift?

### Next Step
What does this unlock?

Agents should always write or update this contract before starting major work.

## 2.2 Freeze Before Build

Do not let agents code against a moving target.

Before implementation, freeze:
- paper claim,
- target gaps,
- objective,
- solver family,
- baseline list,
- experiment protocol.

Only then build.

## 2.3 Checkpoint Culture

Each work unit should end with a checkpoint:
- what was built,
- what was validated,
- what failed,
- what changed,
- what remains blocked.

No silent iteration.

## 2.4 Gap-Driven Iteration

Iteration must be based on observed deficiencies, not vague ambition.

Examples:
- baseline too weak,
- synthetic design not discriminative,
- solver unstable,
- orthogonality not preserved,
- graph prior not helping,
- manuscript claim too broad.

Each new iteration should be justified by a concrete observed gap.

---

# 3. Frozen Base-Paper Claim

## Working Claim

> We study sparse, graph-smooth, explicitly orthogonal multi-component PCA and evaluate when it improves support structure and interpretability relative to strong baselines.

This is the current paper-1 claim.

## Paper-1 Target Gaps

Paper 1 should tackle only **1–2 gaps**.

### Primary Gap
**Lack of explicit simultaneous orthogonality in graph-aware sparse PCA pipelines.**

### Secondary Gap
**Need for a more defensible optimization framing than naive threshold-then-project approaches.**

That means paper 1 is about:
- orthogonal multi-component extraction,
- graph-smooth sparse estimation,
- a practical optimization scheme,
- strong synthetic evidence,
- modest theory.

## Non-Goals for Paper 1

Do **not** treat these as core contributions yet:
- joint graph learning,
- full statistical consistency theory,
- deep KL geometry as main novelty,
- universal superiority over all prior work,
- many real-world datasets,
- a grand theory paper.

These may become paper 2 or later extensions.

---

# 4. Core Research Positioning

## What This Project Is

This project is a **targeted integration paper** at the intersection of:
- sparse PCA,
- graph-aware PCA,
- simultaneous orthogonal multi-component extraction,
- structured manifold/proximal optimization.

## What This Project Is Not

This project is not:
- “the first graph-aware PCA paper,”
- “the first graph+sparsity PCA paper,”
- “the first manifold optimization paper with structure,”
- “the final theory of graph-constrained sparse PCA.”

## Safe Positioning Statement

> Existing literature covers several pairwise combinations of sparse loadings, graph-aware regularization, orthogonality-preserving optimization, and structured nonsmooth methods. The specific combination of feature-graph smoothness, sparse multi-component loadings, explicit simultaneous orthogonality, and a practical PCA-specific optimization framework remains comparatively underexplored.

Use this as the default framing unless later evidence forces refinement.

---

# 5. Program Architecture

The program should be run as a layered pipeline.

## Layer 1: Research Framing
- scope
- literature map
- gap analysis
- contribution freeze

## Layer 2: Mathematical Design
- objective
- constraints
- variants
- assumptions
- solver choice

## Layer 3: Experimental Infrastructure
- synthetic generator
- graph generator
- metrics
- baselines
- ablations
- result logging

## Layer 4: Evaluation
- synthetic core study
- stress tests
- graph corruption tests
- runtime tests
- optional real-data study

## Layer 5: Theory
- stationarity-oriented results
- sufficient decrease / boundedness / convergence-to-stationarity
- only after method is stable

## Layer 6: Manuscript Integration
- methods
- setup
- results
- discussion
- limitations
- claim tightening

Agents should know which layer they are operating in.

---

# 6. Recommended Document Dependencies

## `01_problem_scope.md`
Must be finalized before:
- `03_formulation.md`
- `05_baselines.md`
- `11_manuscript_map.md`

## `03_formulation.md`
Must be finalized before:
- `04_algorithm.md`
- `06_synthetic_protocol.md`
- `07_ablation_plan.md`

## `05_baselines.md`
Must be finalized before:
- synthetic runs
- results tables
- manuscript results claims

## `06_synthetic_protocol.md`
Must be finalized before:
- any large experiment sweep
- ablations
- result interpretation

## `09_results_schema.md`
Must be finalized before:
- large experiment runs
- figure generation
- manuscript table writing

This dependency structure prevents drift and rework.

---

# 7. Current Recommended Mathematical Direction

## Preferred Formulation for Paper 1

Use the **split-variable formulation** as the primary method candidate:

$$
\min_{A^\top A = I_r,\; B}
-\operatorname{tr}(A^\top \widehat{\Sigma} A)
+ \lambda_1 \|B\|_1
+ \lambda_2 \operatorname{tr}(B^\top L B)
+ \frac{\rho}{2}\|A-B\|_F^2.
$$

## Why This Is Preferred

Compared with the direct Stiefel + nonsmooth formulation, this split form:
- separates orthogonality and sparse graph regularization,
- is easier to explain,
- is easier to optimize in a controlled way,
- is easier to analyze modestly,
- avoids pretending that naive threshold-plus-projection is enough.

## Alternative Formulation

The direct Stiefel form remains a useful comparison or later variant:

$$
\max_{V^\top V = I}
\operatorname{tr}(V^\top \widehat{\Sigma} V)
- \lambda_1 \|V\|_1
- \lambda_2 \operatorname{tr}(V^\top L V).
$$

But paper 1 should not depend on this being the primary algorithmic path unless implementation becomes cleaner than expected.

---

# 8. Current Recommended Solver Direction

## Paper-1 Solver Family

Use an alternating method:

### A-step
Optimize the orthogonal variable on the Stiefel manifold:
- manifold gradient step,
- retraction,
- stable orthogonality enforcement.

### B-step
Optimize the sparse graph-smooth variable in Euclidean space:
- proximal gradient,
- primal-dual,
- or another first-order graph-coupled sparse solver.

### Outer Loop
Alternating minimization with monitored decrease and convergence diagnostics.

## What Not to Do First

Do not begin with:
- semismooth Newton as the whole identity,
- fancy second-order manifold machinery,
- huge theoretical ambitions before experiments work.

## Paper-1 Success Criteria for the Solver

The first solver is good enough if:
- orthogonality is preserved,
- objective decreases reasonably,
- runtime is acceptable,
- recovered supports are interpretable,
- graph priors visibly matter in the intended regimes.

---

# 9. Baseline Matrix Policy

This project must not compare only against PCA and SparsePCA.

## Required Baseline Categories

### Unstructured
- PCA
- standard sparse PCA

### Orthogonal Sparse
- one joint orthogonal sparse PCA baseline

### Graph-Aware
- GR-PCA

### Graph + Sparse
- at least one graph+sparsity PCA variant

### Our Ablations
- no graph term
- no sparsity term
- deflation instead of joint extraction

## Why This Matters

Without this baseline matrix, the paper will overstate novelty and under-test its own contribution.

---

# 10. Synthetic Study as the Core Validation Harness

The synthetic study is the **main proving ground** for paper 1.

## Main Research Questions for Synthetic Study

- When the graph prior is correct, does graph-smooth sparse orthogonal PCA recover more meaningful structure?
- What tradeoff does graph regularization introduce between variance and structural coherence?
- Does joint orthogonal extraction help compared with deflation?
- How sensitive is performance to graph corruption?

## Minimum Synthetic Graph Families

- chain
- grid
- stochastic block model

Optional:
- random geometric graph

## Minimum Support Regimes

- connected support
- nearly connected support
- disconnected support

## Minimum Component Regimes

- disjoint support across components
- shared support across components

## Minimum Sweeps

- sample size
- dimension
- sparsity level
- SNR
- graph corruption
- number of components
- regularization parameters

---

# 11. Ablation Study Policy

Ablations are mandatory, not optional.

## Mandatory Ablations

### No Graph
Set graph regularization to zero.

Purpose:
- test whether graph smoothness is doing useful work.

### No Sparsity
Set sparsity regularization to zero.

Purpose:
- test whether graph structure alone is enough.

### Deflation Instead of Joint Extraction
Replace simultaneous orthogonal extraction with sequential extraction.

Purpose:
- test whether joint orthogonality is actually valuable.

### Graph Corruption
Perturb or randomize the graph.

Purpose:
- test robustness to graph misspecification.

### Shared vs Disjoint Supports
Purpose:
- understand when row-structured / joint behavior matters.

## Ablation Interpretation Rule

Do not claim a modeling ingredient is useful unless its ablation result supports that claim.

---

# 12. Theory Roadmap

Theory should support paper 1, not dominate it.

## Realistic Theorem Ladder

1. lower bounded objective
2. bounded iterates
3. sufficient decrease
4. vanishing successive differences
5. stationarity of accumulation points
6. optional KL-based full-sequence convergence

## Not Paper-1 Priorities

- global optimality
- strong sample complexity results
- graph recovery theory
- full statistical consistency
- universal superiority guarantees

If theory work blocks experiments, theory is too ambitious for the current iteration.

---

# 13. Real Data Policy

Real data should come **after** synthetic evidence is convincing.

## Rule
No real-data expansion until:
- synthetic protocol is stable,
- metrics are stable,
- baselines are implemented,
- the method shows a coherent advantage in at least one synthetic regime.

## First Real Domain Recommendation
- gene expression + pathway/PPI graph

## Why
- graph prior is meaningful,
- sparse interpretation matters,
- the application matches the motivating story well.

Only include one or two real datasets in paper 1.

---

# 14. Standard Output Requirements for Agents

Every agent-produced artifact should be easy to integrate into the manuscript pipeline.

## For Code Tasks
Output:
- changed files
- implementation summary
- acceptance checks
- unresolved issues

## For Experiment Tasks
Output:
- config used
- metrics table
- plots
- interpretation summary
- failures or anomalies

## For Writing Tasks
Output:
- target section
- exact claims made
- supporting evidence required
- risk of overclaiming

## For Theory Tasks
Output:
- theorem statement
- assumptions
- proof sketch
- blocked steps
- dependency on algorithm details

---

# 15. Acceptance Criteria for Paper-1 Readiness

Paper 1 is ready to draft seriously only when all of the following are true:

## Framing
- claim is frozen
- scope is frozen
- related work is literature-tight

## Method
- objective is fixed
- solver is fixed
- implementation is stable

## Evaluation
- baseline matrix is complete
- synthetic study is complete
- mandatory ablations are complete
- one real-data study exists or is explicitly deferred

## Theory
- minimum convergence story exists
- no unsupported theory claims remain

## Writing
- each claim maps to evidence
- limitations section is honest
- novelty language is restrained

---

# 16. Iteration Loop

This project should iterate through the following loop.

## Iteration Loop

### Step 1
Freeze current version:
- claim
- method
- baseline set
- synthetic protocol

### Step 2
Run experiments.

### Step 3
Validate against acceptance criteria.

### Step 4
Identify concrete gaps:
- performance gap
- solver instability
- weak baseline comparison
- unclear interpretation
- unsupported claim

### Step 5
Choose one revision target.

### Step 6
Implement that revision only.

### Step 7
Rerun the relevant pipeline.

### Step 8
Update logs and manuscript map.

Agents should not revise multiple major axes at once unless explicitly instructed.

---

# 17. Practical Work Queue for the Next Phase

## Immediate Priorities

### A. Freeze Paper-1 Scope
Create / finalize:
- `01_problem_scope.md`

### B. Freeze Objective and Solver
Create / finalize:
- `03_formulation.md`
- `04_algorithm.md`

### C. Freeze Baselines
Create / finalize:
- `05_baselines.md`

### D. Freeze Synthetic Protocol
Create / finalize:
- `06_synthetic_protocol.md`

### E. Freeze Ablations
Create / finalize:
- `07_ablation_plan.md`

Only after these should agents begin large implementation or experiment sweeps.

---

# 18. Agent Instructions Summary

Any agent working on this project should follow these rules:

1. Do not broaden the paper claim without explicit approval.
2. Do not add new gaps as central contributions unless they replace old ones.
3. Do not treat literature exploration as progress unless it changes scope or implementation.
4. Do not implement against an unfrozen objective.
5. Do not run large experiments without a frozen protocol and result schema.
6. Do not write strong novelty claims.
7. Always report limitations, uncertainties, and blockers.
8. Prefer a working simple solver over a fancy unstable one.
9. Prefer one strong synthetic pipeline over many scattered experiments.
10. Keep paper 1 narrow and finishable.

---

# 19. One-Sentence Project Identity

Use this when a concise description is needed:

> This project studies how to extract sparse, mutually orthogonal principal components that are also smooth on a known feature graph, using a practical PCA-specific optimization framework and a disciplined synthetic-to-real evaluation pipeline.

---

# 20. Final Guidance

This project will succeed if it behaves like an engineering research program, not like an endless ideation loop.

The right near-term focus is:

- freeze paper 1,
- implement the base formulation,
- build the baseline matrix,
- validate on synthetic data,
- run ablations,
- then iterate.
