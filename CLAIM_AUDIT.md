# Claim Audit

This note audits the current paper draft in `doc/latex/main.tex` against the repository reports and implementation.

## Status summary

- Objective and baseline PG algorithm: aligned with code
- Convergence theorem scope: aligned after prior audit work
- Experimental protocol: broadly aligned with scripts
- Evidence-to-claim alignment: needs tightening in a few places

## High priority issues

### 1. Empirical contribution overclaim

Current draft claim:

- the introduction contribution list states that the paper empirically demonstrates improved connected-support recovery relative to unstructured baselines

Why this is not yet supported:

- `PERFORMANCE_REPORT_V1.md` shows that under graph misspecification, `L1-SPCA-ProxGrad` has higher mean F1 than `NetSPCA-PG` and `NetSPCA-MASPG-CAR`
- LCC ratio is `1.0` for all listed methods in that report, so the current report does not establish a measured structural-coherence advantage in that setting

Required fix:

- weaken the claim so it becomes conditional on informative graphs or frame it as an evaluation goal rather than an established universal result

Status:

- fixed in the paper revision pass

### 2. Laplacian specification ambiguity

Current draft issue:

- the problem setup allows either combinatorial or normalized Laplacians without stating which is used by default in optimization and reporting

Why this matters:

- the source-of-truth audit requires the optimization and metrics to use the same Laplacian object
- the paper should make the default explicit

Required fix:

- state that the combinatorial Laplacian is the default unless an experiment explicitly states otherwise

Status:

- fixed in the paper revision pass

### 3. MASPG-CAR practical-performance wording

Current draft issue:

- the practical variant section says it improves wall-clock efficiency in a way that reads stronger than the current evidence

Why this needs tightening:

- current ablations show only a small runtime gain for MASPG-CAR over PG
- current ProxQN results do not show a material quality improvement over PG

Required fix:

- phrase runtime benefit as an empirical tendency or a design goal, not as a universal result

Status:

- fixed in the paper revision pass

## Moderate issues

### 4. Informative-graph assumption should appear earlier

Current draft issue:

- the paper says the Laplacian encourages coherent support, but the dependence on graph quality appears mostly in later discussion

Required fix:

- mention in the introduction and experiment framing that gains are conditional on graph informativeness

Status:

- fixed in the paper revision pass

### 5. Misspecification failure mode should be explicit

Current draft issue:

- the draft discusses graph misspecification, but does not directly say that unstructured methods can outperform graph-regularized methods when the graph is poor

Required fix:

- add an explicit caution in the experimental design or discussion section

Status:

- fixed in the paper revision pass

## Low priority issues

### 6. Dash-heavy prose

Current draft issue:

- several sentences use dash-heavy punctuation that weakens the tone

Required fix:

- replace with commas, parentheses, or shorter sentences

Status:

- partially addressed in the paper revision pass

## Claims currently supported

- the single-component objective and gradient formula
- the exact composite proximal map
- the baseline PG convergence theorem to critical points under the stated assumptions
- the positioning of MASPG-CAR as a practical variant rather than a proved method
- the use of chain, grid, random geometric graph, and SBM synthetic families

## Claims that must remain conditional

- improved support recovery over unstructured baselines
- improved structural coherence in practice
- runtime improvement from accelerations
- benefits from graph regularization under real-world graph quality uncertainty
