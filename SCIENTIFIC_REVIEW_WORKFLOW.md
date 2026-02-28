# Scientific Review Workflow

This document defines a practical review workflow for the paper and experiments in this repository.

Primary targets:

- `doc/latex/main.tex`
- `README.md`
- `REPRODUCIBILITY.md`
- `EXPERIMENTS.md`
- `AUDIT_REPORT_METHOD.md`
- `AUDIT_REPORT_CONVERGENCE.md`
- `PERFORMANCE_REPORT_V0.md`
- `PERFORMANCE_REPORT_V1.md`
- `scripts/run_experiment.py`
- `scripts/run_sweep.py`
- `scripts/reproduce_paper_artifacts.py`

## Recommended skill set

Best focused package:

- `K-Dense-AI/claude-scientific-writer`

Best broader research package:

- `K-Dense-AI/claude-scientific-skills`

Install first:

1. `scientific-writing`
2. `peer-review`
3. `research-lookup`
4. `citation-management`

Install next:

1. `literature-review`
2. `venue-templates`
3. `scientific-critical-thinking`
4. `scholar-evaluation`

## Review sequence

### 1. Claim audit

Goal:

- verify every mathematical and empirical claim in `doc/latex/main.tex`

Use:

- `research-lookup`

Check:

- theorem statements
- convergence claims
- baseline algorithm descriptions
- references for SPCA, graph regularization, ManPG, GPower, Elastic Net SPCA

Outputs:

- unsupported claims list
- outdated references list
- missing citation list

## 2. Reviewer pass

Goal:

- read the paper as a methods reviewer rather than as the author

Use:

- `peer-review`

Check:

- novelty statement
- fairness of baseline comparisons
- whether theory matches implementation
- whether ablations justify the proposed optimizer
- whether conclusions are stronger than the evidence

Outputs:

- major issues
- moderate issues
- minor issues

## 3. Technical rigor pass

Goal:

- test the internal logic of the paper

Use:

- `scientific-critical-thinking`
- `scholar-evaluation`

Check:

- consistency of objective definitions
- sign conventions in optimization problems
- notation consistency across sections
- whether proof assumptions match code behavior
- whether theorems only cover baseline PG and not MASPG-CAR

Outputs:

- logic gaps
- overclaims
- wording that should be weakened or made precise

## 4. Related work pass

Goal:

- tighten the literature framing around network-constrained SPCA

Use:

- `literature-review`
- `research-lookup`

Focus areas:

1. sparse PCA
2. graph-regularized PCA
3. generalized power methods
4. proximal gradient and manifold proximal gradient methods
5. graph signal processing interpretation of Laplacian smoothness
6. structured sparsity and row-sparse formulations

Outputs:

- grouped related-work notes
- missing canonical citations
- revised related-work paragraph structure

## 5. Citation and bibliography cleanup

Goal:

- make the bibliography defensible and consistent

Use:

- `citation-management`

Check:

- duplicate entries
- missing author, venue, year, or DOI fields
- citation style consistency
- uncited references
- uncited claims

Outputs:

- cleaned BibTeX database
- consistent citation placement in `doc/latex/main.tex`

## 6. Venue formatting pass

Goal:

- adapt the draft to the target venue or archive format

Use:

- `venue-templates`

Check:

- title and abstract length
- theorem, algorithm, figure, and table formatting
- bibliography style
- appendix placement
- reproducibility statement requirements

Outputs:

- venue compliance checklist
- required source edits

## 7. Final prose pass

Goal:

- tighten the writing after the paper is technically correct

Use:

- `scientific-writing`

Check:

- remove inflated language
- shorten repetitive exposition
- keep claims proportional to evidence
- improve abstract, introduction, and discussion flow
- prefer precise wording over promotional wording

Outputs:

- submission-ready prose pass

## Repository-specific review checklist

### Paper and code alignment

- Does `doc/latex/main.tex` state the single-component objective exactly as implemented?
- Are the signs of the covariance and Laplacian terms correct everywhere?
- Is the constraint written as an `l2` ball when the implementation uses radial projection?
- Are baseline PG guarantees clearly separated from MASPG-CAR practical behavior?

### Experiments and reproducibility

- Do `scripts/run_experiment.py` and `scripts/run_sweep.py` reproduce the tables claimed in the paper?
- Are random seeds, graph families, and support construction documented?
- Are runtime, F1, explained variance, LCC ratio, and Laplacian energy all logged consistently?
- Are comparison methods configured fairly under the same data-generation protocol?

### Performance claims

- Does `PERFORMANCE_REPORT_V1.md` support any speed or accuracy claim in the paper?
- If MASPG-CAR is faster, is the tradeoff in explained variance stated?
- If ProxQN is not materially stronger, is that limitation stated honestly?

### Writing quality

- Are long sentences split where they obscure the core point?
- Do equations introduce all symbols before use?
- Are section transitions clear and necessary?
- Is the tone closer to a rigorous paper than to promotional copy?

## Minimal operating procedure

For each revision cycle:

1. Audit `doc/latex/main.tex` against the current reports.
2. Re-run the relevant experiment script if a quantitative claim changes.
3. Update the matching report before changing the prose.
4. Rebuild the PDF and check for formatting regressions.
5. Do one final reviewer-style pass before commit.

## Suggested local output files

When using the workflow, keep review artifacts in the repo root unless a venue-specific structure is introduced:

- `CLAIM_AUDIT.md`
- `PEER_REVIEW_NOTES.md`
- `RELATED_WORK_NOTES.md`
- `CITATION_CLEANUP.md`
- `VENUE_COMPLIANCE.md`

## Current priority order for this project

1. tighten claim to evidence alignment in `doc/latex/main.tex`
2. improve related work coverage around graph regularization and manifold optimization
3. clean bibliography and citation placement
4. state optimizer limitations precisely in the experiments section
5. adapt the draft to the intended submission format
