# Paper Snapshots and Active Branches

**Frozen version:** `paper1-trackA-v2`

This directory contains the frozen Track‑A manuscript snapshot used for the current technical report / workshop‑style submission.

## Contents
- `paper1-trackA-v1.pdf` — frozen PDF corresponding to tag `paper1-trackA-v1`
- `paper1-trackA-v2.pdf` — not yet archived in this folder (build from tag `paper1-trackA-v2`)

## Status
This snapshot is **frozen**. Do not edit or replace the PDF without creating a new tag and snapshot.

## Provenance
- Tag: `paper1-trackA-v2`
- Build command (from repo root):

```bash
pdflatex latex/manuscript_sample.tex
BIBINPUTS=latex: bibtex manuscript_sample
pdflatex latex/manuscript_sample.tex
pdflatex latex/manuscript_sample.tex
```

## paper1-phase1.5-submission-v1 (Frozen)

This is the submission snapshot for Phase 1.5.

Includes:
- Final manuscript with ΔF1 panel (Proposed − SparseNoGraph)
- Repeatable low-to-moderate λ2 regime claim
- Explicit non-claim (no universal recovery improvement)
- Robustness across λ1 and ρ
- Reproducible plotting script

Tag: `paper1-phase1.5-submission-v1`  
Branch: `paper1-phase1.5-submission`

This version is frozen and should not be modified.

## Phase 2 (Active)

This is the active Phase 2 branch with real-data sanity check and grid topology additions.

Tag: (none; active development)  
Branch: `codex/paper1-phase2`

Do not edit frozen snapshots; make Phase 2 changes only on the active branch.
