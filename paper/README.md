# Paper Snapshots

This directory contains frozen manuscript snapshots and their provenance.

## Track A (Frozen)

**Frozen version:** `paper1-trackA-v1`

Contents:
- `paper1-trackA-v1.pdf` — frozen PDF corresponding to tag `paper1-trackA-v1`

Status: **frozen**. Do not edit or replace the PDF without creating a new tag and snapshot.

Provenance:
- Tag: `paper1-trackA-v1`

## Track B Phase 1.5 (Frozen)

**Frozen version:** `paper1-phase1.5-submission-v1`

Contents:
- `paper1-trackB-phase1.5-v1.pdf` — frozen PDF corresponding to tag `paper1-phase1.5-submission-v1`

Status: **frozen**. Do not edit or replace the PDF without creating a new tag and snapshot.

Provenance:
- Tag: `paper1-phase1.5-submission-v1`
- Branch: `paper-trackB-phase1.5-submission`

## Build command (from repo root)

```bash
pdflatex latex/manuscript_sample.tex
BIBINPUTS=latex: bibtex manuscript_sample
pdflatex latex/manuscript_sample.tex
pdflatex latex/manuscript_sample.tex
```
