# Paper Snapshot (Track-A)

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
