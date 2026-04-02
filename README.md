# GRPCA-GD

Graph-smooth sparse orthogonal PCA with a split-variable solver and a synthetic-first evaluation. This repository contains the Track-A manuscript, code, and reproducible synthetic experiments.

## Repo Structure and Worktrees (Read This First)
This project uses Git worktrees. Each worktree is a full checkout of the same repo, tied to a specific branch.

Current worktrees:
- **Main repo:** `/Users/zhoufuwang/Projects/GRPCA-GD` (baseline reference; avoid editing for active Track A changes)
- **Active Phase 2 worktree:** `/Users/zhoufuwang/Projects/GRPCA-GD/.worktrees/paper-trackA-phase2-final`  
  Branch: `paper-trackA-phase2-final` (authoritative Track A paper state)
- **Historical worktree:** `/Users/zhoufuwang/Projects/GRPCA-GD/.worktrees/paper-trackA-phase1-revision`  
  Branch: `paper-trackA-phase1-revision` (legacy snapshot)

Naming convention (must be adhered to for both branch and folder names):
- Worktree directories: `GRPCA-GD/.worktrees/<branch-name>`
- Branch names: `paper-<track>-<phase>-<status>` (e.g., `paper-trackA-phase2-final`)
- Historical/scratch branches must still follow the same naming scheme.

If you are working on Track A Phase 2, always use:
`/Users/zhoufuwang/Projects/GRPCA-GD/.worktrees/paper-trackA-phase2-final`

## Methods Compared (Paper-1)
- PCA (dense baseline)
- Minimal A-ManPG (external sparse orthogonal baseline, no graph)
- SparseNoGraph (in-family ablation with \(\lambda_2=0\))
- Proposed (sparse + graph-smooth + joint orthogonality)

## Experiments Included
- Chain robustness (seeds 0–4)
- SBM robustness (seeds 0–4)
- SBM \(\lambda_2\) sweep (single seed)

## Quickstart

Run a smoke test (r=1, rho=5.0, eta_A=0.05):

```bash
uv run python main.py configs/smoke_r1.yaml
```

Run the small r=3 config:

```bash
uv run python main.py configs/small_r3.yaml
```

## Reproduce Paper Results

1) Run robustness configs (chain + SBM, seeds 0–4):

```bash
for s in 0 1 2 3 4; do
  uv run python main.py configs/robust_chain_seed${s}.yaml
  uv run python main.py configs/robust_sbm_seed${s}.yaml
 done
```

2) Run SBM \(\lambda_2\) sweep:

```bash
for val in 0p00 0p05 0p10 0p20 0p50; do
  uv run python main.py configs/sbm_lambda2_${val}.yaml
 done
```

3) Regenerate the sweep panel figure:

```bash
uv run python - <<'PY'
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

root = Path('outputs')
order = ['0p00','0p05','0p10','0p20','0p50']
lams = [0.0,0.05,0.1,0.2,0.5]

support_f1 = []
smooth_norm = []
expl_var = []

for tag in order:
    metrics = json.loads((root / f'sbm_lambda2_{tag}' / 'metrics.json').read_text())
    proposed = metrics['Proposed']
    support_f1.append(proposed['support_metrics']['union']['f1'])
    smooth_norm.append(proposed['graph_smoothness_norm_trueL'])
    expl_var.append(proposed['shared_explained_variance'])

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].plot(lams, support_f1, marker='o')
axes[0].set_title('Support F1')
axes[0].set_xlabel('lambda2')

axes[1].plot(lams, smooth_norm, marker='o')
axes[1].set_title('Graph Smoothness (norm)')
axes[1].set_xlabel('lambda2')

axes[2].plot(lams, expl_var, marker='o')
axes[2].set_title('Shared Explained Variance')
axes[2].set_xlabel('lambda2')

plt.tight_layout()
Path('figures').mkdir(exist_ok=True)
plt.savefig('figures/sbm_lambda2_sweep_panel.png', dpi=200)
PY
```

4) Update tables in `latex/manuscript_sample.tex` using the newly generated outputs. The tables in the manuscript should match the means/stds computed from the `metrics.json` files in `outputs/`.

## Manuscript Build

Compile **from the repo root** (not from inside `latex/`):

```bash
pdflatex latex/manuscript_sample.tex
BIBINPUTS=latex: bibtex manuscript_sample
pdflatex latex/manuscript_sample.tex
pdflatex latex/manuscript_sample.tex
```

## Outputs
Each run writes to `outputs/<run_name>/`:
- `artifacts.npz` and `artifacts.json`
- `manifest.json`
- `metrics.json` and `metrics.csv`
- `plots/` convergence traces

## Paper Snapshot

The frozen Track-A manuscript PDF is stored at:
- `paper/paper1-trackA-v1.pdf`
