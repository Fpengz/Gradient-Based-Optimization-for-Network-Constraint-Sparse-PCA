# Paper-1 Claim-Audit Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align manuscript scope and theory to executed evidence, add a minimal A-ManPG baseline, and update experiments/tables/figures accordingly.

**Architecture:** Track-A empirical-method paper. Minimal A-ManPG baseline (no graph) added with shared metrics only; optimization discussion replaces convergence analysis; manuscript scope narrowed to chain + SBM + SBM \(\lambda_2\) sweep; figure paths stabilized; placeholder appendices removed.

**Tech Stack:** LaTeX manuscript, Python experiment runner, NumPy/SciPy/Matplotlib, PyYAML, pytest, uv.

---

## Task 1: Add Minimal A-ManPG Baseline (Code + Tests)

**Files:**
- Create: `/Users/zhoufuwang/Projects/GRPCA-GD/src/grpca_gd/amanpg.py`
- Modify: `/Users/zhoufuwang/Projects/GRPCA-GD/src/grpca_gd/runner.py`
- Modify: `/Users/zhoufuwang/Projects/GRPCA-GD/src/grpca_gd/__init__.py` (if exporting)
- Test: `/Users/zhoufuwang/Projects/GRPCA-GD/tests/test_amanpg.py`

- [ ] **Step 1: Write failing unit test for minimal A-ManPG solver**

```python
import numpy as np

from grpca_gd.amanpg import AmanpgConfig, solve_amanpg


def test_amanpg_returns_orthonormal_columns():
    rng = np.random.default_rng(0)
    p, r = 10, 3
    Sigma_hat = np.eye(p)
    A0 = rng.standard_normal((p, r))
    cfg = AmanpgConfig(lambda1=0.1, eta_A=0.05, max_iters=5, tol_obj=1e-12, tol_orth=1e-8)
    result = solve_amanpg(A0, Sigma_hat, cfg)

    gram = result.A.T @ result.A
    assert np.allclose(gram, np.eye(r), atol=1e-6)
```

- [ ] **Step 2: Run the test to confirm failure**

Run:

```bash
uv run pytest /Users/zhoufuwang/Projects/GRPCA-GD/tests/test_amanpg.py::test_amanpg_returns_orthonormal_columns -v
```

Expected: FAIL (module/function not found).

- [ ] **Step 3: Implement minimal A-ManPG solver**

Create `/Users/zhoufuwang/Projects/GRPCA-GD/src/grpca_gd/amanpg.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .objective import objective_terms
from .solver import soft_threshold
from .stiefel import qr_retraction, rgrad


@dataclass
class AmanpgConfig:
    lambda1: float
    eta_A: float
    max_iters: int
    tol_obj: float
    tol_orth: float


@dataclass
class AmanpgResult:
    A: np.ndarray
    history: Dict[str, np.ndarray]


def solve_amanpg(A0: np.ndarray, sigma_hat: np.ndarray, cfg: AmanpgConfig) -> AmanpgResult:
    A = A0.copy()
    history: Dict[str, List[float]] = {
        "total_objective": [],
        "negative_variance_term": [],
        "sparsity_penalty": [],
        "orthogonality_error": [],
        "sparsity_fraction": [],
    }

    prev_obj = None
    for _ in range(cfg.max_iters):
        grad = -2.0 * (sigma_hat @ A)
        A_step = A - cfg.eta_A * rgrad(A, grad)
        A_sparse = soft_threshold(A_step, cfg.eta_A * cfg.lambda1)
        A = qr_retraction(A_sparse)

        terms = objective_terms(A, A, sigma_hat, np.zeros_like(sigma_hat), cfg.lambda1, 0.0, 0.0)
        obj = terms["total_objective"]

        history["total_objective"].append(obj)
        history["negative_variance_term"].append(terms["negative_variance_term"])
        history["sparsity_penalty"].append(terms["sparsity_penalty"])
        history["orthogonality_error"].append(float(np.linalg.norm(A.T @ A - np.eye(A.shape[1]), ord="fro")))
        history["sparsity_fraction"].append(float(np.mean(np.abs(A) > 1e-8)))

        if prev_obj is not None:
            rel = abs(prev_obj - obj) / max(1.0, abs(prev_obj))
            if rel <= cfg.tol_obj and history["orthogonality_error"][-1] <= cfg.tol_orth:
                break
        prev_obj = obj

        if not np.isfinite(obj):
            raise FloatingPointError("Objective became non-finite")

    history_np = {k: np.array(v, dtype=float) for k, v in history.items()}
    return AmanpgResult(A=A, history=history_np)
```

- [ ] **Step 4: Re-run the unit test**

Run:

```bash
uv run pytest /Users/zhoufuwang/Projects/GRPCA-GD/tests/test_amanpg.py::test_amanpg_returns_orthonormal_columns -v
```

Expected: PASS.

- [ ] **Step 5: Add a lightweight sanity test for objective terms consistency**

Append to `/Users/zhoufuwang/Projects/GRPCA-GD/tests/test_amanpg.py`:

```python
from grpca_gd.objective import objective_terms


def test_amanpg_objective_terms_sum():
    rng = np.random.default_rng(1)
    p, r = 8, 2
    Sigma_hat = np.eye(p)
    A = rng.standard_normal((p, r))
    terms = objective_terms(A, A, Sigma_hat, np.zeros((p, p)), lambda1=0.1, lambda2=0.0, rho=0.0)
    total = terms["negative_variance_term"] + terms["sparsity_penalty"]
    assert np.isclose(total, terms["total_objective"])
```

- [ ] **Step 6: Run full test suite**

Run:

```bash
uv run pytest /Users/zhoufuwang/Projects/GRPCA-GD/tests -v
```

Expected: PASS.

- [ ] **Step 7: Commit A-ManPG code + tests**

```bash
git add /Users/zhoufuwang/Projects/GRPCA-GD/src/grpca_gd/amanpg.py \
  /Users/zhoufuwang/Projects/GRPCA-GD/src/grpca_gd/runner.py \
  /Users/zhoufuwang/Projects/GRPCA-GD/tests/test_amanpg.py

git commit -m "feat: add minimal A-ManPG baseline" 
```

---

## Task 2: Integrate A-ManPG Baseline in Runner + Artifacts

**Files:**
- Modify: `/Users/zhoufuwang/Projects/GRPCA-GD/src/grpca_gd/runner.py`
- Modify: `/Users/zhoufuwang/Projects/GRPCA-GD/src/grpca_gd/artifacts.py` (only if schema needs new fields)

- [ ] **Step 1: Wire baseline in runner**

Add import:

```python
from .amanpg import AmanpgConfig, solve_amanpg
```

Add minimal A-ManPG run after PCA init (use same A0, same lambda1, eta_A, max_iters, tol_obj, tol_orth):

```python
amanpg_cfg = AmanpgConfig(
    lambda1=cfg["lambda1"],
    eta_A=cfg["eta_A"],
    max_iters=cfg["max_iters"],
    tol_obj=cfg["tol_obj"],
    tol_orth=cfg["tol_orth"],
)
amanpg_result = solve_amanpg(A0, Sigma_hat, amanpg_cfg)
A_amanpg = amanpg_result.A
B_amanpg = A_amanpg
B_aligned_amanpg, perm_amanpg, signs_amanpg = _alignment(A_amanpg, B_amanpg, dataset.true_loadings)
amanpg_support = support_metrics(B_aligned_amanpg, dataset.true_supports)
```

Metrics entry (use shared metrics only; no raw objective comparisons):

```python
amanpg_eval = {
    "method_name": "A-ManPG",
    "objective_terms": objective_terms(
        A_amanpg,
        B_amanpg,
        Sigma_hat,
        np.zeros_like(L),
        cfg["lambda1"],
        0.0,
        0.0,
    ),
    "sparsity_fraction": sparsity_fraction(B_amanpg),
    "orthogonality_error": orthogonality_error(A_amanpg),
    "laplacian_energy": laplacian_energy(B_amanpg, np.zeros_like(L)),
    "support_metrics": amanpg_support,
    "graph_smoothness_raw_trueL": graph_smoothness_raw(B_amanpg, L),
    "graph_smoothness_norm_trueL": graph_smoothness_norm(B_amanpg, L),
    "shared_explained_variance": explained_variance(
        orthonormalize(B_amanpg), Sigma_hat
    ),
}
metrics_out["A-ManPG"] = amanpg_eval
```

Add A-ManPG artifacts for parity:

```python
arrays.update(
    {
        "amanpg_A": A_amanpg,
        "amanpg_B": B_amanpg,
        "amanpg_matching_perm": perm_amanpg,
        "amanpg_matching_signs": signs_amanpg,
        **{f"amanpg_history_{k}": v for k, v in amanpg_result.history.items()},
    }
)
```

- [ ] **Step 2: Update artifacts meta to record baseline list**

In `artifacts.json`, add a baseline list or note (if desired) such as:

```python
"baseline_methods": ["PCA", "A-ManPG", "SparseNoGraph"],
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest /Users/zhoufuwang/Projects/GRPCA-GD/tests -v
```

Expected: PASS.

- [ ] **Step 4: Commit runner changes**

```bash
git add /Users/zhoufuwang/Projects/GRPCA-GD/src/grpca_gd/runner.py

git commit -m "feat: integrate A-ManPG baseline metrics" 
```

---

## Task 3: Rerun Experiments and Update Metrics/Tables

**Files:**
- Update outputs under `/Users/zhoufuwang/Projects/GRPCA-GD/outputs/`
- Modify: `/Users/zhoufuwang/Projects/GRPCA-GD/results/reporting_template.md`
- Modify: `/Users/zhoufuwang/Projects/GRPCA-GD/latex/manuscript_sample.tex`

- [ ] **Step 1: Rerun chain robustness (seeds 0–4)**

```bash
for s in 0 1 2 3 4; do
  uv run /Users/zhoufuwang/Projects/GRPCA-GD/main.py /Users/zhoufuwang/Projects/GRPCA-GD/configs/robust_chain_seed${s}.yaml
 done
```

- [ ] **Step 2: Rerun SBM robustness (seeds 0–4)**

```bash
for s in 0 1 2 3 4; do
  uv run /Users/zhoufuwang/Projects/GRPCA-GD/main.py /Users/zhoufuwang/Projects/GRPCA-GD/configs/robust_sbm_seed${s}.yaml
 done
```

- [ ] **Step 3: Rerun SBM \(\lambda_2\) sweep**

```bash
for val in 0p00 0p05 0p10 0p20 0p50; do
  uv run /Users/zhoufuwang/Projects/GRPCA-GD/main.py /Users/zhoufuwang/Projects/GRPCA-GD/configs/sbm_lambda2_${val}.yaml
 done
```

- [ ] **Step 4: Regenerate the sweep panel plot**

Run the existing plotting script or regenerate via the same method used previously (ensure output file is a single panel). Save to:

```
/Users/zhoufuwang/Projects/GRPCA-GD/figures/sbm_lambda2_sweep_panel.png
```

- [ ] **Step 5: Update reporting template with A-ManPG row**

Edit `/Users/zhoufuwang/Projects/GRPCA-GD/results/reporting_template.md` to include A-ManPG in any summary tables and update text to list all four methods.

- [ ] **Step 6: Update Results tables with new A-ManPG row and values**

Edit the chain and SBM tables in `/Users/zhoufuwang/Projects/GRPCA-GD/latex/manuscript_sample.tex` to include the A-ManPG row and updated means/stds. Verify that PCA/SparseNoGraph numbers are consistent across graph families or explain any equalities.

- [ ] **Step 7: Commit updated outputs (figures + reporting template + manuscript tables)**

```bash
git add /Users/zhoufuwang/Projects/GRPCA-GD/results/reporting_template.md \
  /Users/zhoufuwang/Projects/GRPCA-GD/latex/manuscript_sample.tex \
  /Users/zhoufuwang/Projects/GRPCA-GD/figures/sbm_lambda2_sweep_panel.png

git commit -m "docs: update results tables and figures with A-ManPG"
```

---

## Task 4: Manuscript Scope and Theory Edits

**Files:**
- Modify: `/Users/zhoufuwang/Projects/GRPCA-GD/latex/manuscript_sample.tex`

- [ ] **Step 1: Update Contributions and Experimental Setup to executed study**

Replace the Contributions bullet list with a narrowed version that matches:
- split-variable objective + alternating solver
- shared-metric synthetic evaluation (chain + SBM, 5 seeds)
- minimal A-ManPG baseline and SparseNoGraph ablation

Replace Experimental Setup with executed protocol:
- chain + SBM graphs only
- support regimes used (connected/disconnected as implemented)
- five seeds (0–4)
- methods = PCA, A-ManPG, SparseNoGraph, Proposed
- shared metrics only, explained variance as evaluation metric
- SBM \(\lambda_2\) sweep single-seed
- hyperparameter choice disclosure: \(\lambda_1=0.1, \lambda_2=0.1\) (sweep range), \(\rho=5.0\), \(\eta_A=0.05\)

- [ ] **Step 2: Rename and rewrite “Convergence Analysis”**

Rename section to **Optimization Discussion** and replace content with:
- nonconvexity statement (Stiefel + coupling)
- convexity of B-subproblem for fixed A
- explicit A/B update rules used
- \(\eta_A\) fixed experimental value and \(\eta_B\) formula
- explicit stopping criteria (relative objective change, coupling gap, orthogonality error, max iters)
- closing sentence: no formal convergence proof; out of scope

- [ ] **Step 3: Notation fixes at first use**

Add definitions at first use (objective, algorithm update):
- entrywise \(\|\cdot\|_1\)
- soft-thresholding \(\mathrm{soft}(\cdot,\tau)\)
- \(\mathrm{sym}(M)=(M+M^\top)/2\)
- orthonormalization used for shared explained variance (QR)

- [ ] **Step 4: Remove placeholder appendices**

Delete appendix sections or replace with real content (if any). No placeholder language remains.

- [ ] **Step 5: Update figure path**

Replace `../outputs/sbm_lambda2_sweep/lambda2_panel.png` with `figures/sbm_lambda2_sweep_panel.png`.

- [ ] **Step 6: Commit manuscript edits**

```bash
git add /Users/zhoufuwang/Projects/GRPCA-GD/latex/manuscript_sample.tex

git commit -m "docs: align scope and optimization discussion"
```

---

## Task 5: Final Verification

**Files:**
- Manuscript PDF output

- [ ] **Step 1: Build the LaTeX manuscript**

```bash
cd /Users/zhoufuwang/Projects/GRPCA-GD/latex
pdflatex -interaction=nonstopmode manuscript_sample.tex
bibtex manuscript_sample || true
pdflatex -interaction=nonstopmode manuscript_sample.tex
pdflatex -interaction=nonstopmode manuscript_sample.tex
```

Expected: PDF builds without missing-figure errors; no placeholder appendix references.

- [ ] **Step 2: Spot-check claims**

Confirm:
- Abstract and Introduction match executed evidence.
- Results overview lists four methods and shared metrics only.
- A-ManPG baseline described as external minimal comparator.
- No raw objective comparisons remain.

- [ ] **Step 3: Commit any final fixes**

```bash
git add /Users/zhoufuwang/Projects/GRPCA-GD/latex/manuscript_sample.tex

git commit -m "docs: finalize claim-audit alignment"
```

---

## Assumptions

- A dedicated worktree was not created for this plan; work proceeds in the current repo.
- Minimal A-ManPG uses ambient gradient + soft-threshold + QR retraction; this is explicitly labeled as a minimal comparator.
- The plotting step for the SBM sweep reuses existing logic; output is saved under `figures/`.
