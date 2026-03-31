# Paper-1 Claim-Audit Alignment Design

> **Goal:** Align the manuscript’s front matter, experimental setup, theory section, and figure/appendix hygiene with the executed evidence; add a minimal A-ManPG baseline and update reporting accordingly.

**Architecture:** Track-A empirical-method paper. The manuscript will be tightened to reflect only executed chain/SBM experiments, shared metrics, and the SBM \(\lambda_2\) sweep. Theory is downgraded to an Optimization Discussion without formal proof. A minimal A-ManPG baseline is added as an external sparse orthogonal no-graph comparator.

**Tech Stack:** LaTeX manuscript, Python experiment runner, NumPy/SciPy/Matplotlib, PyYAML, pytest.

---

## 1) Manuscript Alignment (Track A)

**What to change**
- Rewrite **Experimental Setup**, **Contributions**, and any remaining scope language in Abstract/Intro/Results overview to match the executed study only:
  - Graphs: chain + SBM
  - Seeds: 0–4 (five seeds)
  - Methods: PCA, minimal A-ManPG, SparseNoGraph, Proposed
  - Metrics: shared metrics only (support F1, graph smoothness norm on true graph, shared explained variance)
  - SBM \(\lambda_2\) sweep: single-seed
- Remove mention of unrun graph families, parameter grids, or baselines not reported.
- State explicitly that **shared explained variance is an evaluation metric** (not any method’s training objective).
- Define the no-graph comparators distinctly:
  - **SparseNoGraph** = in-family ablation with \(\lambda_2=0\)
  - **Minimal A-ManPG** = external sparse orthogonal no-graph comparator
- Keep the core empirical claim narrow: topology-dependent recovery–smoothness behavior and topology-aware \(\lambda_2\) tuning.
- **Target venue framing:** workshop / technical report / narrow empirical-method paper unless stronger theory and real-data evidence are later added.

---

## 2) Theory Downgrade: “Optimization Discussion”

**What to change**
- Rename **Convergence Analysis** → **Optimization Discussion**.
- Replace theorem-like text with a concrete technical summary:
  - Nonconvexity due to Stiefel constraint + split coupling
  - \(B\)-subproblem convex for fixed \(A\)
  - Updates used in experiments:
    - One retracted Riemannian gradient step for \(A\)
    - One proximal-gradient step for \(B\)
  - State actual step sizes:
    - \(\eta_A\) = **fixed experimental value**; clarify whether this is empirical or formula-based
    - \(\eta_B = 1/(2\lambda_2\|L\|_2+\rho)\)
  - State actual stopping criteria used:
    - relative objective change
    - coupling gap
    - orthogonality tolerance
    - max iterations
- Closing sentence: no formal convergence proof is provided; a formal convergence analysis is outside the scope of this paper.

---

## 3) Baseline Integration & Evaluation Parity

**What to change**
- Add **minimal A-ManPG** as the external sparse orthogonal no-graph baseline.
- Keep **SparseNoGraph** as the in-family ablation with \(\lambda_2=0\).
- Evaluate **PCA, minimal A-ManPG, SparseNoGraph, Proposed** using the same shared metrics only:
  - support F1
  - graph smoothness norm on true graph
  - shared explained variance
- Explicitly state **raw objective values are not compared across methods** because training objectives differ.
- **Fairness controls for minimal A-ManPG:** same initialization family, stopping criteria, iteration budget, and disclosed tuning procedure as Proposed where applicable.

---

## 4) Figures, Tables, Appendix Hygiene

**What to change**
- Move SBM \(\lambda_2\) sweep panel into `figures/` and update LaTeX paths to `figures/...`.
- Remove placeholder appendices entirely unless they contain real content.
- Define missing notation **at first point of use**:
  - entrywise \(\|\cdot\|_1\)
  - soft-thresholding \(\mathrm{soft}(\cdot,\tau)\)
  - \(\mathrm{sym}(M)=(M+M^\top)/2\)
- State \(\eta_A\) and stopping criteria in the manuscript, not just code.
- **Metric definition:** define the orthonormalization procedure used to compute shared explained variance.
- **Hyperparameter disclosure:** explicitly state how \(\lambda_1,\lambda_2,\rho\) were chosen for reported experiments.

---

## Self-Review Checklist

- No mentions of unrun graphs/baselines remain.
- Experimental Setup and Contributions match executed evidence.
- Optimization section is technical but not theorem-implying.
- Figures point to `figures/` paths only.
- No placeholder appendices remain.
- All missing notation defined at first use.
- Baseline definitions are unambiguous (SparseNoGraph vs A-ManPG).
- Shared explained variance definition is explicit.
- Hyperparameter selection is disclosed.
