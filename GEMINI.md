# Gradient-Based Optimization for Network-Constrained Sparse PCA

## Project Overview

This project implements a **Network-Constrained Sparse Principal Component Analysis (NC-SPCA)** algorithm using **Proximal Gradient Descent**.

The goal is to find principal component loading vectors $w$ that maximize explained variance while satisfying two regularizers:
1.  **Sparsity ($\ell_1$):** Few non-zero elements (feature selection).
2.  **Graph Smoothness (Laplacian):** Features connected in a given graph $G$ should have similar weights.

### Mathematical Formulation
The core optimization problem is:

$$
\min_{\|w\|_2 \le 1} \left( - w^\top \hat\Sigma w + \lambda_1 \|w\|_1 + \lambda_2 w^\top L w \right)
$$

Where:
*   $\hat\Sigma$ is the empirical covariance matrix.
*   $L$ is the graph Laplacian.
*   $\lambda_1, \lambda_2$ are hyperparameters controlling sparsity and smoothness.

## Directory Structure

*   `src/`: Source code package.
    *   `models/`: Estimator implementations (`NetworkSparsePCA`).
    *   `optim/`: Optimization algorithms (Proximal Gradient).
    *   `objectives/`: Loss functions and gradients.
    *   `data/`: Synthetic data generation.
*   `doc/`: Theoretical documentation and mathematical derivations.
    *   `network_constrained_spca_guide.md`: Main research guide.
    *   `algorithms/graph_spca_objective.md`: Detailed gradient and proximal step derivations.
*   `experiments/`: Scripts for running comparisons and ablations.
*   `notebooks/`: Jupyter notebooks for exploration and plotting.

## Building and Running

This project uses `uv` for dependency management.

### Prerequisites
*   Python >= 3.10
*   `uv` (Universal Python Package Manager)

### Commands

*   **Install Dependencies:**
    ```bash
    uv sync
    ```

*   **Run Main Script:**
    ```bash
    uv run main.py
    ```

*   **Run Tests:**
    ```bash
    uv run pytest
    ```

*   **Run Experiments:**
    ```bash
    uv run scripts/run_experiment.py
    ```

## Development Roadmap

The implementation follows the guide in `doc/network_constrained_spca_guide.md`.

1.  **Core Solver (`src/optim`):** Implement `proximal_gradient_descent` with:
    *   Gradient of smooth part: $\nabla f(w) = -2\hat\Sigma w + 2\lambda_2 L w$
    *   Proximal operator: Soft-thresholding + Projection onto $\ell_2$ ball.
2.  **Estimator (`src/models`):** Create `NetworkSparsePCA` class inheriting from `sklearn.base.BaseEstimator`.
3.  **Data Generation (`src/data`):** Implement synthetic data generators with known graph structure (spiked covariance).
4.  **Validation:** Reproduce baselines (PCA, SPCA) and verify NC-SPCA recovers structure better than L1-only SPCA.

## Coding Conventions

*   **Style:** Follows standard Python (PEP 8). Tools: `ruff`, `black`.
*   **Typing:** Use type hints for all function signatures.
*   **Documentation:** 
    *   **Style:** Strict adherence to **Numpy-style docstrings**.
    *   **Required Sections:** `Parameters`, `Returns`, `Raises` (if applicable), and `Examples`.
    *   **Content:** clearly explain mathematical operations, referring to the `doc/` folder where necessary. Use LaTeX math formatting within docstrings for clarity (e.g., `.. math::`).
    *   **Completeness:** All public modules, classes, and functions must be documented.
*   **Testing:** Unit tests for solvers and estimators using `pytest`.

## Recommended AI/MCP Workflows

To maximize productivity on this research project, the following agent skills and tools are highly recommended:

1.  **Research Assistant / Technical Writer:**
    *   *Task:* Keeping the `doc/` folder in sync with the code `src/`.
    *   *Skill:* Ability to read LaTeX/Markdown and verify that the implemented code (e.g., gradient steps) matches the derived math.
    *   *Hook:* On changes to `src/optim`, check `doc/algorithms` to ensure the derivation remains accurate.

2.  **Scientific Visualization:**
    *   *Task:* Generating plots for `experiments/`.
    *   *Tool:* `matplotlib` or `seaborn` integration.
    *   *Workflow:* Automatically plotting "Loss vs. Iteration" and "Sparsity Pattern" heatmaps after running experiments to visually verify convergence.

3.  **Codebase Investigator:**
    *   *Task:* tracing complex dependency chains in mathematical operations.
    *   *Tool:* `codebase_investigator` agent.
    *   *Usage:* Use this to map how changes in the `objectives` module propagate to the `optim` solvers.

4.  **Strict Linting & Formatting:**
    *   *Tools:* `ruff` (linter) and `black` (formatter).
    *   *Hook:* Pre-commit hooks to ensure no "lazy" code enters the repo. Research code often rots; strict linting prevents this.
