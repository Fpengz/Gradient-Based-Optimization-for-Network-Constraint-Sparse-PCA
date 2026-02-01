# Agentic Coding Guidelines

This document provides instructions for AI agents and developers working on the `nc-spca` (Gradient-Based Optimization for Network Constraint Sparse PCA) repository.

## 1. Environment & Build

The project uses `uv` for dependency management and packaging.

### Setup
- **Install dependencies:**
  ```bash
  uv sync
  ```
- **Activate environment:**
  - Windows: `.venv\Scripts\activate`
  - Unix: `source .venv/bin/activate`

### Build
- This is a Python project. No compilation step is required for pure Python code.
- Dependencies are defined in `pyproject.toml`.

## 2. Testing, Linting, and Formatting

Ensure all checks pass before submitting changes.

### Commands
| Task | Command | Description |
|------|---------|-------------|
| **Test** | `pytest` | Run all tests (uses `pytest.ini_options` in `pyproject.toml`) |
| **Lint** | `ruff check .` | Run linting checks |
| **Format** | `black .` | Format code |
| **Type Check** | `mypy .` | Run static type checking (if mypy is configured) |

### Running Specific Tests
- **Single Test File:**
  ```bash
  pytest tests/test_file.py
  ```
- **Single Test Case:**
  ```bash
  pytest tests/test_file.py::test_function_name
  ```
- **Verbose Output:**
  ```bash
  pytest -v
  ```

### Configuration
- **Pytest:** Configured in `pyproject.toml` under `[tool.pytest.ini_options]`. `pythonpath` includes `src`.
- **Ruff/Black:** Configured in `pyproject.toml` or default settings.

## 3. Code Style & Conventions

Adhere strictly to the existing style.

### General
- **Indentation:** 4 spaces (standard Python).
- **Line Length:** Follow Black's default (88 characters).
- **Encoding:** UTF-8.

### Naming
- **Functions/Variables:** `snake_case` (e.g., `calculate_variance`, `user_id`).
- **Classes:** `PascalCase` (e.g., `SparsePCA`, `NetworkConstraint`).
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`, `DEFAULT_TOLERANCE`).
- **Private Members:** Prefix with `_` (e.g., `_helper_function`).

### Imports
- Sort imports using `isort` or `ruff`.
- Group imports:
    1. Standard library
    2. Third-party libraries (e.g., `numpy`, `scikit-learn`)
    3. Local application imports
- Use absolute imports where possible (e.g., `from models.vanilla import VanillaSPCA`).

### Type Hints
- Use Python type hints for function arguments and return values.
- **Example:**
  ```python
  import numpy as np

  def calculate_loss(X: np.ndarray, w: np.ndarray, alpha: float) -> float:
      ...
  ```

### Docstrings
- Use NumPy style docstrings for scientific/data-focused code.
- Document all public modules, classes, and functions.
- **Structure:**
  ```python
  def function_name(arg1, arg2):
      """
      Short summary.

      Extended description if needed.

      Parameters
      ----------
      arg1 : type
          Description of arg1.
      arg2 : type
          Description of arg2.

      Returns
      -------
      type
          Description of return value.
      """
  ```

### Error Handling
- Use specific exceptions (e.g., `ValueError`, `TypeError`) instead of generic `Exception`.
- Fail fast and provide informative error messages.
- **Example:**
  ```python
  if n_components <= 0:
      raise ValueError(f"n_components must be positive, got {n_components}")
  ```

## 4. Project Structure

- `src/`: Core source code.
    - `models/`: Model implementations (e.g., `network_sparse_pca.py`).
    - `objectives/`: Loss functions and optimization objectives.
    - `optim/`: Optimization algorithms.
    - `constraints/`: Constraint implementations.
    - `utils/`: Helper functions.
- `data/`: Data storage and generation scripts.
    - `synthetic/`: Synthetic data generation.
- `experiments/`: Scripts for running experiments and evaluations.
- `notebooks/`: Jupyter notebooks for exploration and visualization.
- `tests/`: Unit and integration tests (create if missing).
- `doc/`: Documentation.

## 5. Development Workflow

1.  **Understand:** Read related code and documentation.
2.  **Branch:** Create a new branch for features or fixes.
3.  **Implement:** Write code in `src/`.
4.  **Test:** Add tests in `tests/` covering new functionality.
5.  **Verify:** Run `pytest`, `ruff check`, and `black` before committing.

## 6. Specific Instructions for Agents

- **File Paths:** Always use absolute paths or resolve relative paths from the project root.
- **No Assumptions:** Verify existence of files and dependencies before using them.
- **Tools:** Use `uv` for package operations.
- **Testing:** If tests are missing, create a `tests/` directory and add a `conftest.py` if needed. Use `pytest` for all testing.
- **Refactoring:** When refactoring, ensuring behavior preservation is paramount. Run tests frequently.
