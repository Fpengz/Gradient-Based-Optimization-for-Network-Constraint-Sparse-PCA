# NC-SPCA Architecture

## Core modules

- `src/models/`
  - `PCAEstimator`: dense PCA baseline with unified estimator state.
  - `SparsePCA_L1_ProxGrad`: L1-SPCA baseline.
  - `ZouSparsePCA`: Elastic-Net SPCA style baseline.
  - `GeneralizedPowerMethod`: GPower baseline.
  - `NetworkSparsePCA`: graph + sparsity model (`pg` and `maspg_car` modes).
  - `NetworkSparsePCA_StiefelManifold`: multi-component manifold proximal-gradient solver on the Stiefel set.
  - `TorchNetworkSparsePCA`: optional PyTorch backend for PG/MASPG-style updates.
  - `TorchNetworkSparsePCA_GeooptStiefel`: optional Geoopt Stiefel-manifold backend.
- `src/utils/`
  - `graph.py`: graph constructors and Laplacians.
  - `metrics.py`: explained variance, support F1, top-k support F1, connectivity, smoothness.
- `src/experiments/`
  - `synthetic_benchmark.py`: synthetic data generation, baseline construction, graph-misspecification perturbation, benchmark execution, summary aggregation.
  - `real_benchmark.py`: colon/pitprop loading, feature-graph construction, and real-data method comparison.

## Runner scripts

- `scripts/run_experiment.py`
  - one comparison run with repeated seeds.
  - outputs raw records, summary CSV, and LaTeX table.
- `scripts/run_sweep.py`
  - grid sweep on `lambda1`/`lambda2`.
  - outputs sweep CSV, LaTeX summary, and figure files.
- `scripts/reproduce_figures.py`
  - one-command orchestration of key synthetic + real comparisons.

## Unified estimator API

All estimators expose:

- `fit(X, ...)`
- `components_`
- `history_`
- `converged_`
- `n_iter_`
- `objective_`

`NetworkSparsePCA` additionally exposes:

- `fit_path(...)` for warm-start continuation across `(lambda1, lambda2)` grids.

This keeps the comparison suite fair and interchangeable.
