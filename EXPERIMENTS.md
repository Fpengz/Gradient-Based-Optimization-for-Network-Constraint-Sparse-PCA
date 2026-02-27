# Experiment Protocol

## Baseline set (core)

1. `PCA`
2. `L1-SPCA-ProxGrad`
3. `Graph-PCA` (`lambda1 = 0`)
4. `NetSPCA-PG`
5. `NetSPCA-MASPG-CAR`

## Optional extras

1. `GPower`
2. `ElasticNet-SPCA`
3. `NetSPCA-Stiefel` (when `--include-stiefel-manifold` and `n_components > 1`)

## Synthetic setup

- Data model: rank-1 latent signal with additive Gaussian noise.
- Ground-truth support: sampled as a connected set on the feature graph.
- Graph families: `chain`, `grid`, `er`, `sbm`.
- Primary metrics:
  - explained variance
  - precision / recall / F1
  - top-k precision / recall / F1 (`k = |S*|`)
  - support size
  - connected support LCC ratio
  - Laplacian energy
  - runtime
  - convergence rate

## Graph misspecification protocol

- Use `graph_misspec_rate` to flip a controlled fraction of undirected graph edges before fitting graph-aware methods.
- Keep data generation fixed and only perturb the estimator graph.
- Recommended sweep: `0.0, 0.05, 0.10, 0.20`.

## Fairness choices

- Shared `n_components=1` in baseline comparisons.
- For multi-component runs, compare deflation-based methods against `NetSPCA-Stiefel`.
- Consistent support threshold for metrics.
- Repeated trials with controlled seeds.
- All methods run via the same benchmark harness and logging schema.

## Recommended defaults (audit-backed)

- Start grid:
  - `lambda1`: `0.1, 0.15, 0.2, 0.3`
  - `lambda2`: `0.05, 0.1, 0.2, 0.25`
- Use `NetworkSparsePCA.fit_path(...)` for continuation warm starts when scanning `(lambda1, lambda2)`.
- Prefer:
  - `NetSPCA-PG` for theorem-aligned convergence analysis.
  - `NetSPCA-MASPG-CAR` for practical iteration/runtime reduction.
- Track all of:
  - `objective_curve`
  - `pg_residual_curve`
  - `rel_change_curve`

## Outputs for paper writing

- CSV summaries for tables.
- LaTeX table exports (`summary_table.tex`, `netspca_summary.tex`).
- Sweep figures for trade-off plots.
- Real-data comparison outputs (`colon-comparison-*`, `pitprop-comparison-*`).

## Reproduction commands

```bash
uv run python scripts/run_experiment.py --dataset synthetic --n-repeats 3
uv run python scripts/run_experiment.py --dataset synthetic --graph-misspec-rate 0.15 --n-repeats 3
uv run python scripts/run_experiment.py --dataset synthetic --n-components 3 --include-stiefel-manifold --n-repeats 3
uv run python scripts/run_sweep.py --n-repeats 2
uv run python scripts/run_sweep.py --n-repeats 2 --graph-misspec-rate 0.1
uv run python scripts/run_experiment.py --dataset colon
uv run python scripts/run_experiment.py --dataset pitprop
uv run python scripts/reproduce_figures.py
```
