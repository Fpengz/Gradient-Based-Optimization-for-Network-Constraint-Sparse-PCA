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

## Synthetic setup

- Data model: rank-1 latent signal with additive Gaussian noise.
- Ground-truth support: sampled as a connected set on the feature graph.
- Graph families: `chain`, `grid`, `er`, `sbm`.
- Primary metrics:
  - explained variance
  - precision / recall / F1
  - support size
  - connected support LCC ratio
  - Laplacian energy
  - runtime

## Fairness choices

- Shared `n_components=1` in baseline comparisons.
- Consistent support threshold for metrics.
- Repeated trials with controlled seeds.
- All methods run via the same benchmark harness and logging schema.

## Outputs for paper writing

- CSV summaries for tables.
- LaTeX table exports (`summary_table.tex`, `netspca_summary.tex`).
- Sweep figures for trade-off plots.
- Real-data comparison outputs (`colon-comparison-*`, `pitprop-comparison-*`).

## Reproduction commands

```bash
uv run python scripts/run_experiment.py --dataset synthetic --n-repeats 3
uv run python scripts/run_sweep.py --n-repeats 2
uv run python scripts/run_experiment.py --dataset colon
uv run python scripts/run_experiment.py --dataset pitprop
uv run python scripts/reproduce_figures.py
```
