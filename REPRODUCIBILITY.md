# Reproducibility Guide

## Environment

```bash
uv sync --dev
```

## Deterministic runs

- Every run accepts `--seed`.
- Repeated trials use `seed + repeat_index`.
- Config is stored alongside results in `config.json`.

## Main paper-style comparison

```bash
uv run python scripts/run_experiment.py \
  --n-repeats 3 \
  --lambda1 0.15 \
  --lambda2 0.25 \
  --max-iter 400 \
  --seed 42
```

Artifacts are written to `results/synth-comparison-<timestamp>/`:

- `config.json`
- `records.json` / `records.csv`
- `summary.csv`
- `summary_table.tex`
- `significance.csv` / `significance.json` (paired tests vs `NetSPCA-PG`, when pairings are available)

Recorded metrics include:

- explained variance
- support precision/recall/F1
- top-k precision/recall/F1 (`k = |S*|`)
- LCC ratio
- Laplacian energy
- runtime / convergence

## Graph misspecification robustness

```bash
uv run python scripts/run_experiment.py \
  --dataset synthetic \
  --graph-misspec-rate 0.15 \
  --n-repeats 3 \
  --seed 42
```

This perturbs the estimator graph (not the data-generating covariance) and logs `graph_misspec_rate` in outputs.

## Multi-component manifold benchmark

```bash
uv run python scripts/run_experiment.py \
  --dataset synthetic \
  --n-components 3 \
  --include-stiefel-manifold \
  --n-repeats 3 \
  --seed 42
```

## Real dataset runs

```bash
uv run python scripts/run_experiment.py --dataset colon --seed 42
uv run python scripts/run_experiment.py --dataset pitprop --seed 42
```

Artifacts are written to:

- `results/colon-comparison-<timestamp>/`
- `results/pitprop-comparison-<timestamp>/`

Each folder contains `config.json`, `records.json/csv`, and `summary.csv`.

## Hyperparameter sweeps

```bash
uv run python scripts/run_sweep.py \
  --lambda1-grid 0.01,0.05,0.1,0.2,0.5 \
  --lambda2-grid 0.0,0.01,0.05,0.1,0.5,1.0 \
  --n-repeats 2 \
  --seed 42
```

Artifacts are written to `results/synth-sweep-<timestamp>/`:

- `records.csv`
- `netspca_records.csv`
- `netspca_summary.csv`
- `netspca_summary.tex`
- `variance_vs_sparsity.png`
- `connectivity_vs_lambda2.png`

Optional robustness sweep:

```bash
uv run python scripts/run_sweep.py --graph-misspec-rate 0.1 --n-repeats 2 --seed 42
```

## One-command bundle

```bash
uv run python scripts/reproduce_figures.py
```

This executes synthetic comparison, synthetic sweep, and colon comparison in sequence.

## Backend comparison (implementation backend parity)

```bash
uv run python scripts/run_backend_comparison.py --n-repeats 3 --seed 42
```

Artifacts are written to `results/backend-comparison-<timestamp>/`:

- `config.json`
- `records.csv`
- `summary.csv`
- `significance.csv` / `significance.json` (backend vs NumPy paired tests)

## Dynamic graph robustness

```bash
uv run python scripts/run_dynamic_graph_experiment.py --n-steps 5 --seed 42
```

Artifacts are written to `results/dynamic-graph-<timestamp>/`:

- `config.json`
- `records.csv`
- `summary.csv`

## Pinned paper artifact manifests

```bash
uv run python scripts/reproduce_paper_artifacts.py
```

This executes the default manifest set in `benchmarks/manifests/`:

- `paper_core.json`
- `paper_misspec.json`
- `paper_large_scale.json`

## Large-scale stress (`p >= 2000`)

```bash
uv run python scripts/run_large_scale_stress.py \
  --n-features-grid 2000,3000 \
  --n-repeats 1 \
  --max-iter 200 \
  --seed 42
```

Outputs include stationarity diagnostics in `summary.csv`:

- `pg_residual_last_mean`
- `pg_residual_ratio_mean`
- `objective_monotone_rate`

## Validation checks

```bash
uv run ty check src/experiments src/models src/utils scripts tests
uv run pytest -q
uv run ruff check src/experiments src/models src/utils scripts tests
uv run black --check src/experiments src/models src/utils scripts tests
```
