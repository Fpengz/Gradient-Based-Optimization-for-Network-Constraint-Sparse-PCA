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

## Validation checks

```bash
uv run ty check src/experiments src/models src/utils scripts/run_experiment.py scripts/run_sweep.py scripts/reproduce_figures.py tests
uv run pytest -q
uv run ruff check src/experiments src/models src/utils scripts/run_experiment.py scripts/run_sweep.py scripts/reproduce_figures.py tests
uv run black --check src/experiments src/models src/utils scripts/run_experiment.py scripts/run_sweep.py scripts/reproduce_figures.py tests
```
