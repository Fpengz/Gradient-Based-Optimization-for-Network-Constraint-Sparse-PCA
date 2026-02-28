# Reproducibility Guide

## Environment

```bash
uv sync --dev
```

Optional remote or alternate backends:

```bash
pip install torch geoopt wandb
```

## New architecture runs

The new package uses Hydra config composition and writes all outputs under `outputs/`.

Run the default synthetic experiment:

```bash
uv run nc-spca-run
```

Run a specific optimizer and regularization setting:

```bash
uv run nc-spca-run optimizer=pg objective.lambda1=0.1 objective.lambda2=0.25 experiment.seed=7
```

Run a multirun sweep:

```bash
uv run nc-spca-sweep -m optimizer=pg,prox_qn objective.lambda1=0.05,0.1 objective.lambda2=0.1,0.2
```

Run aligned method comparisons:

```bash
uv run nc-spca-sweep -m method=pca,l1_spca,graph_pca,nc_spca_pg,nc_spca_maspg_car,nc_spca_prox_qn data=pitprop experiment=real_pitprop
```

Run the paper-core config explicitly:

```bash
uv run nc-spca-reproduce experiment=paper_core
```

Inspect a finished run:

```bash
uv run nc-spca-visualize outputs/nc_spca/paper_core/<run_id>
```

## New output layout

Each run directory contains:

- `resolved_config.json`
- `env.json`
- `git_commit.txt` when available
- `metrics.jsonl`
- `events.jsonl`
- `seed_manifest.json`
- `summary.json`
- `checkpoints/latest/`
- `checkpoints/best/`
- `artifacts/records.json`
- `artifacts/records.csv`

Determinism policy:

- experiment repeat `r` uses `seed + r`
- the seed manifest is written per run
- the resolved config is persisted once at run start

## Legacy benchmark stack

The repository still ships the older experiment scripts during migration:

```bash
uv run python scripts/run_experiment.py --n-repeats 3
uv run python scripts/run_sweep.py --n-repeats 2
uv run python scripts/reproduce_paper_artifacts.py
```

Those commands still write to `results/`. New architecture runs should use `outputs/`.

## Supported new-method presets

The `conf/method/` group currently includes:

- `pca`
- `l1_spca`
- `gpower`
- `elastic_net_spca`
- `graph_pca`
- `nc_spca_pg`
- `nc_spca_maspg_car`
- `nc_spca_prox_qn`
- `nc_spca_block`

Real-data presets:

```bash
uv run nc-spca-run method=nc_spca_pg data=colon experiment=real_colon
uv run nc-spca-run method=pca data=pitprop experiment=real_pitprop
```

Block-model preset:

```bash
uv run nc-spca-run method=nc_spca_block data=synthetic_grid data.n_samples=36 data.n_features=16 data.support_size=4 experiment.repeats=1
```

Manuscript support-pattern figure:

```bash
.venv/bin/python scripts/plot_block_support_patterns.py --output doc/latex/figures/block_support_patterns.png
```

## Validation

```bash
uv run ty check src/nc_spca tests/test_architecture_core.py tests/test_experiment_runner.py tests/test_cli_run.py tests/test_checkpointing.py tests/test_config_loader.py
uv run ruff check src/nc_spca tests/test_architecture_core.py tests/test_experiment_runner.py tests/test_cli_run.py tests/test_checkpointing.py tests/test_config_loader.py
uv run pytest -q
```
