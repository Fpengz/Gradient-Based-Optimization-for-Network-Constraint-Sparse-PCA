# Gradient-Based Optimization for Network-Constrained Sparse PCA

This repository contains two layers:

1. the legacy benchmark and paper pipeline that produced the current experiments
2. a new research-grade architecture under `src/nc_spca/` with typed config, compositional objectives and optimizers, and filesystem-first run tracking

## Installation

```bash
uv sync --dev
```

Optional backends:

```bash
pip install torch geoopt wandb
```

## New architecture quick start

The new stack uses Hydra configs in `conf/` and writes reproducible outputs to `outputs/`.

Run the default synthetic NC-SPCA experiment:

```bash
uv run nc-spca-run
```

Override the optimizer and objective from the command line:

```bash
uv run nc-spca-run optimizer=prox_qn objective.lambda1=0.05 objective.lambda2=0.2
```

Use aligned method presets:

```bash
uv run nc-spca-run method=pca data=pitprop experiment=real_pitprop
uv run nc-spca-run method=nc_spca_block data=synthetic_grid data.n_samples=36 data.n_features=16 data.support_size=4 experiment.repeats=1
uv run nc-spca-run method=nc_spca_block data=synthetic_grid data.n_components=2 data.support_overlap_mode=shared objective.sparsity_mode=l21 objective.group_lambda=0.05 experiment=block_synth_core
```

Run a Hydra multirun comparison:

```bash
uv run nc-spca-sweep -m method=pca,l1_spca,graph_pca,nc_spca_pg,nc_spca_maspg_car,nc_spca_prox_qn data=pitprop experiment=real_pitprop
```

Reproduce the default paper-style synthetic configuration:

```bash
uv run nc-spca-reproduce experiment=paper_core
```

Run the native multi-component block experiment:

```bash
uv run nc-spca-run method=nc_spca_block experiment=block_synth_core data=synthetic_grid data.n_components=2 objective.sparsity_mode=l21
```

Generate the manuscript support-pattern figure for the shared-support block example:

```bash
.venv/bin/python scripts/plot_block_support_patterns.py --output doc/latex/figures/block_support_patterns.png
```

Inspect a finished run:

```bash
uv run nc-spca-visualize outputs/nc_spca/paper_core/<run_id>
```

## New package layout

- `conf/method/`
- `src/nc_spca/objectives/`
- `src/nc_spca/optimizers/`
- `src/nc_spca/models/`
- `src/nc_spca/experiments/`
- `src/nc_spca/data/`
- `src/nc_spca/metrics/`
- `src/nc_spca/tracking/`
- `src/nc_spca/config/`
- `src/nc_spca/cli/`
- `conf/`

See `ARCHITECTURE.md` for the package boundaries and run artifact policy.

## Reproducibility contract

Each new run records:

- resolved config
- environment manifest
- git commit when available
- metrics and event streams
- seed manifest
- checkpoints
- tabular artifacts

The filesystem is the source of truth. Optional WandB logging mirrors local state when enabled.

## Legacy benchmark stack

The existing comparison scripts and manuscript pipeline remain available during migration:

- `scripts/run_experiment.py`
- `scripts/run_sweep.py`
- `scripts/reproduce_figures.py`
- `scripts/reproduce_paper_artifacts.py`

That stack still targets the older `src/models/`, `src/experiments/`, and `doc/` layout. New architecture work should go into `src/nc_spca/` and `conf/`.

The new package CLI is now sufficient for:

- baseline comparisons through `method=...`
- synthetic single-model runs
- real-data runs on `colon` and `pitprop`
- native block manifold runs through `method=nc_spca_block`

## Paper review workflow

For manuscript audit, citation cleanup, related work review, and paper-to-code consistency checks, see:

- `SCIENTIFIC_REVIEW_WORKFLOW.md`

## Validation

```bash
uv run ty check src/nc_spca tests/test_architecture_core.py tests/test_experiment_runner.py tests/test_cli_run.py tests/test_checkpointing.py tests/test_config_loader.py
uv run ruff check src/nc_spca tests/test_architecture_core.py tests/test_experiment_runner.py tests/test_cli_run.py tests/test_checkpointing.py tests/test_config_loader.py
uv run pytest -q
```
