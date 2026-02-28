# NC-SPCA Architecture

## Current architecture

The repository now has a dedicated research framework under `src/nc_spca/`. The design separates mathematics, numerical methods, and benchmark orchestration into different packages:

- `src/nc_spca/objectives/`
  - mathematical objectives and their smooth and nonsmooth pieces
- `src/nc_spca/optimizers/`
  - reusable optimization methods such as PG, MASPG-CAR, and ProxQN
- `src/nc_spca/models/`
  - model orchestration that composes an objective, optimizer, backend, and initializer
- `src/nc_spca/experiments/`
  - experiment runners, aggregation, and artifact generation
- `src/nc_spca/data/`
  - graph builders, synthetic generators, and dataset-facing code
- `src/nc_spca/metrics/`
  - evaluation metrics used by experiments
- `src/nc_spca/tracking/`
  - filesystem-first tracking, optional remote mirroring, and checkpoint IO
- `src/nc_spca/config/`
  - typed configuration schema and loaders
- `src/nc_spca/cli/`
  - Hydra-backed package entrypoints

## Design rules

1. Objective code does not know about experiments.
2. Optimizers do not write files or parse CLI arguments.
3. Experiments do not implement math updates.
4. Tracking owns run directories, metrics, events, and checkpoints.
5. Reproducible runs are driven by committed config files under `conf/`.

## Configuration flow

Hydra is the configuration layer.

- `conf/config.yaml` composes the default stack
- `conf/method/` selects aligned model, objective, and optimizer triples
- `conf/backend/` selects the numerical backend
- `conf/data/` selects the dataset or synthetic graph family
- `conf/objective/` defines the mathematical objective
- `conf/optimizer/` defines the numerical method
- `conf/model/` defines the estimator-level composition
- `conf/experiment/` defines the repetition and suite policy
- `conf/tracking/` defines local and optional WandB mirroring

At runtime, YAML is resolved into typed dataclasses in `src/nc_spca/config/schema.py`.

## Run outputs

All new runs write to `outputs/` and the filesystem is the source of truth.

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
- `artifacts/`

## Migration status

The new framework is implemented for the single-component NC-SPCA stack and synthetic experiment flow.

It now also covers:

- real-data single-model runs for `colon` and `pitprop`
- native multi-component block NC-SPCA through a first-class manifold proximal-gradient model
- baseline wrappers for `PCA`, `L1-SPCA`, `GPower`, and elastic-net SPCA
- Hydra multirun comparisons via `method=` presets

Legacy code still exists under the older `src/models/`, `src/experiments/`, `scripts/`, and `doc/` layout. That code remains usable during migration, but new architecture work should target `src/nc_spca/`, `conf/`, and `outputs/`.
