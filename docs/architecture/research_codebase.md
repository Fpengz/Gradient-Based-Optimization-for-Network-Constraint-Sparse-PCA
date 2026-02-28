# Research Codebase Layout

## Execution code

- `src/nc_spca/`
  - typed configuration
  - objective definitions
  - optimizers
  - model composition
  - experiment runners
  - tracking and checkpointing

## Config

- `conf/`
  - method presets
  - backend
  - data
  - objective
  - optimizer
  - model
  - experiment
  - tracking

The `conf/method/` group is the main comparison surface. It binds model, objective, and optimizer selections into one override such as `method=pca` or `method=nc_spca_prox_qn`.

## Generated artifacts

- `outputs/`
  - run directories
  - metrics
  - checkpoints
  - artifacts

## Legacy migration boundary

The legacy benchmark stack still lives under:

- `src/models/`
- `src/experiments/`
- `scripts/`
- `doc/`

Do not add new architecture code there. New work should target `src/nc_spca/`, `conf/`, and `outputs/`.

Current first-class new-package support includes:

- synthetic single-model runs
- real-data runs on `colon` and `pitprop`
- baseline wrappers for `PCA`, `L1-SPCA`, `GPower`, and elastic-net SPCA
- native block NC-SPCA through a Stiefel-manifold proximal-gradient path
