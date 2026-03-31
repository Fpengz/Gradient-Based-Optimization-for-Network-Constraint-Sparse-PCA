# GRPCA-GD

Minimal scaffolding for graph-smooth sparse orthogonal PCA with synthetic data and a single baseline.

## Quickstart

Run the smoke test (r=1, rho=5.0, eta_A=0.05):

```bash
uv run python main.py configs/smoke_r1.yaml
```

Run the first small r=3 config (rho=5.0, eta_A=0.05):

```bash
uv run python main.py configs/small_r3.yaml
```

## Outputs
Each run writes to `outputs/<run_name>/`:
- `artifacts.npz` and `artifacts.json`
- `manifest.json`
- `metrics.json` and `metrics.csv`
- `plots/` convergence traces
