from __future__ import annotations

from pathlib import Path


LAMBDA2_GRID = [0.0, 0.05, 0.1, 0.2, 0.5]
SEEDS = [0, 1, 2]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "configs" / "grid_sweep"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        for lam in LAMBDA2_GRID:
            tag = str(lam).replace(".", "p")
            cfg = f"""n: 200
p: 196
r: 3
support_size: 10
snr: 1.0
lambda1: 0.1
rho: 5.0
max_iters: 200
tol_obj: 1.0e-06
tol_gap: 0.0001
tol_orth: 1.0e-06
eta_A: 0.05
baseline: PCA
track: B
phase: 2
seed: {seed}
lambda2: {lam}
graph_family: grid
grid_rows: 14
grid_cols: 14
support_type: connected
output_dir: outputs/sweep/grid/seed{seed}/lambda2_{tag}
"""
            path = cfg_dir / f"grid_lambda2_sweep_seed{seed}_{tag}.yaml"
            path.write_text(cfg)


if __name__ == "__main__":
    main()
