from __future__ import annotations

from pathlib import Path


LAMBDA2_GRID = [0.0, 0.05, 0.1, 0.2, 0.5]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "configs" / "realdata"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    for lam in LAMBDA2_GRID:
        tag = str(lam).replace(".", "p")
        cfg = f"""dataset_type: real
dataset_name: tcga_brca_xena
graph_source: string_ppi
track: B
phase: 2
cache_dir: data/real
max_genes: 2000
string_score_threshold: 700
string_version: "11.5"
seed: 0
r: 3
lambda1: 0.1
lambda2: {lam}
rho: 5.0
max_iters: 200
tol_obj: 1.0e-06
tol_gap: 0.0001
tol_orth: 1.0e-06
eta_A: 0.05
baseline: PCA
output_dir: outputs/real_data/brca/lambda2_{tag}
"""
        path = cfg_dir / f"tcga_brca_lambda2_{tag}.yaml"
        path.write_text(cfg)


if __name__ == "__main__":
    main()
