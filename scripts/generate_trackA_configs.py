from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def _dump_config(path: Path, cfg: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _base_config() -> Dict[str, object]:
    return {
        "n": 200,
        "p": 200,
        "r": 3,
        "support_size": 10,
        "snr": 1.0,
        "lambda1": 0.1,
        "rho": 5.0,
        "max_iters": 200,
        "tol_obj": 1.0e-3,
        "tol_gap": 1.0e-1,
        "tol_orth": 1.0e-6,
        "eta_A": 0.05,
        "baseline": "PCA",
    }


def generate_sweeps(
    seeds: List[int],
    lambda2_grid: List[float],
) -> None:
    for graph_family in ("sbm", "chain"):
        for seed in seeds:
            for lambda2 in lambda2_grid:
                cfg = _base_config()
                cfg.update(
                    {
                        "seed": seed,
                        "lambda2": float(lambda2),
                        "graph_family": graph_family,
                        "support_type": "connected",
                    }
                )
                if graph_family == "sbm":
                    cfg.update(
                        {
                            "sbm_blocks": 3,
                            "sbm_p_in": 0.2,
                            "sbm_p_out": 0.02,
                        }
                    )
                tag = f"{lambda2:.2f}".replace(".", "p")
                name = f"{graph_family}_lambda2_sweep_seed{seed}_{tag}.yaml"
                cfg["output_dir"] = (
                    f"outputs/sweep/{graph_family}/seed{seed}/lambda2_{tag}"
                )
                _dump_config(CONFIG_DIR / name, cfg)


def generate_graph_aligned(
    seeds: List[int],
    lambda2_grid: List[float],
) -> None:
    for seed in seeds:
        for lambda2 in lambda2_grid:
            cfg = _base_config()
            cfg.update(
                {
                    "seed": seed,
                    "lambda2": float(lambda2),
                    "graph_family": "chain",
                    "support_type": "connected_disjoint",
                    "decoy_count": 20,
                    "decoy_variance_factor": 2.0,
                }
            )
            tag = f"{lambda2:.2f}".replace(".", "p")
            name = f"graph_aligned_chain_seed{seed}_{tag}.yaml"
            cfg["output_dir"] = f"outputs/graph_aligned/seed{seed}/lambda2_{tag}"
            _dump_config(CONFIG_DIR / name, cfg)


def main() -> None:
    seeds = [0, 1, 2]
    lambda2_grid = [0.0, 0.05, 0.1, 0.2, 0.5]
    generate_sweeps(seeds, lambda2_grid)
    generate_graph_aligned(seeds, lambda2_grid)


if __name__ == "__main__":
    main()
