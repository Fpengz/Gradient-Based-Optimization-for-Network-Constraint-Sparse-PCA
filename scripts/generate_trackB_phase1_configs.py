from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "trackB"


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
        "tol_obj": 1.0e-6,
        "tol_gap": 1.0e-4,
        "tol_orth": 1.0e-6,
        "eta_A": 0.05,
        "baseline": "PCA",
        "graph_family": "chain",
        "support_type": "connected_disjoint",
        "track": "B",
        "phase": 1,
    }


def generate_phase1(
    seeds: List[int],
    lambda2_grid: List[float],
    decoys: List[Tuple[str, int, float]],
) -> None:
    for seed in seeds:
        for lambda2 in lambda2_grid:
            for level, count, factor in decoys:
                cfg = _base_config()
                cfg.update(
                    {
                        "seed": seed,
                        "lambda2": float(lambda2),
                        "decoy_intensity": level,
                        "decoy_count": int(count),
                        "decoy_variance_factor": float(factor),
                    }
                )
                tag = f"{lambda2:.2f}".replace(".", "p")
                filename = (
                    f"graph_aligned_seed{seed}_lambda2_{tag}_decoy_{level}.yaml"
                )
                cfg["output_dir"] = (
                    f"results/trackB/phase1/seed{seed}/lambda2_{tag}/decoy_{level}"
                )
                _dump_config(CONFIG_DIR / filename, cfg)


def main() -> None:
    seeds = [0, 1, 2]
    lambda2_grid = [0.0, 0.05, 0.1, 0.2, 0.5]
    decoys = [
        ("low", 6, 1.5),
        ("medium", 12, 2.0),
        ("high", 18, 2.5),
    ]
    generate_phase1(seeds, lambda2_grid, decoys)


if __name__ == "__main__":
    main()
