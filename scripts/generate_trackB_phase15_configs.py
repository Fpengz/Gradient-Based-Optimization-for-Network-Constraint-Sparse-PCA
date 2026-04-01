from pathlib import Path
from typing import Dict, List, Tuple

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "trackB"

LAMBDA2_GRID = [0.00, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.11, 0.12, 0.15, 0.20]


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
        "phase": 1.5,
        "decoy_intensity": "high",
        "decoy_count": 18,
        "decoy_variance_factor": 2.5,
    }


def build_phase15_configs(
    seeds: List[int], lambda2_grid: List[float] | None = None
) -> List[Tuple[Path, Dict[str, object]]]:
    lambda2_grid = lambda2_grid or LAMBDA2_GRID
    configs: List[Tuple[Path, Dict[str, object]]] = []
    for seed in seeds:
        for lambda2 in lambda2_grid:
            cfg = _base_config()
            cfg.update({"seed": seed, "lambda2": float(lambda2)})
            tag = f"{lambda2:.2f}".replace(".", "p")
            filename = f"graph_aligned_phase15_seed{seed}_lambda2_{tag}_decoy_high.yaml"
            cfg["output_dir"] = f"results/trackB/phase1_5/seed{seed}/lambda2_{tag}/decoy_high"
            configs.append((CONFIG_DIR / filename, cfg))
    return configs


def _dump_config(path: Path, cfg: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    configs = build_phase15_configs(seeds=list(range(20)))
    for path, cfg in configs:
        _dump_config(path, cfg)


if __name__ == "__main__":
    main()
