from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "phase2" / "realdata"

LAMBDA2_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SEEDS = [0, 1, 2]


def _dump_config(path: Path, cfg: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _base_config() -> Dict[str, object]:
    return {
        "r": 3,
        "lambda1": 0.1,
        "rho": 5.0,
        "max_iters": 200,
        "tol_obj": 1.0e-6,
        "tol_gap": 1.0e-4,
        "tol_orth": 1.0e-6,
        "eta_A": 0.05,
        "baseline": "PCA",
        "dataset_type": "real",
        "dataset_name": "tcga_brca_string",
        "real_data_dir": "data/real",
        "real_max_genes": 500,
        "string_score_threshold": 700,
        "real_init_noise": 0.001,
        "graph_family": "real",
        "support_type": "connected",
        "support_size": 0,
        "snr": 0.0,
    }


def _lambda2_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _write_manifest(rows: List[Dict[str, object]]) -> None:
    manifest_path = CONFIG_DIR / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config_name",
                "config_path",
                "block",
                "dataset_name",
                "seed",
                "lambda2",
            ],
        )
        writer.writeheader()
        for row in rows:
            cfg = row["config"]
            writer.writerow(
                {
                    "config_name": row["config_name"],
                    "config_path": row["config_path"],
                    "block": cfg.get("block"),
                    "dataset_name": cfg.get("dataset_name"),
                    "seed": cfg.get("seed"),
                    "lambda2": cfg.get("lambda2"),
                }
            )


def generate_realdata_configs() -> None:
    rows: List[Dict[str, object]] = []
    for seed in SEEDS:
        for lambda2 in LAMBDA2_GRID:
            cfg = _base_config()
            cfg.update(
                {
                    "seed": seed,
                    "lambda2": float(lambda2),
                    "block": "realdata",
                }
            )
            tag = _lambda2_tag(lambda2)
            config_stem = f"realdata_tcga_brca_lambda2_{tag}_seed{seed}"
            cfg["output_dir"] = f"outputs/phase2/realdata/{config_stem}/seed_{seed}"
            cfg["config_name"] = f"{config_stem}.yaml"
            config_path = CONFIG_DIR / cfg["config_name"]
            _dump_config(config_path, cfg)
            rows.append(
                {
                    "config_name": cfg["config_name"],
                    "config_path": str(config_path.relative_to(ROOT)),
                    "config": cfg,
                }
            )
    _write_manifest(rows)


def main() -> None:
    generate_realdata_configs()


if __name__ == "__main__":
    main()
