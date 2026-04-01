from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "phase2" / "blockB"

LAMBDA2_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SEEDS = [0, 1, 2]


def _dump_config(path: Path, cfg: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _base_config() -> Dict[str, object]:
    return {
        "n": 200,
        "p": 196,
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
    }


def _lambda2_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_blockB_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    graph_family = "grid"
    support_type = "connected"
    grid_rows = 14
    grid_cols = 14
    corruption_types = ["delete", "rewire"]
    corruption_levels = [0.0, 0.2]

    for seed in SEEDS:
        for corruption_type in corruption_types:
            for corruption_level in corruption_levels:
                for lambda2 in LAMBDA2_GRID:
                    cfg = _base_config()
                    cfg.update(
                        {
                            "seed": seed,
                            "lambda2": float(lambda2),
                            "graph_family": graph_family,
                            "support_type": support_type,
                            "corruption_type": corruption_type,
                            "corruption_level": float(corruption_level),
                            "grid_rows": grid_rows,
                            "grid_cols": grid_cols,
                            "block": "blockB",
                        }
                    )
                    tag = _lambda2_tag(lambda2)
                    level_tag = f"{corruption_level:.2f}".replace(".", "p")
                    config_stem = (
                        f"blockB_{graph_family}_{corruption_type}_{level_tag}_"
                        f"lambda2_{tag}_seed{seed}"
                    )
                    prior_state = (
                        "clean" if float(corruption_level) == 0.0 else "corrupted"
                    )
                    cfg["prior_graph_state"] = prior_state
                    cfg["output_dir"] = (
                        f"outputs/phase2/blockB/{config_stem}/seed_{seed}"
                    )
                    config_name = f"{config_stem}.yaml"
                    cfg["config_name"] = config_name
                    rows.append(
                        {
                            "config_name": config_name,
                            "config_path": str(
                                (CONFIG_DIR / config_name).relative_to(ROOT)
                            ),
                            "config": cfg,
                        }
                    )

    return rows


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
                "graph_family",
                "support_type",
                "corruption_type",
                "corruption_level",
                "k",
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
                    "graph_family": cfg.get("graph_family"),
                    "support_type": cfg.get("support_type"),
                    "corruption_type": cfg.get("corruption_type"),
                    "corruption_level": cfg.get("corruption_level"),
                    "k": cfg.get("r"),
                    "seed": cfg.get("seed"),
                    "lambda2": cfg.get("lambda2"),
                }
            )


def generate_blockB_configs() -> None:
    rows = build_blockB_rows()
    for row in rows:
        _dump_config(CONFIG_DIR / row["config_name"], row["config"])
    _write_manifest(rows)


def main() -> None:
    generate_blockB_configs()


if __name__ == "__main__":
    main()
