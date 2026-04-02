from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "phase2" / "compatibility"

LAMBDA2_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SEEDS = [0, 1, 2]


def _dump_config(path: Path, cfg: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _base_config() -> Dict[str, object]:
    return {
        "n": 200,
        "r": 1,
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
        "block": "compatibility",
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
                "graph_family",
                "support_type",
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
                    "seed": cfg.get("seed"),
                    "lambda2": cfg.get("lambda2"),
                }
            )


def generate_compatibility_configs() -> None:
    rows: List[Dict[str, object]] = []
    graph_specs = [
        ("chain", 200, None),
        ("grid", 196, {"grid_rows": 14, "grid_cols": 14}),
        ("sbm", 200, {"sbm_blocks": 3, "sbm_p_in": 0.2, "sbm_p_out": 0.02}),
    ]
    support_types = {
        "chain": ["connected", "multi_cluster", "fragmented"],
        "grid": ["connected", "multi_cluster", "fragmented"],
        "sbm": ["connected", "multi_cluster", "fragmented", "cross_community"],
    }

    for seed in SEEDS:
        for graph_family, p, extra in graph_specs:
            for support_type in support_types[graph_family]:
                for lambda2 in LAMBDA2_GRID:
                    cfg = _base_config()
                    cfg.update(
                        {
                            "seed": seed,
                            "p": p,
                            "lambda2": float(lambda2),
                            "graph_family": graph_family,
                            "support_type": support_type,
                        }
                    )
                    if extra:
                        cfg.update(extra)
                    tag = _lambda2_tag(lambda2)
                    config_stem = (
                        f"compat_{graph_family}_{support_type}_"
                        f"lambda2_{tag}_seed{seed}"
                    )
                    cfg["output_dir"] = (
                        f"outputs/phase2/compatibility/{config_stem}/seed_{seed}"
                    )
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
    generate_compatibility_configs()


if __name__ == "__main__":
    main()
