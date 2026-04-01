from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "phase2" / "blockA" / "manifest.csv"


def _iter_manifest(manifest_path: Path) -> Iterable[dict]:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_manifest(manifest_path: Path) -> None:
    for row in _iter_manifest(manifest_path):
        config_path = Path(row["config_path"])
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        cfg = _load_config(config_path)
        output_dir = Path(cfg["output_dir"])
        if output_dir.exists() and (output_dir / "metrics.json").exists():
            continue
        subprocess.run([sys.executable, str(ROOT / "main.py"), str(config_path)], check=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 Block A configs")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to blockA manifest.csv",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Run sanity slice (currently identical to manifest run)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    run_manifest(Path(args.manifest))


if __name__ == "__main__":
    main()
