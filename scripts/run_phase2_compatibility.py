from __future__ import annotations

import argparse
from pathlib import Path

from topospca.runner import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "phase2" / "compatibility"
MANIFEST = CONFIG_DIR / "manifest.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 compatibility configs.")
    parser.add_argument("--sanity", action="store_true", help="Run only first 6 configs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not MANIFEST.exists():
        raise FileNotFoundError("manifest.csv not found; run config generator first.")
    rows = MANIFEST.read_text(encoding="utf-8").strip().splitlines()[1:]
    if args.sanity:
        rows = rows[:6]
    for line in rows:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        config_path = ROOT / parts[1]
        run(str(config_path))


if __name__ == "__main__":
    main()
