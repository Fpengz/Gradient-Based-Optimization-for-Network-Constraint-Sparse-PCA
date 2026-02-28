"""Lightweight artifact inspection entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a finished NC-SPCA run summary.")
    parser.add_argument("run_dir", type=Path, help="Run directory under outputs/")
    args = parser.parse_args()

    summary_path = args.run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found under {args.run_dir}")
    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
