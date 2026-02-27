"""Run pinned benchmark manifests to reproduce paper-ready artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default="benchmarks/manifests",
        help="Directory containing JSON benchmark manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Root directory for generated artifacts.",
    )
    parser.add_argument(
        "--manifests",
        type=str,
        default="paper_core.json,paper_misspec.json,paper_large_scale.json",
        help="Comma-separated manifest file names (relative to --manifest-dir).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable
    manifest_dir = Path(args.manifest_dir)
    manifests = [m.strip() for m in args.manifests.split(",") if m.strip()]
    for name in manifests:
        path = manifest_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        cmd = [py, payload["script"], "--output-dir", args.output_dir, *payload["args"]]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
