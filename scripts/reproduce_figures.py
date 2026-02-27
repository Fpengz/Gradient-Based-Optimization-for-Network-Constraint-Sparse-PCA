"""One-command generation of key comparison and sweep artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-repeats", type=int, default=2)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    py = sys.executable
    _run(
        [
            py,
            "scripts/run_experiment.py",
            "--dataset",
            "synthetic",
            "--output-dir",
            args.output_dir,
            "--seed",
            str(args.seed),
            "--max-iter",
            str(args.max_iter),
            "--n-repeats",
            str(args.n_repeats),
        ]
    )
    _run(
        [
            py,
            "scripts/run_sweep.py",
            "--output-dir",
            args.output_dir,
            "--seed",
            str(args.seed),
            "--max-iter",
            str(args.max_iter),
            "--n-repeats",
            str(args.n_repeats),
            "--lambda1-grid",
            "0.01,0.05,0.1,0.2,0.5",
            "--lambda2-grid",
            "0.0,0.01,0.05,0.1,0.5,1.0",
        ]
    )
    _run(
        [
            py,
            "scripts/run_experiment.py",
            "--dataset",
            "colon",
            "--output-dir",
            args.output_dir,
            "--seed",
            str(args.seed),
            "--max-iter",
            str(args.max_iter),
            "--lambda1",
            "0.1",
            "--lambda2",
            "0.1",
        ]
    )


if __name__ == "__main__":
    main()
