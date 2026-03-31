import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grpca_gd.runner import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRPCA-GD experiments")
    parser.add_argument("config", help="Path to YAML config")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
