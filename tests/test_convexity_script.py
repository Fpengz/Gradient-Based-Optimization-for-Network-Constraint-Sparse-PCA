from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_convexity_script_writes_expected_figures(tmp_path: Path) -> None:
    script = Path("scripts/convexity_and_objectives.py")

    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "quadratic_forms.png").is_file()
    assert (tmp_path / "spca_objective_contours.png").is_file()
    assert (tmp_path / "spca_objective_surfaces.png").is_file()
