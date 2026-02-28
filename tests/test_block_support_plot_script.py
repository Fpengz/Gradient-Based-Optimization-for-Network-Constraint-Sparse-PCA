from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_block_support_plot_script_writes_figure(tmp_path: Path) -> None:
    script = Path("scripts/plot_block_support_patterns.py")
    output_path = tmp_path / "block_support_patterns.png"

    result = subprocess.run(
        [sys.executable, str(script), "--output", str(output_path), "--seed", "42"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.is_file()
