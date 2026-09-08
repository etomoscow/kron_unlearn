from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "saves/figures/kforge_corrected/corrected_aggregate_used.csv"


def test_matched_figure_export_contains_exact_forget10_pairs(tmp_path: Path) -> None:
    """A missing filter or failed export must not replace the paper figure."""
    stem = "matched"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/kforge_matched_fixed.py"),
            "--input",
            str(INPUT),
            "--out-dir",
            str(tmp_path),
            "--stem",
            stem,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / f"{stem}.pdf").stat().st_size > 0
    assert (tmp_path / f"{stem}.png").stat().st_size > 0

    selected = pd.read_csv(tmp_path / f"{stem}_data.csv")
    assert len(selected) == 12
    assert set(selected["forget"]) == {"forget10"}
    assert set(selected["algo"]) == {"NPO", "SimNPO"}
    assert set(selected["init"]) == {"scratch", "kforge_s045"}
    assert set(selected["steps"]) == {50, 100, 250}
