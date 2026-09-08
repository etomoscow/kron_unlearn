from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest


matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "saves/figures/kforge_corrected/corrected_aggregate_used.csv"
sys.path.insert(0, str(ROOT / "scripts"))

from kforge_matched_fixed import build_matched_figure  # noqa: E402


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


def test_matched_figure_keeps_the_original_panel_style() -> None:
    """Replacing arrows must not replace the established panel presentation."""
    fig = build_matched_figure(pd.read_csv(INPUT))
    try:
        assert tuple(fig.get_size_inches()) == pytest.approx((7.0, 2.75))
        npo, simnpo = fig.axes
        assert npo.get_legend() is not None
        assert npo.get_legend()._loc == 3  # lower left, inside the NPO panel
        assert npo.get_legend().get_frame_on()
        assert not fig.legends
        assert not fig.texts
        for panel in (npo, simnpo):
            assert panel.images[0].get_alpha() == pytest.approx(0.45)
            assert {text.get_text() for text in panel.texts} >= {
                "less desired",
                "more desired",
                "step counts: 50, 100, 250",
            }
            assert panel.title.get_fontsize() == pytest.approx(8.5)
    finally:
        matplotlib.pyplot.close(fig)
