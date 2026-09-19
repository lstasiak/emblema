from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.pretraining_curve_figures import draw, main
from scripts.pretraining_curve_report import write
from tests.scripts.test_pretraining_curve_report import two_corpora


def test_the_curve_is_drawn_from_the_stored_rows(tmp_path: Path) -> None:
    drawn = draw(two_corpora(), tmp_path / "figures" / "curve.png")

    assert drawn.is_file()
    assert drawn.stat().st_size > 0


def test_the_figure_lands_beside_the_curve_unless_asked_for_elsewhere(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(two_corpora(), tmp_path / "curve")

    main([str(tmp_path / "curve")])
    main([str(tmp_path / "curve"), "--figure", str(tmp_path / "note" / "named.png")])

    assert (tmp_path / "curve" / "pretraining-curve-test-experiment.png").is_file()
    assert (tmp_path / "note" / "named.png").is_file()
    assert capsys.readouterr().out.count(".png") == 2
