"""The figures are drawn from the report's files and from nothing else."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from scripts.excursion_figures import draw, main
from scripts.excursion_report import SIDES, masses_of, measure, write_rows, write_settings
from tests.scripts.test_excursion_report import EXPERIMENT, trained_run
from tests.support.experiments import MIXTURE

pytestmark = pytest.mark.ml


@pytest.fixture
def excursions(tmp_path: Path) -> Path:
    store = InMemoryArtifactStore()
    run, validation = trained_run(tmp_path, store)
    directory = tmp_path / "excursions"
    write_settings(directory, {"threshold": "3", "huber_delta": "1"})
    write_rows(directory / SIDES, masses_of(validation, threshold=3.0))
    measure([run], store, validation, {EXPERIMENT: MIXTURE}, directory, device="cpu")
    return directory


def test_every_figure_of_the_diagnostic_is_drawn_from_the_stored_files(
    excursions: Path, tmp_path: Path
) -> None:
    drawn = draw(excursions, tmp_path / "figures")

    assert sorted(path.name for path in drawn) == [
        "excursion-by-magnitude.png",
        "excursion-by-unit.png",
        "excursion-relative-loss.png",
    ]
    assert all(path.stat().st_size > 0 for path in drawn)


def test_the_figures_land_beside_the_files_unless_asked_for_elsewhere(
    excursions: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main([str(excursions)])
    main([str(excursions), "--figures", str(tmp_path / "note")])
    capsys.readouterr()

    assert len(list((excursions / "figures").iterdir())) == 3
    assert len(list((tmp_path / "note").iterdir())) == 3


def test_nothing_scored_is_refused_with_a_sentence(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no scored backbone"):
        draw(tmp_path, tmp_path / "figures")
