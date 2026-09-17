"""The figures are drawn from stored runs and from nothing else."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.adapters.in_memory.training_runtime import InMemoryTrainingRuntime
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from scripts.corpus_saturation_figures import draw, label_of, main
from scripts.corpus_saturation_report import leg, plan, stored_runs
from tests.scripts.test_corpus_saturation_report import (
    DEVICE,
    REVISION,
    experiment,
    published_reader,
)

pytestmark = pytest.mark.ml


@pytest.fixture
def results(tmp_path: Path) -> Path:
    reader, manifest = published_reader(tmp_path)
    planned = plan(experiment(), manifest, reader, fractions=(0.5, 1.0))
    leg(
        planned,
        tmp_path / "runs",
        InMemoryTrainingRuntime,
        InMemoryArtifactStore(),
        InMemoryExperimentTracker,
        device=DEVICE,
        revision=REVISION,
    )
    return tmp_path / "runs"


def test_every_figure_of_the_measurement_is_drawn_from_the_stored_runs(
    results: Path, tmp_path: Path
) -> None:
    drawn = draw(stored_runs(results), tmp_path / "figures")

    assert sorted(path.name for path in drawn) == [
        "corpus-saturation-epochs.png",
        "corpus-saturation-generalisation.png",
        "corpus-saturation-validation.png",
    ]
    assert all(path.stat().st_size > 0 for path in drawn)


def test_the_figures_land_beside_the_runs_unless_asked_for_elsewhere(
    results: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main([str(results)])
    main([str(results), "--figures", str(tmp_path / "note")])
    capsys.readouterr()

    assert len(list((results.parent / "figures").iterdir())) == 3
    assert len(list((tmp_path / "note").iterdir())) == 3


def test_an_experiment_is_labelled_by_its_corpus_and_the_shape_it_ran(results: Path) -> None:
    run = stored_runs(results)[0]

    assert label_of(run) == "test-corpus, tier S"


def test_nothing_finished_is_refused_with_a_sentence(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no finished run"):
        draw([], tmp_path / "figures")
