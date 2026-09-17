"""What the saturation report derives, stores and resumes, without training a model.

The runtime in memory reports the shape of a run and none of the learning, which is all the
report's own logic needs: that every share spends one budget of steps, that a run is written as
it goes, that a finished run is left alone and an interrupted one picked up from its checkpoint,
and that the curve the runs trace is judged by the domain.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("torch")

from emblema.pretraining.adapters.experiments.experiment_file import ExperimentFile
from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.adapters.in_memory.training_corpus_reader import (
    InMemoryTrainingCorpusReader,
)
from emblema.pretraining.adapters.in_memory.training_runtime import InMemoryTrainingRuntime
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.pretraining.ports.training_runtime import TrainingRuntime
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore, Retention
from scripts.corpus_saturation_report import (
    OUTCOME,
    PlannedRun,
    RecordingTracker,
    curve_of,
    equal_step_budget,
    experiments,
    leg,
    main,
    plan,
    read_epochs,
    read_run,
    render,
    stored_runs,
    train,
    write_settings,
)
from tests.support.experiments import budget, configuration, corpus, epoch_outcome, share
from tests.support.handoff import MANIFEST, outcome
from tests.support.handoff import pretraining_input as described
from tests.support.published import EMPTY_UNIT, TRAINING_UNITS, publish

pytestmark = pytest.mark.ml

DEVICE, REVISION = "cpu", "0123456789abcdef0123456789abcdef01234567"


def experiment(**budget_overrides: object) -> ExperimentFile:
    """An experiment of the measurement over the test corpus, at a budget a test affords."""
    stated: dict[str, object] = {
        "epochs": 4,
        "batch_size": 2,
        "accumulation_steps": 1,
        "learning_rate": 1e-3,
        "warmup_epochs": 1,
        "final_lr_fraction": 0.01,
        "seed": 4,
    }
    return ExperimentFile.model_validate(
        {
            "name": "saturation-test-s",
            "tier": "S",
            "corpus": "test-corpus",
            "corpus_fraction": 1.0,
            "precision": "fp32",
            "dropout": 0.0,
            "decoder_layers": 1,
            "masking": {
                "channel_rate": 0.15,
                "block_rate": 0.6,
                "block_span": 0.5,
                "token_rate": 0.1,
            },
            "budget": stated | budget_overrides,
            "checkpoint": {"every_steps": 2},
        }
    )


def published_reader(tmp_path: Path) -> tuple[InMemoryTrainingCorpusReader, ArtifactRef]:
    published = publish(InMemoryArtifactStore(), tmp_path / "publisher")
    reader = InMemoryTrainingCorpusReader()
    reader.publish(
        published.manifest,
        published.described,
        published.corpus,
        TRAINING_UNITS,
        empty_units=(EMPTY_UNIT,),
    )
    return reader, published.manifest


def in_memory_tracker() -> InMemoryExperimentTracker:
    return InMemoryExperimentTracker()


def test_the_equal_step_budget_spends_the_base_steps_over_another_number_of_batches() -> None:
    base = budget(epochs=2, warmup_epochs=1)

    tenth = equal_step_budget(base, base_batches=100, batches=10)
    third = equal_step_budget(base, base_batches=100, batches=33)

    assert (tenth.epochs, tenth.warmup_epochs) == (20, 10.0)
    assert tenth.epochs * tenth.steps_per_epoch(10) == 200
    # Whole epochs: the nearest, and the warmup keeps its half of the run.
    assert (third.epochs, third.warmup_epochs) == (6, 3.0)


def test_the_plan_reads_each_share_of_the_corpus_and_spends_one_budget(tmp_path: Path) -> None:
    reader, manifest = published_reader(tmp_path)

    planned = plan(experiment(), manifest, reader, fractions=(0.5, 1.0))

    whole, half = planned[1], planned[0]
    assert (whole.fraction, half.fraction) == (1.0, 0.5)
    assert len(whole.windows.training) == 3
    assert len(half.windows.training) == len(reader.read(manifest, share(0.5, seed=4)).training)
    assert half.steps == whole.steps == 8
    assert half.configuration.budget.epochs > whole.configuration.budget.epochs
    assert list(half.windows.validation) == list(whole.windows.validation)


def test_the_epochs_override_is_the_whole_corpus_s_and_every_share_follows(
    tmp_path: Path,
) -> None:
    reader, manifest = published_reader(tmp_path)

    planned = plan(experiment(), manifest, reader, fractions=(0.5, 1.0), epochs=2)

    assert planned[1].configuration.budget.epochs == 2
    assert planned[0].steps == planned[1].steps == 4


def test_the_validation_side_is_thinned_evenly_and_alike_for_every_share() -> None:
    reader = InMemoryTrainingCorpusReader()
    windows = corpus(training=6, validation=5)
    reader.publish(MANIFEST, described(), windows)

    planned = plan(experiment(), MANIFEST, reader, fractions=(0.5, 1.0), validation_stride=2)

    assert list(planned[1].windows.validation) == list(windows.validation)[::2]
    assert list(planned[0].windows.validation) == list(planned[1].windows.validation)
    with pytest.raises(ValueError, match="stride"):
        plan(experiment(), MANIFEST, reader, validation_stride=0)


def test_a_smoke_plan_trains_every_share_for_the_stated_epochs(tmp_path: Path) -> None:
    reader, manifest = published_reader(tmp_path)

    planned = plan(experiment(), manifest, reader, fractions=(0.5, 1.0), equal_steps=False)

    assert [run.configuration.budget.epochs for run in planned] == [4, 4]
    assert planned[0].steps < planned[1].steps


def test_an_experiment_stating_a_share_of_the_corpus_is_refused(tmp_path: Path) -> None:
    reader, manifest = published_reader(tmp_path)
    partial = experiment().model_copy(update={"corpus_fraction": 0.5})

    with pytest.raises(ValueError, match="whole corpus"):
        plan(partial, manifest, reader)


def test_each_epoch_is_written_as_it_is_measured_and_the_outcome_when_the_run_ends(
    tmp_path: Path,
) -> None:
    inner = InMemoryExperimentTracker()
    tracker = RecordingTracker(inner, tmp_path)
    tracker.begin(configuration(), corpus="c", run="0.5")

    tracker.log_epoch(epoch_outcome(0, checkpoint=MANIFEST))
    assert [row.epoch for row in read_epochs(tmp_path)] == [0]
    tracker.log_epoch(epoch_outcome(1))
    assert not (tmp_path / OUTCOME).exists()

    tracker.end(outcome())
    assert (tmp_path / OUTCOME).is_file()
    assert inner.outcome is not None
    write_settings(tmp_path, {"experiment": "x", "corpus_fraction": "1", "planned_steps": "1"})
    stored = read_run(tmp_path)
    assert stored is not None
    assert [row.checkpoint for row in stored.epochs] == [MANIFEST, None]
    assert stored.last_checkpoint == MANIFEST


def test_the_epoch_a_resumed_run_re_enters_is_passed_on_but_not_written_twice(
    tmp_path: Path,
) -> None:
    first = RecordingTracker(InMemoryExperimentTracker(), tmp_path)
    first.begin(configuration(), corpus="c", run="0.5")
    first.log_epoch(epoch_outcome(0))
    first.log_epoch(epoch_outcome(1, checkpoint=MANIFEST))

    inner = InMemoryExperimentTracker()
    resumed = RecordingTracker(inner, tmp_path)
    resumed.begin(configuration(), corpus="c", run="0.5")
    resumed.log_epoch(epoch_outcome(1, training_loss=9.0))
    resumed.log_epoch(epoch_outcome(2))

    rows = read_epochs(tmp_path)
    assert [row.epoch for row in rows] == [0, 1, 2]
    assert rows[1].training_loss != 9.0
    assert [epoch.epoch for epoch in inner.epochs] == [1, 2]


class Dropping:
    """Runtimes whose first session drops after the first epoch, as a platform's does.

    Built over whatever store the report hands in, so the report's checkpoint log is in front
    of it; every session's ``resume_from`` is kept.
    """

    def __init__(self) -> None:
        self.resumed_from: list[ArtifactRef | None] = []

    def __call__(self, store: ArtifactStore) -> TrainingRuntime:
        return _DroppingRuntime(self, InMemoryTrainingRuntime(store))


class _DroppingRuntime:
    def __init__(self, sessions: Dropping, inner: InMemoryTrainingRuntime) -> None:
        self._sessions = sessions
        self._inner = inner

    def train(
        self,
        configuration: ExperimentConfiguration,
        corpus: TrainingCorpus,
        resume_from: ArtifactRef | None = None,
    ) -> Iterator[EpochOutcome]:
        self._sessions.resumed_from.append(resume_from)
        epochs = self._inner.train(configuration, corpus, resume_from)
        return self._dropped(epochs) if len(self._sessions.resumed_from) == 1 else epochs

    @staticmethod
    def _dropped(epochs: Iterator[EpochOutcome]) -> Iterator[EpochOutcome]:
        yield next(epochs)
        raise RuntimeError("session dropped")


class DroppingStore(InMemoryArtifactStore):
    """A store whose process dies on the given transient put, once: a session that drops
    between two checkpoints of one epoch."""

    def __init__(self, dies_on: int) -> None:
        super().__init__()
        self._dies_on = dies_on
        self._transient = 0

    def put(self, content: bytes, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        if retention is Retention.TRANSIENT:
            self._transient += 1
            if self._transient == self._dies_on:
                raise RuntimeError("session dropped")
        return super().put(content, retention)


def planned_run(**overrides: object) -> PlannedRun:
    """A run over five invented windows in batches of two: three steps an epoch, a checkpoint
    every two, so the one an epoch reports falls inside it."""
    windows = corpus(training=5, validation=2)
    stated = configuration(budget=budget(epochs=3, batch_size=2))
    return PlannedRun("saturation-test-s", "invented", stated, windows, 3)


def test_an_interrupted_run_is_picked_up_from_its_last_checkpoint_and_ends_whole(
    tmp_path: Path,
) -> None:
    sessions, store = Dropping(), InMemoryArtifactStore()
    planned = planned_run()
    directory = tmp_path / "runs" / planned.experiment / planned.name

    with pytest.raises(RuntimeError, match="dropped"):
        train(
            planned, directory, sessions, store, in_memory_tracker, device=DEVICE, revision=REVISION
        )
    interrupted = read_run(directory)
    assert interrupted is not None
    assert not interrupted.finished
    assert [row.epoch for row in interrupted.epochs] == [0]

    finished = train(
        planned, directory, sessions, store, in_memory_tracker, device=DEVICE, revision=REVISION
    )

    assert sessions.resumed_from == [None, interrupted.last_checkpoint]
    assert finished.finished
    assert [row.epoch for row in finished.epochs] == [0, 1, 2]


def test_a_session_that_drops_inside_an_epoch_resumes_from_the_last_checkpoint_written(
    tmp_path: Path,
) -> None:
    # Three steps an epoch, a checkpoint every step; the process dies on the third put, before
    # the first epoch reports anything. The log knows two checkpoints the epochs do not.
    sessions, store = Dropping(), DroppingStore(dies_on=3)
    stated = configuration(
        budget=budget(epochs=2, batch_size=2), checkpoint=CheckpointPolicy(every_steps=1)
    )
    planned = PlannedRun("saturation-test-s", "invented", stated, corpus(training=5), 3)
    directory = tmp_path / "runs" / planned.experiment / planned.name

    with pytest.raises(RuntimeError, match="dropped"):
        train(
            planned, directory, sessions, store, in_memory_tracker, device=DEVICE, revision=REVISION
        )
    interrupted = read_run(directory)
    assert interrupted is not None
    assert interrupted.epochs == ()
    resumed_from = interrupted.last_checkpoint
    assert resumed_from is not None
    assert json.loads(store.get(resumed_from))["steps"] == 2

    finished = train(
        planned, directory, sessions, store, in_memory_tracker, device=DEVICE, revision=REVISION
    )

    assert sessions.resumed_from[1:] == [resumed_from]
    assert finished.finished
    assert [row.epoch for row in finished.epochs] == [0, 1]


def test_a_finished_run_is_not_trained_again(tmp_path: Path) -> None:
    sessions, store = Dropping(), InMemoryArtifactStore()
    planned = planned_run()
    directory = tmp_path / "runs" / planned.experiment / planned.name
    with pytest.raises(RuntimeError, match="dropped"):
        train(
            planned, directory, sessions, store, in_memory_tracker, device=DEVICE, revision=REVISION
        )
    train(planned, directory, sessions, store, in_memory_tracker, device=DEVICE, revision=REVISION)
    count = len(sessions.resumed_from)

    again = train(
        planned, directory, sessions, store, in_memory_tracker, device=DEVICE, revision=REVISION
    )

    assert len(sessions.resumed_from) == count
    assert again.finished


def test_the_report_renders_the_curve_and_the_verdict_of_each_experiment(tmp_path: Path) -> None:
    reader, manifest = published_reader(tmp_path)
    planned = plan(experiment(), manifest, reader, fractions=(0.5, 1.0))
    results = tmp_path / "runs"

    stored = leg(
        planned,
        results,
        InMemoryTrainingRuntime,
        InMemoryArtifactStore(),
        in_memory_tracker,
        device=DEVICE,
        revision=REVISION,
    )
    report = render(stored_runs(results), device=DEVICE, revision=REVISION)

    assert [run.fraction for run in stored] == [0.5, 1.0]
    assert all(run.finished for run in stored)
    assert "### saturation-test-s" in report
    assert "| 50% |" in report
    assert "| 100% |" in report
    assert "Best validation (epoch)" in report
    assert any(
        word in report for word in ("data-limited", "saturated", "overfitting", "not learnt")
    )
    curve = curve_of(stored)
    assert [point.fraction for point in curve.points] == [0.5, 1.0]
    best = stored[-1].best_epoch()
    assert best.validation_loss == min(row.validation_loss for row in stored[-1].epochs)
    assert f"at {best.epoch + 1}/{len(stored[-1].epochs)}" in report


def test_the_report_alone_renders_the_experiments_named_and_all_of_them_unless(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reader, manifest = published_reader(tmp_path)
    leg(
        plan(experiment(), manifest, reader, fractions=(0.5, 1.0)),
        tmp_path / "runs",
        InMemoryTrainingRuntime,
        InMemoryArtifactStore(),
        in_memory_tracker,
        device=DEVICE,
        revision=REVISION,
    )

    main(["--report-only", "--workspace", str(tmp_path), "--device", DEVICE])
    every = capsys.readouterr().out
    main(
        [
            "--report-only",
            "--workspace",
            str(tmp_path),
            "--device",
            DEVICE,
            "--experiment",
            "saturation-esa_ad-s",
        ]
    )
    named = capsys.readouterr().out

    assert "### saturation-test-s" in every
    assert "### saturation-test-s" not in named
    assert "corpus saturation" in named


def test_a_single_finished_share_is_reported_without_a_verdict(tmp_path: Path) -> None:
    reader, manifest = published_reader(tmp_path)
    planned = plan(experiment(), manifest, reader, fractions=(1.0,))

    leg(
        planned,
        tmp_path / "runs",
        InMemoryTrainingRuntime,
        InMemoryArtifactStore(),
        in_memory_tracker,
        device=DEVICE,
        revision=REVISION,
    )
    report = render(stored_runs(tmp_path / "runs"), device=DEVICE, revision=REVISION)

    assert "Fewer than two shares finished" in report


def test_smoke_runs_of_unequal_budgets_are_reported_as_no_curve(tmp_path: Path) -> None:
    reader, manifest = published_reader(tmp_path)
    planned = plan(experiment(), manifest, reader, fractions=(0.5, 1.0), equal_steps=False)
    # The half share seeded 4 holds fewer windows than the whole, so it spends fewer steps.
    assert planned[0].steps < planned[1].steps

    leg(
        planned,
        tmp_path / "runs",
        InMemoryTrainingRuntime,
        InMemoryArtifactStore(),
        in_memory_tracker,
        device=DEVICE,
        revision=REVISION,
    )
    report = render(stored_runs(tmp_path / "runs"), device=DEVICE, revision=REVISION)

    assert "These runs trace no curve" in report
    assert "further apart" in report


def test_an_experiment_whose_corpus_was_not_given_is_refused_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(["--experiment", "saturation-esa_ad-s", "--workspace", str(tmp_path / "w")])

    assert "no --corpus names" in capsys.readouterr().err
    assert not (tmp_path / "w").exists()


def test_nothing_to_run_is_refused_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(["--workspace", str(tmp_path / "w")])

    assert "nothing to run" in capsys.readouterr().err


def test_every_experiment_of_the_measurement_in_the_repository_states_the_whole_corpus() -> None:
    stated = experiments()

    assert stated, "the repository states at least one saturation experiment"
    assert all(file.corpus_fraction == 1.0 for file in stated.values())
    assert all(name.startswith("saturation-") for name in stated)
