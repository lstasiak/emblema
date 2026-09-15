from collections.abc import Iterator
from typing import Any

import pytest

from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.adapters.in_memory.training_runtime import InMemoryTrainingRuntime
from emblema.pretraining.application.use_cases.pretrain_backbone import (
    PretrainBackbone,
    PretrainBackboneCommand,
)
from emblema.pretraining.domain.exceptions import (
    InvalidTrainingOutcomeError,
    UnsupportedPrecisionError,
)
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from tests.support.experiments import WEIGHTS, budget, configuration, corpus
from tests.support.experiments import epoch_outcome as epoch

CORPUS = corpus(training=8, validation=4)


def command(**overrides: Any) -> PretrainBackboneCommand:
    stated: dict[str, Any] = {
        "configuration": configuration(
            budget=budget(epochs=2, batch_size=2), checkpoint=CheckpointPolicy(every_steps=3)
        ),
        "corpus": CORPUS,
        "run": "first",
    }
    return PretrainBackboneCommand(**(stated | overrides))


class SilentRuntime:
    """A runtime that reports the epochs it is given, so a caller can be held to what it does."""

    def __init__(self, *epochs: EpochOutcome) -> None:
        self._epochs = epochs
        self.resumed_from: ArtifactRef | None = None

    def train(
        self,
        configuration: ExperimentConfiguration,
        corpus: TrainingCorpus,
        resume_from: ArtifactRef | None = None,
    ) -> Iterator[EpochOutcome]:
        self.resumed_from = resume_from
        return iter(self._epochs)


def test_the_run_is_recorded_as_it_happens_and_ended_with_what_it_produced() -> None:
    tracker = InMemoryExperimentTracker()
    pretrain = PretrainBackbone(InMemoryTrainingRuntime(InMemoryArtifactStore()), tracker)

    outcome = pretrain(command())

    assert tracker.run == "first"
    assert tracker.corpus == CORPUS.name
    assert [epoch.epoch for epoch in tracker.epochs] == [0, 1]
    assert tracker.outcome == outcome


def test_the_outcome_carries_the_epochs_and_the_weights_the_run_kept() -> None:
    store = InMemoryArtifactStore()
    pretrain = PretrainBackbone(InMemoryTrainingRuntime(store), InMemoryExperimentTracker())

    outcome = pretrain(command())

    assert len(outcome.epochs) == 2
    assert store.exists(outcome.backbone)


def test_a_run_is_picked_up_from_the_checkpoint_the_command_names() -> None:
    store = InMemoryArtifactStore()
    runtime = InMemoryTrainingRuntime(store)
    interrupted = PretrainBackbone(runtime, InMemoryExperimentTracker())(command())
    checkpoint = interrupted.epochs[0].checkpoint

    resumed = PretrainBackbone(runtime, InMemoryExperimentTracker())(
        command(resume_from=checkpoint)
    )

    assert [epoch.epoch for epoch in resumed.epochs] == [0, 1]


def test_an_epoch_is_recorded_before_the_next_one_is_asked_for() -> None:
    tracker = InMemoryExperimentTracker()
    seen: list[int] = []

    class Watching(SilentRuntime):
        def train(self, configuration, corpus, resume_from=None):
            for reported in self._epochs:
                seen.append(len(tracker.epochs))
                yield reported

    pretrain = PretrainBackbone(Watching(epoch(0), epoch(1, backbone=WEIGHTS)), tracker)
    pretrain(command())

    assert seen == [0, 1]


def test_a_runtime_that_reported_no_epoch_is_refused() -> None:
    pretrain = PretrainBackbone(SilentRuntime(), InMemoryExperimentTracker())

    with pytest.raises(InvalidTrainingOutcomeError, match="without reporting a backbone"):
        pretrain(command())


def test_a_run_that_ended_without_weights_is_refused() -> None:
    pretrain = PretrainBackbone(SilentRuntime(epoch(0)), InMemoryExperimentTracker())

    with pytest.raises(InvalidTrainingOutcomeError, match="without reporting a backbone"):
        pretrain(command())


def test_an_interrupted_run_is_left_without_an_ending() -> None:
    tracker = InMemoryExperimentTracker()

    class Dying(SilentRuntime):
        def train(self, configuration, corpus, resume_from=None):
            yield epoch(0)
            raise RuntimeError("the session dropped")

    with pytest.raises(RuntimeError, match="dropped"):
        PretrainBackbone(Dying(), tracker)(command())

    assert len(tracker.epochs) == 1
    assert tracker.outcome is None


def test_a_run_the_runtime_refuses_is_never_begun() -> None:
    """A tracker holding a run that never started would read as an interrupted one."""

    class Refusing(SilentRuntime):
        def train(self, configuration, corpus, resume_from=None):
            raise UnsupportedPrecisionError("cpu cannot train at fp16")

    tracker = InMemoryExperimentTracker()

    with pytest.raises(UnsupportedPrecisionError):
        PretrainBackbone(Refusing(), tracker)(command())

    assert tracker.configuration is None
    assert tracker.epochs == []
