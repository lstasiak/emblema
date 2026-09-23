"""Contract of the TrainingRuntime port, run against every adapter.

What every runtime owes its caller: one outcome per epoch, numbered as the run numbers them; the
weights on the last epoch and nowhere else; a checkpoint whenever the policy asks for one; a run
picked up from a checkpoint that carries on where it stopped; and a refusal for a checkpoint that
belongs to another run or to one that is over. What a run learns is not a contract — only the
runtime that trains has that, and its own tests measure it.

The handoff runtime replays a run made elsewhere, so it is driven through a simulated platform:
the in-memory runtime trains on the far side of an exchange and the handoff runtime reports what
came back, which holds the replay to everything the other two are held to.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from emblema.pretraining.adapters.in_memory.training_runtime import InMemoryTrainingRuntime
from emblema.pretraining.domain.exceptions import IncompatibleCheckpointError
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy
from emblema.pretraining.domain.training.epoch_outcome import EpochOutcome
from emblema.pretraining.domain.training.experiment_configuration import ExperimentConfiguration
from emblema.pretraining.domain.training.training_mixture import TrainingMixture
from emblema.pretraining.ports.training_runtime import TrainingRuntime
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.retention import Retention
from emblema.shared.ports.artifact_store import ArtifactStore
from tests.support.experiments import budget, configuration, continued, corpus, mixture
from tests.support.platform import SimulatedPlatform

torch = pytest.importorskip("torch")

from emblema.pretraining.adapters.training.torch_training_runtime import (  # noqa: E402
    TorchTrainingRuntime,
)

pytestmark = pytest.mark.ml

# Eight training windows in batches of two: four micro-batches an epoch, so a checkpoint every
# third step falls inside an epoch and one every fourth falls on its last step.
CORPUS = corpus(training=8, validation=4)
MIXTURE = mixture(CORPUS)
BATCHES_PER_EPOCH = 4

ADAPTERS: dict[str, Callable[[ArtifactStore], TrainingRuntime]] = {
    "in-memory": InMemoryTrainingRuntime,
    "torch": lambda store: TorchTrainingRuntime(store, device="cpu"),
    "handoff": SimulatedPlatform,
}


class CountingStore:
    """A store that says how much has been written to it, and nothing else of its own."""

    def __init__(self) -> None:
        self._store = InMemoryArtifactStore()
        self.writes = 0

    def put(self, content: bytes, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        self.writes += 1
        return self._store.put(content, retention)

    def put_file(self, source: Path, retention: Retention = Retention.DURABLE) -> ArtifactRef:
        self.writes += 1
        return self._store.put_file(source, retention)

    def get(self, ref: ArtifactRef) -> bytes:
        return self._store.get(ref)

    def get_file(self, ref: ArtifactRef, destination: Path) -> None:
        self._store.get_file(ref, destination)

    def exists(self, ref: ArtifactRef) -> bool:
        return self._store.exists(ref)


@pytest.fixture
def store() -> InMemoryArtifactStore:
    return InMemoryArtifactStore()


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def build(request: pytest.FixtureRequest) -> Callable[[ArtifactStore], TrainingRuntime]:
    factory: Callable[[ArtifactStore], TrainingRuntime] = request.param
    return factory


def stated(every_steps: int = 3, **overrides: object) -> ExperimentConfiguration:
    return configuration(
        **{
            "budget": budget(epochs=2, batch_size=2),
            "checkpoint": CheckpointPolicy(every_steps=every_steps),
            **overrides,
        }
    )


def run(
    runtime: TrainingRuntime,
    configured: ExperimentConfiguration,
    resume_from: ArtifactRef | None = None,
    read: TrainingMixture = MIXTURE,
) -> list[EpochOutcome]:
    return list(runtime.train(configured, read, resume_from))


def test_a_run_reports_one_outcome_per_epoch_in_order(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    outcomes = run(build(store), stated())

    assert [outcome.epoch for outcome in outcomes] == [0, 1]


def test_the_weights_are_reported_on_the_last_epoch_and_nowhere_else(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    outcomes = run(build(store), stated())

    assert outcomes[0].backbone is None
    assert outcomes[-1].backbone is not None
    assert store.exists(outcomes[-1].backbone)


def test_the_backbone_is_the_weights_of_the_epoch_the_run_kept(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    outcomes = run(build(store), stated())

    kept = [outcome.weights for outcome in outcomes if outcome.weights is not None]
    assert kept, "an epoch that is the best so far leaves its weights"
    assert outcomes[-1].backbone in kept
    assert all(store.exists(weights) for weights in kept)


def test_a_run_over_several_corpora_scores_each_of_them_every_epoch(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    mixed = mixture(CORPUS, continued(name="second", seed=2, training=4, validation=2))

    outcomes = run(build(store), stated(), read=mixed)

    assert [outcome.epoch for outcome in outcomes] == [0, 1]
    assert all(
        [scored.corpus for scored in outcome.validation] == ["invented", "second"]
        for outcome in outcomes
    )


def test_a_checkpoint_is_written_when_the_policy_asks_and_is_readable(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    outcomes = run(build(store), stated(every_steps=3))

    assert outcomes[0].checkpoint is not None
    assert store.exists(outcomes[0].checkpoint)


def test_a_policy_no_step_of_an_epoch_meets_leaves_that_epoch_without_one(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    outcomes = run(build(store), stated(every_steps=100))

    assert [outcome.checkpoint for outcome in outcomes] == [None, None]


def test_a_run_picked_up_mid_epoch_finishes_the_epochs_that_were_left(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    configured = stated(every_steps=3)
    interrupted = run(build(store), configured)
    # The first epoch's checkpoint was written three steps into four, so the run stopped inside it.
    resumed = run(build(store), configured, interrupted[0].checkpoint)

    assert [outcome.epoch for outcome in resumed] == [0, 1]
    assert resumed[-1].backbone is not None


def test_a_run_picked_up_on_an_epoch_boundary_starts_at_the_next_epoch(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    configured = stated(every_steps=BATCHES_PER_EPOCH)
    interrupted = run(build(store), configured)

    resumed = run(build(store), configured, interrupted[0].checkpoint)

    assert [outcome.epoch for outcome in resumed] == [1]


def test_a_checkpoint_of_another_run_is_refused(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    interrupted = run(build(store), stated())
    elsewhere = stated(budget=budget(epochs=2, batch_size=2, seed=99))

    with pytest.raises(IncompatibleCheckpointError, match="offered to run"):
        run(build(store), elsewhere, interrupted[0].checkpoint)


def test_a_checkpoint_of_the_same_run_over_another_corpus_is_refused(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    configured = stated()
    interrupted = run(build(store), configured)

    with pytest.raises(IncompatibleCheckpointError):
        run(
            build(store),
            configured,
            interrupted[0].checkpoint,
            mixture(corpus(training=8, name="other")),
        )


def test_a_run_that_has_finished_cannot_be_picked_up(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    configured = stated(every_steps=BATCHES_PER_EPOCH)
    finished = run(build(store), configured)

    with pytest.raises(IncompatibleCheckpointError, match="finished"):
        run(build(store), configured, finished[-1].checkpoint)


def test_nothing_is_written_until_the_first_outcome_is_asked_for(
    build: Callable[[ArtifactStore], TrainingRuntime],
) -> None:
    counting = CountingStore()

    epochs: Iterator[EpochOutcome] = build(counting).train(stated(), MIXTURE)

    assert counting.writes == 0
    next(epochs)
    assert counting.writes > 0


def test_bytes_that_are_not_a_checkpoint_are_refused(
    build: Callable[[ArtifactStore], TrainingRuntime], store: InMemoryArtifactStore
) -> None:
    ref = store.put(b"a checkpoint, honestly", Retention.TRANSIENT)

    with pytest.raises(IncompatibleCheckpointError, match="not a checkpoint"):
        run(build(store), stated(), ref)
