"""What the handoff runtime refuses, and when: before an epoch is asked for, naming the reason.

That a replayed run reports its epochs as any runtime does is the port's contract; here is what
a result must be for the runtime to replay it at all.
"""

from dataclasses import replace

import pytest

from emblema.pretraining.adapters.handoff.handoff_training_runtime import HandoffTrainingRuntime
from emblema.pretraining.adapters.in_memory.handoff_exchange import InMemoryHandoffExchange
from emblema.pretraining.domain.exceptions import (
    PretrainingResultRejectedError,
    UnreadableHandoffDocumentError,
)
from emblema.pretraining.domain.handoff.pretraining_result import PretrainingResult
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import Retention
from emblema.shared.ports.exceptions import ArtifactNotFoundError
from tests.support.experiments import WEIGHTS, budget, configuration, corpus, mixture
from tests.support.handoff import CHECKPOINT, CONFIGURATION, MIXTURE, order, outcome, result


class Delivered:
    """A store holding the weights, and a result naming them, reported through an exchange."""

    def __init__(self, **overrides: object) -> None:
        self.store = InMemoryArtifactStore()
        self.exchange = InMemoryHandoffExchange()
        kept = self.store.put(b"weights", Retention.DURABLE)
        self.result = result(outcome=outcome_naming(kept), **overrides)
        self.ref = self.exchange.report(self.result)

    def runtime(self) -> HandoffTrainingRuntime:
        return HandoffTrainingRuntime(self.exchange, self.store, self.ref)


def outcome_naming(kept: ArtifactRef) -> TrainingOutcome:
    made = outcome()
    epochs = tuple(
        replace(epoch, backbone=None if epoch.backbone is None else kept) for epoch in made.epochs
    )
    return TrainingOutcome(backbone=kept, epochs=epochs)


def test_the_epochs_reported_are_replayed_as_the_run_that_was_ordered() -> None:
    delivered = Delivered()

    epochs = list(delivered.runtime().train(CONFIGURATION, MIXTURE))

    assert epochs == list(delivered.result.outcome.epochs)


def test_a_result_of_another_configuration_is_refused_before_an_epoch_is_asked_for() -> None:
    delivered = Delivered()

    with pytest.raises(PretrainingResultRejectedError, match="seed=1 for 2"):
        delivered.runtime().train(configuration(budget=budget(seed=2)), MIXTURE)


def test_a_result_over_other_data_is_refused() -> None:
    delivered = Delivered()

    with pytest.raises(PretrainingResultRejectedError, match="the data ordered"):
        delivered.runtime().train(CONFIGURATION, mixture(corpus(seed=7)))


def test_a_result_picked_up_from_another_checkpoint_is_refused() -> None:
    delivered = Delivered()

    with pytest.raises(PretrainingResultRejectedError, match="picked up from"):
        delivered.runtime().train(CONFIGURATION, MIXTURE, CHECKPOINT)


def test_a_result_naming_weights_the_store_does_not_hold_is_refused() -> None:
    exchange = InMemoryHandoffExchange()
    reported: PretrainingResult = result()
    replay = HandoffTrainingRuntime(exchange, InMemoryArtifactStore(), exchange.report(reported))

    with pytest.raises(PretrainingResultRejectedError, match="does not hold"):
        replay.train(CONFIGURATION, MIXTURE)


def test_a_reference_that_is_not_a_result_is_refused() -> None:
    exchange = InMemoryHandoffExchange()
    placed = exchange.place(order())
    replay = HandoffTrainingRuntime(exchange, InMemoryArtifactStore(), placed)

    with pytest.raises(UnreadableHandoffDocumentError):
        replay.train(CONFIGURATION, MIXTURE)


def test_a_result_that_is_not_there_is_reported() -> None:
    replay = HandoffTrainingRuntime(InMemoryHandoffExchange(), InMemoryArtifactStore(), WEIGHTS)

    with pytest.raises(ArtifactNotFoundError):
        replay.train(CONFIGURATION, MIXTURE)
