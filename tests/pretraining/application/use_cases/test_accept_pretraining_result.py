"""The whole handoff on in-memory adapters: ordered here, trained there, accepted here."""

from dataclasses import replace
from typing import Any

import pytest

from emblema.pretraining.adapters.handoff.handoff_training_runtime import HandoffTrainingRuntime
from emblema.pretraining.adapters.in_memory.experiment_tracker import InMemoryExperimentTracker
from emblema.pretraining.application.use_cases.accept_pretraining_result import (
    AcceptPretrainingResult,
    AcceptPretrainingResultCommand,
)
from emblema.pretraining.application.use_cases.fulfil_pretraining_order import (
    FulfilPretrainingOrderCommand,
)
from emblema.pretraining.application.use_cases.pretrain_backbone import PretrainBackbone
from emblema.pretraining.domain.exceptions import (
    BackboneAlreadyDeliveredError,
    BackboneNotFoundError,
    PretrainingResultRejectedError,
)
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.retention import Retention
from tests.support.experiments import budget, configuration, corpus
from tests.support.handoff import COMMIT, MANIFEST, ORDERED_AT, OTHER_COMMIT, backbone_id, result
from tests.support.handoff import pretraining_input as described
from tests.support.handoff_process import InMemoryHandoff, order_command


def delivered(machines: InMemoryHandoff) -> ArtifactRef:
    placed = machines.order()(order_command())
    return machines.fulfil()(FulfilPretrainingOrderCommand(order=placed.order, git_commit=COMMIT))


def forged(machines: InMemoryHandoff, **overrides: Any) -> ArtifactRef:
    """A result the machine that trains could not have reported: the delivery, rewritten."""
    honest = machines.exchange.read_result(delivered(machines))
    return machines.exchange.report(replace(honest, **overrides))


def test_accepting_makes_the_backbone_ready_with_the_weights_the_run_delivered() -> None:
    machines = InMemoryHandoff()
    reported = delivered(machines)

    accepted = machines.accept(reported)(AcceptPretrainingResultCommand(result=reported))

    ready = machines.backbones.get(accepted)
    assert ready.is_ready
    assert ready.artifact == machines.exchange.read_result(reported).outcome.backbone
    assert (ready.result, ready.delivered_at) == (reported, ORDERED_AT)


def test_the_replayed_run_is_recorded_here_as_the_run_it_was() -> None:
    machines = InMemoryHandoff()
    reported = delivered(machines)

    machines.accept(reported)(AcceptPretrainingResultCommand(result=reported))

    trained, replayed = machines.trackers
    assert replayed.run == trained.run == "first"
    assert replayed.epochs == trained.epochs
    assert replayed.outcome == trained.outcome


def test_a_result_made_with_other_code_is_refused_before_anything_is_recorded() -> None:
    # The machine that trains stops on other code itself; a result claiming it is one that came
    # some other way, and the acceptance is the second line.
    machines = InMemoryHandoff()
    reported = forged(machines, git_commit=OTHER_COMMIT)

    with pytest.raises(PretrainingResultRejectedError, match="commit"):
        machines.accept(reported)(AcceptPretrainingResultCommand(result=reported))

    assert not machines.backbones.get(backbone_id(1)).is_ready
    assert machines.trackers[-1].configuration is None


def test_a_result_of_another_configuration_is_refused_naming_the_parameter() -> None:
    machines = InMemoryHandoff()
    reported = forged(machines, configuration=configuration(budget=budget(seed=2)))

    with pytest.raises(PretrainingResultRejectedError, match=r"seed=2 for 1"):
        machines.accept(reported)(AcceptPretrainingResultCommand(result=reported))

    assert machines.trackers[-1].configuration is None


def test_a_runtime_replaying_another_result_than_the_one_accepted_is_refused() -> None:
    # The command names one result and the runtime was built on another of the same order: the
    # checks pass for the first, and only the outcome says the weights would be the second's.
    machines = InMemoryHandoff()
    reported = delivered(machines)
    honest = machines.exchange.read_result(reported)
    elsewhere = machines.store.put(b"other weights", Retention.DURABLE)
    epochs = (*honest.outcome.epochs[:-1], replace(honest.outcome.epochs[-1], backbone=elsewhere))
    other = machines.exchange.report(
        replace(honest, outcome=replace(honest.outcome, backbone=elsewhere, epochs=epochs))
    )
    accepting = AcceptPretrainingResult(
        machines.exchange,
        machines.reader,
        machines.backbones,
        PretrainBackbone(
            HandoffTrainingRuntime(machines.exchange, machines.store, other),
            InMemoryExperimentTracker(),
        ),
        machines.clock,
    )

    with pytest.raises(PretrainingResultRejectedError, match="replayed a run delivering"):
        accepting(AcceptPretrainingResultCommand(result=reported))

    assert not machines.backbones.get(backbone_id(1)).is_ready


def test_a_delivery_taken_twice_is_refused_the_second_time() -> None:
    machines = InMemoryHandoff()
    reported = delivered(machines)
    machines.accept(reported)(AcceptPretrainingResultCommand(result=reported))

    with pytest.raises(BackboneAlreadyDeliveredError):
        machines.accept(reported)(AcceptPretrainingResultCommand(result=reported))

    assert machines.trackers[-1].configuration is None


def test_a_result_for_a_backbone_the_registry_does_not_know_is_reported() -> None:
    machines = InMemoryHandoff()
    stranger = machines.exchange.report(result(backbone=backbone_id(42)))

    with pytest.raises(BackboneNotFoundError):
        machines.accept(stranger)(AcceptPretrainingResultCommand(result=stranger))


def test_a_result_over_data_this_process_reads_differently_is_refused() -> None:
    machines = InMemoryHandoff()
    reported = delivered(machines)
    # The manifest yields other windows than the ones the order was signed over.
    machines.reader.publish(MANIFEST, described(), corpus(seed=7))

    with pytest.raises(PretrainingResultRejectedError, match="the data ordered"):
        machines.accept(reported)(AcceptPretrainingResultCommand(result=reported))
