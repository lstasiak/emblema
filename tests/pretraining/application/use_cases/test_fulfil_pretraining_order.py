import pytest

from emblema.pretraining.application.use_cases.fulfil_pretraining_order import (
    FulfilPretrainingOrderCommand,
)
from emblema.pretraining.domain.exceptions import PretrainingOrderRejectedError
from tests.support.experiments import corpus
from tests.support.handoff import COMMIT, CONFIGURATION, CORPUS, MANIFEST, OTHER_COMMIT
from tests.support.handoff import pretraining_input as described
from tests.support.handoff_process import InMemoryHandoff, order_command


def test_fulfilling_trains_the_order_and_reports_what_the_run_made() -> None:
    machines = InMemoryHandoff()
    placed = machines.order()(order_command())

    reported = machines.fulfil()(
        FulfilPretrainingOrderCommand(order=placed.order, git_commit=COMMIT)
    )

    result = machines.exchange.read_result(reported)
    assert (result.order, result.backbone) == (placed.order, placed.backbone)
    assert (result.configuration, result.corpus) == (CONFIGURATION, CORPUS.shape)
    assert result.git_commit == COMMIT
    assert result.resumed_from is None
    assert machines.store.exists(result.outcome.backbone)
    assert [epoch.epoch for epoch in result.outcome.epochs] == [0, 1]


def test_the_run_is_recorded_by_the_tracker_of_the_machine_that_trains() -> None:
    machines = InMemoryHandoff()
    placed = machines.order()(order_command(run="tracked"))

    machines.fulfil()(FulfilPretrainingOrderCommand(order=placed.order, git_commit=COMMIT))

    (tracker,) = machines.trackers
    assert tracker.run == "tracked"
    assert tracker.outcome is not None


def test_a_run_picked_up_from_a_checkpoint_reports_where_it_was_picked_up() -> None:
    machines = InMemoryHandoff()
    placed = machines.order()(order_command())
    first = machines.exchange.read_result(
        machines.fulfil()(FulfilPretrainingOrderCommand(order=placed.order, git_commit=COMMIT))
    )
    checkpoint = first.outcome.epochs[0].checkpoint
    assert checkpoint is not None

    reported = machines.fulfil()(
        FulfilPretrainingOrderCommand(order=placed.order, git_commit=COMMIT, resume_from=checkpoint)
    )

    assert machines.exchange.read_result(reported).resumed_from == checkpoint


def test_a_machine_running_other_code_than_ordered_stops_before_it_trains() -> None:
    machines = InMemoryHandoff()
    placed = machines.order()(order_command())

    with pytest.raises(PretrainingOrderRejectedError, match=f"runs commit {OTHER_COMMIT}"):
        machines.fulfil()(
            FulfilPretrainingOrderCommand(order=placed.order, git_commit=OTHER_COMMIT)
        )

    assert machines.trackers[0].configuration is None


def test_a_corpus_that_is_not_the_one_ordered_stops_the_run_before_it_trains() -> None:
    machines = InMemoryHandoff()
    placed = machines.order()(order_command())
    # The manifest now yields other windows than it did when the order was signed.
    machines.reader.publish(MANIFEST, described(), corpus(seed=7))

    with pytest.raises(PretrainingOrderRejectedError, match="signs run"):
        machines.fulfil()(FulfilPretrainingOrderCommand(order=placed.order, git_commit=COMMIT))

    assert machines.trackers[0].configuration is None
