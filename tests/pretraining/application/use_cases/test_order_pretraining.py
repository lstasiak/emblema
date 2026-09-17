from dataclasses import replace

import pytest

from emblema.pretraining.domain.exceptions import PretrainingOrderRejectedError
from emblema.pretraining.domain.handoff.pretraining_order import PretrainingOrder
from emblema.pretraining.domain.training.run_signature import RunSignature
from tests.support.experiments import corpus
from tests.support.handoff import COMMIT, CONFIGURATION, CORPUS, MANIFEST, ORDERED_AT
from tests.support.handoff import pretraining_input as described
from tests.support.handoff_process import InMemoryHandoff, order_command


def test_ordering_registers_the_backbone_and_places_its_order() -> None:
    machines = InMemoryHandoff()

    placed = machines.order()(order_command())

    ordered = machines.backbones.get(placed.backbone)
    assert not ordered.is_ready
    assert (ordered.run, ordered.git_commit, ordered.ordered_at) == ("first", COMMIT, ORDERED_AT)
    assert ordered.input == described()
    assert machines.exchange.read_order(placed.order) == PretrainingOrder.of(ordered)


def test_the_order_signs_the_corpus_as_it_was_read_not_as_it_was_described() -> None:
    machines = InMemoryHandoff()
    # The same manifest described the same way but yielding fewer windows: another run.
    machines.reader.publish(MANIFEST, described(), corpus(training=6))

    placed = machines.order()(order_command())

    signature = machines.backbones.get(placed.backbone).signature
    assert signature == RunSignature.of(CONFIGURATION, corpus(training=6))
    assert signature != RunSignature.of(CONFIGURATION, CORPUS)


def test_the_order_reads_the_share_of_the_corpus_the_configuration_states() -> None:
    machines = InMemoryHandoff()
    half = replace(CONFIGURATION, corpus_fraction=0.5)

    placed = machines.order()(order_command(configuration=half))

    read = machines.reader.read(MANIFEST, half.corpus_share)
    assert len(read.training) == len(CORPUS.training) // 2
    assert machines.backbones.get(placed.backbone).signature == RunSignature.of(half, read)


def test_a_manifest_of_another_corpus_than_the_experiment_names_is_refused() -> None:
    machines = InMemoryHandoff()

    with pytest.raises(PretrainingOrderRejectedError, match="names corpus 'cmapss'"):
        machines.order()(order_command(corpus="cmapss"))


def test_two_orders_of_one_run_are_two_backbones() -> None:
    machines = InMemoryHandoff()

    first = machines.order()(order_command())
    second = machines.order()(order_command())

    assert first.backbone != second.backbone
    assert first.order != second.order
