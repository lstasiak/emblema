from dataclasses import replace

import pytest

from emblema.pretraining.domain.exceptions import (
    BackboneNotFoundError,
    IncompatibleStartError,
    InvalidTrainingMixtureError,
    PretrainingOrderRejectedError,
)
from emblema.pretraining.domain.handoff.pretraining_order import PretrainingOrder
from emblema.pretraining.domain.training.corpus_fraction import CorpusFraction
from emblema.pretraining.domain.training.corpus_passes import CorpusPasses
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.trained_parameters import TrainedParameters
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.support.experiments import WEIGHTS, configuration, continued, corpus, mixture
from tests.support.handoff import (
    COMMIT,
    CONFIGURATION,
    CORPUS,
    DELIVERED_AT,
    MANIFEST,
    ORDERED_AT,
    RESULT_REF,
    backbone,
    backbone_id,
)
from tests.support.handoff import pretraining_input as described
from tests.support.handoff_process import InMemoryHandoff, order_command


def test_ordering_registers_the_backbone_and_places_its_order() -> None:
    machines = InMemoryHandoff()

    placed = machines.order()(order_command())

    ordered = machines.backbones.get(placed.backbone)
    assert not ordered.is_ready
    assert (ordered.run, ordered.git_commit, ordered.ordered_at) == ("first", COMMIT, ORDERED_AT)
    assert ordered.inputs == (described(),)
    assert machines.exchange.read_order(placed.order) == PretrainingOrder.of(ordered)


def test_the_order_signs_the_corpus_as_it_was_read_not_as_it_was_described() -> None:
    machines = InMemoryHandoff()
    # The same manifest described the same way but yielding fewer windows: another run.
    machines.reader.publish(MANIFEST, described(), corpus(training=6))

    placed = machines.order()(order_command())

    signature = machines.backbones.get(placed.backbone).signature
    assert signature == RunSignature.of(CONFIGURATION, mixture(corpus(training=6)))
    assert signature != RunSignature.of(CONFIGURATION, mixture(CORPUS))


def test_the_order_reads_the_share_of_the_corpus_the_configuration_states() -> None:
    machines = InMemoryHandoff()
    half = replace(CONFIGURATION, corpus_fraction=0.5)

    placed = machines.order()(order_command(configuration=half))

    read = machines.reader.read(MANIFEST, half.corpus_share)
    assert len(read.training) == len(CORPUS.training) // 2
    assert machines.backbones.get(placed.backbone).signature == RunSignature.of(half, mixture(read))


def test_a_manifest_of_another_corpus_than_the_experiment_names_is_refused() -> None:
    machines = InMemoryHandoff()

    with pytest.raises(PretrainingOrderRejectedError, match="names corpus 'cmapss'"):
        machines.order()(order_command(corpora=(("cmapss", MANIFEST),)))


def test_two_orders_of_one_run_are_two_backbones() -> None:
    machines = InMemoryHandoff()

    first = machines.order()(order_command())
    second = machines.order()(order_command())

    assert first.backbone != second.backbone
    assert first.order != second.order


SECOND_MANIFEST = ArtifactRef("durable/second", Checksum.of_bytes(b"second manifest"))


def test_an_order_over_several_corpora_reads_them_in_the_order_named() -> None:
    machines = InMemoryHandoff()
    second = continued(name="second", seed=2)
    machines.reader.publish(
        SECOND_MANIFEST,
        described(corpus="second", manifest=SECOND_MANIFEST, vocabulary_size=5),
        second,
    )

    placed = machines.order()(
        order_command(corpora=((CORPUS.name, MANIFEST), ("second", SECOND_MANIFEST)))
    )

    ordered = machines.backbones.get(placed.backbone)
    assert [read.corpus for read in ordered.inputs] == ["invented", "second"]
    assert ordered.vocabulary_size == 5
    assert ordered.signature == RunSignature.of(CONFIGURATION, mixture(CORPUS, second))
    assert machines.exchange.read_order(placed.order).manifests == (MANIFEST, SECOND_MANIFEST)


def test_corpora_whose_vocabularies_were_not_chained_are_refused() -> None:
    machines = InMemoryHandoff()
    apart = corpus(name="apart", seed=5, channels=("apart/x", "apart/y", "apart/z"))
    machines.reader.publish(
        SECOND_MANIFEST, described(corpus="apart", manifest=SECOND_MANIFEST), apart
    )

    with pytest.raises(InvalidTrainingMixtureError, match="does not continue"):
        machines.order()(
            order_command(corpora=((CORPUS.name, MANIFEST), ("apart", SECOND_MANIFEST)))
        )


def test_a_corpus_with_a_fraction_of_its_own_is_read_at_it_and_the_rest_at_the_mixture_s() -> None:
    machines = InMemoryHandoff()
    second = ArtifactRef("durable/second", Checksum.of_bytes(b"second"))
    other = continued(name="other", seed=5)
    machines.reader.publish(second, described(corpus="other", manifest=second), other)
    mixed = replace(
        CONFIGURATION,
        corpus_fraction=0.5,
        fractions=(CorpusFraction(corpus="other", fraction=0.25),),
    )

    placed = machines.order()(
        order_command(configuration=mixed, corpora=((CORPUS.name, MANIFEST), ("other", second)))
    )

    half = machines.reader.read(MANIFEST, mixed.corpus_share)
    quarter = machines.reader.read(second, mixed.corpus_share_of("other"))
    assert len(half.training) == len(CORPUS.training) // 2
    assert len(quarter.training) == len(other.training) // 4
    assert machines.backbones.get(placed.backbone).signature == RunSignature.of(
        mixed, mixture(half, quarter)
    )


def test_a_fraction_for_a_corpus_the_order_does_not_read_is_refused() -> None:
    machines = InMemoryHandoff()
    mixed = replace(CONFIGURATION, fractions=(CorpusFraction(corpus="meters", fraction=0.1),))

    with pytest.raises(PretrainingOrderRejectedError, match=r"fractions for corpora .*meters"):
        machines.order()(order_command(configuration=mixed))


def test_passes_for_a_corpus_the_order_does_not_read_are_refused() -> None:
    machines = InMemoryHandoff()
    weighted = replace(CONFIGURATION, passes=(CorpusPasses(corpus="stays", passes=4),))

    with pytest.raises(PretrainingOrderRejectedError, match="does not read: \\['stays'\\]"):
        machines.order()(order_command(configuration=weighted))


CONTINUED = configuration(name="continued", start=WEIGHTS, trained=TrainedParameters.CHANNEL_ROWS)


def with_a_ready_parent(**overrides: object) -> InMemoryHandoff:
    # Registered under an identifier the sequential generator will not hand the child.
    machines = InMemoryHandoff()
    parent = backbone(id=backbone_id(99), **overrides)
    machines.backbones.save(parent.deliver(RESULT_REF, WEIGHTS, DELIVERED_AT))
    return machines


def test_an_order_continuing_stored_weights_registers_the_backbone_holding_them_as_parent() -> None:
    machines = with_a_ready_parent()
    second = continued(name="second", seed=2)
    machines.reader.publish(
        SECOND_MANIFEST, described(corpus="second", manifest=SECOND_MANIFEST), second
    )

    placed = machines.order()(
        order_command(configuration=CONTINUED, corpora=(("second", SECOND_MANIFEST),))
    )

    child = machines.backbones.get(placed.backbone)
    assert child.parent == machines.backbones.get_by_artifact(WEIGHTS).id
    assert child.configuration.start == WEIGHTS
    assert machines.exchange.read_order(placed.order).configuration == CONTINUED


def test_an_order_from_its_seed_registers_no_parent() -> None:
    machines = with_a_ready_parent()

    placed = machines.order()(order_command())

    assert machines.backbones.get(placed.backbone).parent is None


def test_an_order_continuing_weights_no_ready_backbone_holds_is_refused() -> None:
    machines = InMemoryHandoff()

    with pytest.raises(BackboneNotFoundError, match="no ready backbone"):
        machines.order()(order_command(configuration=CONTINUED))


def test_an_order_continuing_a_backbone_of_another_shape_is_refused() -> None:
    machines = with_a_ready_parent(configuration=replace(CONFIGURATION, decoder_layers=2))

    with pytest.raises(IncompatibleStartError, match="another decoder"):
        machines.order()(order_command(configuration=CONTINUED))


def test_an_order_over_corpora_off_the_parent_s_chain_is_refused() -> None:
    machines = with_a_ready_parent()
    apart = corpus(name="apart", seed=5, channels=("apart/x", "apart/y", "apart/z"))
    machines.reader.publish(
        SECOND_MANIFEST, described(corpus="apart", manifest=SECOND_MANIFEST), apart
    )

    with pytest.raises(IncompatibleStartError, match="not published on the chain"):
        machines.order()(
            order_command(configuration=CONTINUED, corpora=(("apart", SECOND_MANIFEST),))
        )


def test_an_order_over_a_corpus_from_the_middle_of_the_parent_s_chain_is_accepted() -> None:
    # The parent's vocabulary continues the corpus's own: a run over that corpus alone reads
    # under the shorter vocabulary, and the rows carried over still name its channels.
    machines = InMemoryHandoff()
    longer = continued(name="longer", seed=3)
    longer_described = described(corpus="longer", manifest=SECOND_MANIFEST, vocabulary_size=5)
    machines.reader.publish(SECOND_MANIFEST, longer_described, longer)
    parent = backbone(id=backbone_id(99), inputs=(longer_described,)).deliver(
        RESULT_REF, WEIGHTS, DELIVERED_AT
    )
    machines.backbones.save(parent)

    placed = machines.order()(order_command(configuration=CONTINUED))

    assert machines.backbones.get(placed.backbone).parent == parent.id
