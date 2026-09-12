from dataclasses import replace

import pytest

from emblema.catalog.application.assemblers.published_corpus_manifest_assembler import (
    PublishedCorpusManifestAssembler,
)
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.catalog.domain.archived_corpus import ArchivedCorpus
from emblema.catalog.domain.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.exceptions import InvalidUnitSplitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation_manifest import TokenisationManifest
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.unit_split import UnitSplit
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.catalog.domain.support import CORPUS, OTHER_SCHEMA, SCHEMA, version_id

BLOCK = ArtifactRef("durable/sha256/" + "0" * 64, Checksum.of_bytes(b"block"))
FIRST, SECOND, THIRD = UnitKey("u1"), UnitKey("u2"), UnitKey("u3")
# A vocabulary this corpus continues: one foreign channel first, then this corpus's two.
VOCABULARY = ChannelVocabulary().extended_with("other", OTHER_SCHEMA).extended_with(CORPUS, SCHEMA)
FITTED = ChannelStatistics(8, 0.5, 0.25)
SCHEME = TokenisationScheme.for_vocabulary(VOCABULARY).with_statistics(
    VOCABULARY.id_of(CORPUS, "pressure"), FITTED
)
MANIFEST = TokenisationManifest(
    corpus=CORPUS,
    corpus_version=version_id(),
    corpus_checksum=Checksum.of_bytes(b"raw"),
    archived=ArchivedCorpus(BLOCK, (SECOND, FIRST), 3, 8),
    window=WindowSpec(10.0, 5.0),
    scheme=SCHEME,
    split=UnitSplit(training=frozenset({FIRST, THIRD}), validation=frozenset({SECOND})),
    split_seed=1,
    empty_units=(THIRD,),
)


@pytest.fixture
def assembler() -> PublishedCorpusManifestAssembler:
    return PublishedCorpusManifestAssembler()


def test_the_published_form_restores_to_the_manifest_it_came_from(
    assembler: PublishedCorpusManifestAssembler,
) -> None:
    assert assembler.restore(assembler.assemble(MANIFEST)) == MANIFEST


def test_channels_go_out_in_identifier_order_with_their_statistics_beside(
    assembler: PublishedCorpusManifestAssembler,
) -> None:
    message = assembler.assemble(MANIFEST)

    assert [channel.channel_id for channel in message.channels] == [1, 2, 3]
    assert (message.channels[0].corpus, message.channels[0].statistics) == ("other", None)
    pressure = next(channel for channel in message.channels if channel.channel == "pressure")
    assert pressure.statistics == PublishedChannelStatistics(8, 0.5, 0.25)


def test_units_go_out_as_text_in_block_order_and_the_split_sorted(
    assembler: PublishedCorpusManifestAssembler,
) -> None:
    message = assembler.assemble(MANIFEST)

    assert message.units == ("u2", "u1")
    assert message.empty_units == ("u3",)
    assert (message.training_units, message.validation_units) == (("u1", "u3"), ("u2",))


def test_a_description_the_catalog_rejects_does_not_restore(
    assembler: PublishedCorpusManifestAssembler,
) -> None:
    # Well formed as a message — sorted, disjoint, covering the units — yet a split with no
    # held-out unit is not a split the Catalog would ever have drawn.
    message = replace(
        assembler.assemble(MANIFEST), training_units=("u1", "u2", "u3"), validation_units=()
    )

    with pytest.raises(InvalidUnitSplitError):
        assembler.restore(message)
