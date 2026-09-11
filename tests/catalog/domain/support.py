"""Builders shared by the Data Catalog tests."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.corpus import Corpus
from emblema.catalog.domain.corpus_content import CorpusContent
from emblema.catalog.domain.corpus_description import CorpusDescription
from emblema.catalog.domain.corpus_source import CorpusSource
from emblema.catalog.domain.corpus_unit import CorpusUnit, TimeExtent
from emblema.catalog.domain.identifiers import CorpusId, UnitKey
from emblema.catalog.domain.licence import Licence
from emblema.catalog.domain.observation import Observation
from emblema.catalog.domain.static_feature import StaticFeature
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime
from emblema.shared.kernel.timestamps import UtcDateTime

SCHEMA = ChannelSchema(frozenset({Channel("temperature", "K"), Channel("pressure", "Pa")}))
OTHER_SCHEMA = ChannelSchema(frozenset({Channel("vibration", "m/s2")}))
# The same two measured channels plus one static feature of every unit.
STATIC_SCHEMA = ChannelSchema(SCHEMA.channels | {Channel("age", "years", timeless=True)})
CORPUS = "test-corpus"
LICENCE = Licence("CC-BY-4.0", permits_derivatives=True)
SOURCE = CorpusSource("NASA PCoE", "https://example.org/cmapss")
EPOCH = datetime(2026, 9, 8, 12, tzinfo=UTC)
AT = UtcDateTime(EPOCH)


def instant(seconds: int) -> UtcDateTime:
    return UtcDateTime(EPOCH + timedelta(seconds=seconds))


def content(data: bytes = b"records", units: int = 10, observations: int = 200) -> CorpusContent:
    return CorpusContent(Checksum.of_bytes(data), units, observations)


def description(
    data: bytes = b"records",
    schema: ChannelSchema = SCHEMA,
    regime: SamplingRegime = SamplingRegime.REGULAR,
    units: int = 10,
    observations: int = 200,
) -> CorpusDescription:
    return CorpusDescription(schema, regime, content(data, units, observations))


def corpus_id(number: int = 1) -> CorpusId:
    return CorpusId(UUID(int=number))


def version_id(number: int = 1) -> CorpusVersionId:
    return CorpusVersionId(UUID(int=100 + number))


def empty_corpus() -> Corpus:
    return Corpus(corpus_id(), "C-MAPSS", SOURCE)


def unit(
    key: str = "u1", start: float = 0.0, end: float = 10.0, *statics: StaticFeature
) -> CorpusUnit:
    return CorpusUnit(UnitKey(key), TimeExtent(start, end), statics)


def grid(channels: Sequence[str], times: Sequence[float]) -> tuple[Observation, ...]:
    """Every channel observed at every time, with a value that tells the two apart."""
    return tuple(
        Observation(channel, time, 10.0 * time + index)
        for time in times
        for index, channel in enumerate(channels)
    )


def scheme_for(schema: ChannelSchema = SCHEMA, corpus: str = CORPUS) -> TokenisationScheme:
    """An unfitted scheme whose vocabulary holds the channels of ``schema`` under ``corpus``."""
    return TokenisationScheme.for_vocabulary(ChannelVocabulary()).extended_with(corpus, schema)


def identity_scheme(schema: ChannelSchema = SCHEMA, corpus: str = CORPUS) -> TokenisationScheme:
    """A scheme whose normalisation leaves every value as it is: mean 0, spread 1."""
    scheme = scheme_for(schema, corpus)
    for entry in scheme.vocabulary.entries:
        scheme = scheme.with_statistics(entry.channel_id, ChannelStatistics(1, 0.0, 1.0))
    return scheme
