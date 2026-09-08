"""Builders shared by the Data Catalog domain tests."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.corpus import Corpus
from emblema.catalog.domain.corpus_content import CorpusContent
from emblema.catalog.domain.corpus_source import CorpusSource
from emblema.catalog.domain.identifiers import CorpusId
from emblema.catalog.domain.licence import Licence
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.timestamps import UtcDateTime

SCHEMA = ChannelSchema(frozenset({Channel("temperature", "K"), Channel("pressure", "Pa")}))
OTHER_SCHEMA = ChannelSchema(frozenset({Channel("vibration", "m/s2")}))
LICENCE = Licence("CC-BY-4.0", permits_derivatives=True)
SOURCE = CorpusSource("NASA PCoE", "https://example.org/cmapss")
EPOCH = datetime(2026, 9, 8, 12, tzinfo=UTC)
AT = UtcDateTime(EPOCH)


def instant(seconds: int) -> UtcDateTime:
    return UtcDateTime(EPOCH + timedelta(seconds=seconds))


def content(data: bytes = b"records", record_count: int = 10) -> CorpusContent:
    return CorpusContent(Checksum.of_bytes(data), record_count)


def corpus_id(number: int = 1) -> CorpusId:
    return CorpusId(UUID(int=number))


def version_id(number: int = 1) -> CorpusVersionId:
    return CorpusVersionId(UUID(int=100 + number))


def empty_corpus() -> Corpus:
    return Corpus(corpus_id(), "C-MAPSS", SOURCE)
