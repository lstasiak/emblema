"""Builders shared by the tests of the Catalog's published language."""

from uuid import UUID

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum

VERSION_ID = CorpusVersionId(UUID(int=101))
BLOCK = ArtifactRef("durable/sha256/" + "0" * 64, Checksum.of_bytes(b"block"))
PRESSURE = PublishedChannel(
    channel_id=1,
    corpus="alpha",
    channel="pressure",
    unit="Pa",
    statistics=PublishedChannelStatistics(8, 0.5, 0.25),
)
TEMPERATURE = PublishedChannel(channel_id=2, corpus="alpha", channel="temperature", unit="K")


def manifest(
    *,
    channels: tuple[PublishedChannel, ...] = (PRESSURE, TEMPERATURE),
    units: tuple[str, ...] = ("u1", "u2"),
    empty_units: tuple[str, ...] = (),
    training: tuple[str, ...] = ("u1",),
    validation: tuple[str, ...] = ("u2",),
    window_count: int = 3,
    token_count: int = 8,
) -> PublishedCorpusManifest:
    return PublishedCorpusManifest(
        corpus="alpha",
        corpus_version=VERSION_ID,
        corpus_checksum=Checksum.of_bytes(b"raw"),
        block=BLOCK,
        window_length=10.0,
        window_stride=5.0,
        channels=channels,
        units=units,
        empty_units=empty_units,
        training_units=training,
        validation_units=validation,
        split_seed=1,
        window_count=window_count,
        token_count=token_count,
    )
