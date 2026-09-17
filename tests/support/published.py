"""A corpus published the way the Catalog publishes one, cut to a test's size, in any store.

Four windows in a block beside a manifest that names them: two training units with windows, a
training unit without, one held out. What a reader of the published corpus must give back is
stated beside it, so a contract holds every adapter to the same corpus.
"""

from pathlib import Path
from typing import Any, NamedTuple
from uuid import UUID

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.adapters.windows.window_block_writer import WindowBlockWriter
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import Token, TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore
from tests.shared.adapters.windows.support import stored_at

CORPUS = "test-corpus"
VERSION = CorpusVersionId(UUID(int=7))
SOURCE_CHECKSUM = Checksum.of_bytes(b"the source data of the version")
TRAINING_UNIT, VALIDATION_UNIT, EMPTY_UNIT, OTHER_TRAINING_UNIT = "u1", "u2", "u3", "u4"
CHANNELS = (
    PublishedChannel(
        channel_id=1,
        corpus=CORPUS,
        channel="temperature",
        unit="K",
        statistics=PublishedChannelStatistics(3, 0.5, 1.0),
    ),
    PublishedChannel(
        channel_id=2,
        corpus=CORPUS,
        channel="pressure",
        unit="Pa",
        statistics=PublishedChannelStatistics(2, 0.25, 0.5),
    ),
    PublishedChannel(channel_id=3, corpus=CORPUS, channel="age", timeless=True),
)
FIRST = TokenWindow.of(
    (
        Token(channel_id=1, value=-0.5, time=0.0, gap=0.0),
        Token(channel_id=2, value=0.25, time=0.5, gap=0.5),
        Token(channel_id=1, value=1.5, time=1.0, gap=1.0),
    )
)
SECOND = TokenWindow.of(
    (
        Token(channel_id=3, value=2.0, time=0.0, gap=0.0, timeless=True),
        Token(channel_id=1, value=0.125, time=0.25, gap=0.25),
    )
)
# Values a float32 cannot hold: what comes back is what went in at the stored width.
HELD_OUT = TokenWindow.of(
    (
        Token(channel_id=1, value=1 / 3, time=1 / 3, gap=1 / 3),
        Token(channel_id=2, value=2 / 3, time=2 / 3, gap=2 / 3),
    )
)
# The other training unit's one window, so that a share of the training units is a share.
OTHER = TokenWindow.of(
    (
        Token(channel_id=2, value=-1.0, time=0.0, gap=0.0),
        Token(channel_id=2, value=1.0, time=0.5, gap=0.5),
    )
)
# The unit each training window was cut from, in the order the block holds them.
TRAINING_UNITS = (TRAINING_UNIT, TRAINING_UNIT, OTHER_TRAINING_UNIT)


class PublishedCorpus(NamedTuple):
    """Where the manifest is, and what a reader must say and give back for it."""

    manifest: ArtifactRef
    block: ArtifactRef
    described: PretrainingInput
    corpus: TrainingCorpus


def publish(store: ArtifactStore, workspace: Path) -> PublishedCorpus:
    """Write the block and the manifest into ``store``, passing through ``workspace``."""
    path = workspace / "corpus.block"
    with WindowBlockWriter(path, scratch=workspace) as writer:
        writer.add(FIRST, unit=0, start=0.0, end=10.0)
        writer.add(SECOND, unit=0, start=5.0, end=15.0)
        writer.add(HELD_OUT, unit=1, start=0.0, end=10.0)
        writer.add(OTHER, unit=2, start=0.0, end=10.0)
    block = store.put_file(path)
    manifest = store.put(PublishedCorpusManifestJson().encode(manifest_of(block)))
    described = PretrainingInput(
        corpus=CORPUS,
        corpus_version=VERSION,
        corpus_checksum=SOURCE_CHECKSUM,
        manifest=manifest,
        block_checksum=block.checksum,
        vocabulary_size=len(CHANNELS),
    )
    corpus = TrainingCorpus(
        name=CORPUS,
        checksum=block.checksum,
        training=[stored_at(FIRST), stored_at(SECOND), stored_at(OTHER)],
        validation=[stored_at(HELD_OUT)],
        vocabulary_size=len(CHANNELS),
    )
    return PublishedCorpus(manifest, block, described, corpus)


def manifest_of(block: ArtifactRef, **overrides: Any) -> PublishedCorpusManifest:
    stated: dict[str, Any] = {
        "corpus": CORPUS,
        "corpus_version": VERSION,
        "corpus_checksum": SOURCE_CHECKSUM,
        "block": block,
        "window_length": 10.0,
        "window_stride": 5.0,
        "channels": CHANNELS,
        "units": (TRAINING_UNIT, VALIDATION_UNIT, OTHER_TRAINING_UNIT),
        "empty_units": (EMPTY_UNIT,),
        "training_units": (TRAINING_UNIT, EMPTY_UNIT, OTHER_TRAINING_UNIT),
        "validation_units": (VALIDATION_UNIT,),
        "split_seed": 1,
        "window_count": 4,
        "token_count": 9,
    }
    return PublishedCorpusManifest(**(stated | overrides))
