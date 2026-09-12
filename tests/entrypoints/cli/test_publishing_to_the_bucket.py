"""Publish a corpus to the configured bucket, then mount it the way a training notebook would.

The second half runs against a workspace the first half never wrote to, standing in for the
other machine, and gets as far as a batch of tensors without opening a source file. Marked
``integration``: it needs the object store from the settings, the local stack by default and the
remote bucket when the process is started with its environment file.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from emblema.catalog.adapters.archive.block_corpus_archive import BlockCorpusArchive
from emblema.catalog.adapters.in_memory.corpus_repository import InMemoryCorpusRepository
from emblema.catalog.application.assemblers.published_corpus_manifest_assembler import (
    PublishedCorpusManifestAssembler,
)
from emblema.catalog.application.use_cases.publish_corpus import PublishCorpusCommand
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.config.settings import Settings
from emblema.entrypoints.cli.composition_root import CompositionRoot
from emblema.entrypoints.cli.known_corpora import KnownCorpora
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import Retention

if TYPE_CHECKING:
    from mypy_boto3_s3.type_defs import ObjectIdentifierTypeDef

pytestmark = pytest.mark.integration

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "cmapss"
CORPUS = "cmapss"
WINDOW = WindowSpec(length=20.0, stride=7.0)
BATCH_SIZE = 4


@pytest.fixture
def store(request: pytest.FixtureRequest) -> Iterator[S3ArtifactStore]:
    """A store under a prefix of this run's own, emptied afterwards through the raw client."""
    config = Settings().artifact_store
    prefix = f"{config.key_prefix}/{Retention.TRANSIENT}/{uuid4().hex}"
    opened = S3ArtifactStore.connect(
        endpoint_url=config.endpoint_url,
        region=config.region,
        access_key=config.access_key.get_secret_value() if config.access_key else None,
        secret_key=config.secret_key.get_secret_value() if config.secret_key else None,
        bucket=config.bucket,
        key_prefix=prefix,
    )
    yield opened

    client = opened._client
    listing = client.list_objects_v2(Bucket=config.bucket, Prefix=f"{prefix}/")
    keys: list[ObjectIdentifierTypeDef] = [
        {"Key": item["Key"]} for item in listing.get("Contents", [])
    ]
    if keys:
        client.delete_objects(Bucket=config.bucket, Delete={"Objects": keys})


@pytest.fixture
def manifest_ref(store: S3ArtifactStore, tmp_path: Path) -> ArtifactRef:
    # The registry stays in memory: this test is about the bucket, the database has its own.
    root = CompositionRoot(
        Settings(),
        corpus_root=SAMPLE,
        workspace=tmp_path / "publisher",
        subsets=("FD001",),
        corpora=InMemoryCorpusRepository(),
        store=store,
    )
    known = KnownCorpora.default().named(CORPUS)
    return root.services.publish_corpus(
        PublishCorpusCommand(
            name=known.name,
            source=known.source,
            licence=known.licence,
            window=WINDOW,
            validation_fraction=0.5,
            seed=1,
        )
    )


def reader_elsewhere(store: S3ArtifactStore, tmp_path: Path) -> BlockCorpusArchive:
    """An archive over a workspace the publisher never wrote to: the other machine."""
    return BlockCorpusArchive(store, tmp_path / "reader", PublishedCorpusManifestAssembler())


def test_a_corpus_published_to_the_bucket_reads_back_on_another_machine(
    store: S3ArtifactStore, manifest_ref: ArtifactRef, tmp_path: Path
) -> None:
    elsewhere = reader_elsewhere(store, tmp_path)

    manifest = elsewhere.read_manifest(manifest_ref)
    windows = elsewhere.read_windows(manifest.archived, manifest.split.training)

    assert manifest.corpus == CORPUS
    assert len(windows) > 0
    assert (tmp_path / "reader").is_dir(), "the block is fetched to disk so it can be mapped"


def test_a_training_run_gets_batches_without_opening_a_source_file(
    store: S3ArtifactStore, manifest_ref: ArtifactRef, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from emblema.shared.adapters.loaders.window_dataset import WindowDataset
    from emblema.shared.adapters.loaders.window_loader import WindowLoader

    elsewhere = reader_elsewhere(store, tmp_path)
    manifest = elsewhere.read_manifest(manifest_ref)

    windows = elsewhere.read_windows(manifest.archived, manifest.split.training)
    loader = WindowLoader(windows, batch_size=BATCH_SIZE, seed=manifest.split_seed)
    batch = next(iter(loader.batches_of(0)))

    assert WindowDataset(windows)
    assert batch.channel_ids.shape[0] == min(BATCH_SIZE, len(windows))
    assert batch.features.shape[2] > 0
