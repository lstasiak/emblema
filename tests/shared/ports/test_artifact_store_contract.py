"""Contract of the ArtifactStore port, run against every adapter.

The S3 adapter is exercised against whatever bucket the settings describe: the local stack by
default, the remote bucket when the process is started with its environment file. Running the
same tests against both is the proof that switching stores is a matter of configuration.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

import pytest

from emblema.config.settings import Settings
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.storage.layout import content_address
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.artifact_store import ArtifactStore, Retention
from emblema.shared.ports.exceptions import ArtifactIntegrityError, ArtifactNotFoundError

if TYPE_CHECKING:
    from mypy_boto3_s3.type_defs import ObjectIdentifierTypeDef

CONTENT = b"backbone weights"


class Harness(NamedTuple):
    """A store under test plus a way to corrupt what it stored, bypassing the port."""

    store: ArtifactStore
    corrupt: Callable[[ArtifactRef], None]


def in_memory(tmp_path: Path) -> Iterator[Harness]:
    store = InMemoryArtifactStore()

    def corrupt(ref: ArtifactRef) -> None:
        store._content[ref.key] = b"tampered"

    yield Harness(store, corrupt)


def local_directory(tmp_path: Path) -> Iterator[Harness]:
    root = tmp_path / "artifacts"

    def corrupt(ref: ArtifactRef) -> None:
        root.joinpath(*ref.key.split("/")).write_bytes(b"tampered")

    yield Harness(LocalDirectoryArtifactStore(root), corrupt)


def s3(tmp_path: Path) -> Iterator[Harness]:
    config = Settings().artifact_store
    # Each session writes under its own prefix so that parallel runs against one shared bucket
    # never see each other's objects. The prefix is emptied afterwards through the raw client,
    # because the port itself has no delete; it sits under the transient segment so that whatever
    # a killed run leaves behind is still swept by the bucket's lifecycle rule.
    prefix = f"{config.key_prefix}/{Retention.TRANSIENT}/{uuid4().hex}"
    store = S3ArtifactStore.connect(
        endpoint_url=config.endpoint_url,
        region=config.region,
        access_key=config.access_key.get_secret_value() if config.access_key else None,
        secret_key=config.secret_key.get_secret_value() if config.secret_key else None,
        bucket=config.bucket,
        key_prefix=prefix,
    )
    client = store._client

    def corrupt(ref: ArtifactRef) -> None:
        client.put_object(Bucket=config.bucket, Key=f"{prefix}/{ref.key}", Body=b"tampered")

    yield Harness(store, corrupt)

    listing = client.list_objects_v2(Bucket=config.bucket, Prefix=f"{prefix}/")
    keys: list[ObjectIdentifierTypeDef] = [
        {"Key": item["Key"]} for item in listing.get("Contents", [])
    ]
    if keys:
        client.delete_objects(Bucket=config.bucket, Delete={"Objects": keys})


ADAPTERS = [
    pytest.param(in_memory, id="in_memory"),
    pytest.param(local_directory, id="local_directory"),
    pytest.param(s3, id="s3", marks=pytest.mark.integration),
]


@pytest.fixture(params=ADAPTERS)
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Harness]:
    factory: Callable[[Path], Iterator[Harness]] = request.param
    yield from factory(tmp_path)


@pytest.fixture
def store(harness: Harness) -> ArtifactStore:
    return harness.store


def test_put_then_get_returns_the_same_bytes(store: ArtifactStore) -> None:
    ref = store.put(CONTENT)

    assert store.get(ref) == CONTENT


def test_reference_carries_the_checksum_of_the_content(store: ArtifactStore) -> None:
    ref = store.put(CONTENT)

    assert ref.checksum == Checksum.of_bytes(CONTENT)


@pytest.mark.parametrize("retention", list(Retention))
def test_key_reads_retention_then_algorithm_then_digest(
    store: ArtifactStore, retention: Retention
) -> None:
    ref = store.put(CONTENT, retention)

    assert ref.key == f"{retention}/sha256/{ref.checksum.digest}"


def test_storing_the_same_content_twice_yields_the_same_reference(store: ArtifactStore) -> None:
    first = store.put(CONTENT)
    second = store.put(CONTENT)

    assert first == second
    assert store.get(first) == CONTENT


def test_exists_reports_stored_and_missing_references(store: ArtifactStore) -> None:
    missing = content_address(b"never stored", Retention.DURABLE)

    stored = store.put(CONTENT)

    assert store.exists(stored)
    assert not store.exists(missing)


def test_empty_content_is_an_artifact_too(store: ArtifactStore) -> None:
    ref = store.put(b"")

    assert store.get(ref) == b""


def test_get_of_an_unknown_reference_fails(store: ArtifactStore) -> None:
    missing = content_address(b"never stored", Retention.DURABLE)

    with pytest.raises(ArtifactNotFoundError):
        store.get(missing)


def test_get_rejects_content_that_no_longer_matches_the_reference(harness: Harness) -> None:
    ref = harness.store.put(CONTENT)
    harness.corrupt(ref)

    with pytest.raises(ArtifactIntegrityError):
        harness.store.get(ref)


# --- artifacts that travel as files --------------------------------------------------------------


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source.bin"
    path.write_bytes(CONTENT)
    return path


def test_a_file_and_its_bytes_are_one_artifact(store: ArtifactStore, source: Path) -> None:
    from_file = store.put_file(source)

    assert from_file == store.put(CONTENT)
    assert store.get(from_file) == CONTENT


def test_get_file_lays_the_artifact_at_the_destination(
    store: ArtifactStore, source: Path, tmp_path: Path
) -> None:
    ref = store.put_file(source)
    destination = tmp_path / "fetched" / "corpus.bin"

    store.get_file(ref, destination)

    assert destination.read_bytes() == CONTENT


def test_get_file_replaces_whatever_was_at_the_destination(
    store: ArtifactStore, source: Path, tmp_path: Path
) -> None:
    ref = store.put_file(source)
    destination = tmp_path / "fetched.bin"
    destination.write_bytes(b"older run")

    store.get_file(ref, destination)

    assert destination.read_bytes() == CONTENT


def test_put_file_of_a_missing_source_fails(store: ArtifactStore, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store.put_file(tmp_path / "never written.bin")


def test_get_file_of_an_unknown_reference_fails(store: ArtifactStore, tmp_path: Path) -> None:
    missing = content_address(b"never stored", Retention.DURABLE)

    with pytest.raises(ArtifactNotFoundError):
        store.get_file(missing, tmp_path / "fetched.bin")


def test_get_file_leaves_nothing_behind_when_the_content_no_longer_matches(
    harness: Harness, source: Path, tmp_path: Path
) -> None:
    ref = harness.store.put_file(source)
    harness.corrupt(ref)
    destination = tmp_path / "fetched.bin"

    with pytest.raises(ArtifactIntegrityError):
        harness.store.get_file(ref, destination)

    # A reader maps this file without hashing it, so a rejected artifact must not be there at all.
    assert not destination.exists()
