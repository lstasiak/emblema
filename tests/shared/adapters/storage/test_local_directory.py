from pathlib import Path

import pytest

from emblema.shared.adapters.storage.files import CHUNK_SIZE
from emblema.shared.adapters.storage.local_directory import LocalDirectoryArtifactStore
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.ports.exceptions import ArtifactNotFoundError


def test_no_temporary_file_is_left_behind(tmp_path: Path) -> None:
    store = LocalDirectoryArtifactStore(tmp_path)

    ref = store.put(b"weights")

    assert sorted(path.name for path in tmp_path.rglob("*") if path.is_file()) == [
        ref.checksum.digest
    ]


def test_a_reference_climbing_out_of_the_root_holds_nothing(tmp_path: Path) -> None:
    content = b"not an artifact"
    (tmp_path / "elsewhere").write_bytes(content)
    store = LocalDirectoryArtifactStore(tmp_path / "artifacts")
    ref = ArtifactRef("../elsewhere", Checksum.of_bytes(content))

    assert not store.exists(ref)
    with pytest.raises(ArtifactNotFoundError):
        store.get(ref)
    with pytest.raises(ArtifactNotFoundError):
        store.get_file(ref, tmp_path / "fetched.bin")


def test_an_artifact_longer_than_a_chunk_survives_the_round_trip(tmp_path: Path) -> None:
    # Chunking is where a file-shaped store can silently lose or duplicate bytes, and every
    # artifact this port was widened for is longer than one chunk.
    content = bytes(range(256)) * (4 * CHUNK_SIZE // 256 + 1)
    source = tmp_path / "long.bin"
    source.write_bytes(content)
    store = LocalDirectoryArtifactStore(tmp_path / "artifacts")
    destination = tmp_path / "fetched.bin"

    ref = store.put_file(source)
    store.get_file(ref, destination)

    assert ref.checksum == Checksum.of_bytes(content)
    assert destination.read_bytes() == content
