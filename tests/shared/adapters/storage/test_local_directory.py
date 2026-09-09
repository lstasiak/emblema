from pathlib import Path

import pytest

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
