import pytest

from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.exceptions import InvalidArtifactRefError

CHECKSUM = Checksum.of_bytes(b"weights")


def test_holds_key_and_checksum() -> None:
    ref = ArtifactRef("backbones/sha256/abc", CHECKSUM)

    assert ref.key == "backbones/sha256/abc"
    assert ref.checksum == CHECKSUM


@pytest.mark.parametrize("key", ["", " ", " key", "key ", "\tkey\n"])
def test_rejects_blank_or_padded_key(key: str) -> None:
    with pytest.raises(InvalidArtifactRefError):
        ArtifactRef(key, CHECKSUM)
