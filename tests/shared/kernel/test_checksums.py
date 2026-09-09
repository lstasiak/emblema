import hashlib

import pytest

from emblema.shared.kernel.checksums import Checksum, HashAlgorithm
from emblema.shared.kernel.exceptions import InvalidChecksumError

DIGEST = hashlib.sha256(b"emblema").hexdigest()


def test_of_bytes_matches_hashlib() -> None:
    checksum = Checksum.of_bytes(b"emblema")

    assert checksum.algorithm is HashAlgorithm.SHA256
    assert checksum.digest == DIGEST


def test_str_is_algorithm_colon_digest() -> None:
    assert str(Checksum(HashAlgorithm.SHA256, DIGEST)) == f"sha256:{DIGEST}"


def test_parse_round_trips_through_str() -> None:
    checksum = Checksum.of_bytes(b"emblema")

    assert Checksum.parse(str(checksum)) == checksum


def test_matches_only_the_content_it_was_computed_from() -> None:
    checksum = Checksum.of_bytes(b"emblema")

    assert checksum.matches(b"emblema")
    assert not checksum.matches(b"emblema ")


@pytest.mark.parametrize(
    "digest",
    [DIGEST[:-1], DIGEST + "0", DIGEST.upper(), "g" * 64],
    ids=["too-short", "too-long", "uppercase", "non-hex"],
)
def test_rejects_malformed_digest(digest: str) -> None:
    with pytest.raises(InvalidChecksumError):
        Checksum(HashAlgorithm.SHA256, digest)


@pytest.mark.parametrize(
    "text",
    [DIGEST, f"md5:{DIGEST}", f"sha256:{DIGEST[:-1]}"],
    ids=["no-algorithm", "unknown-algorithm", "short-digest"],
)
def test_parse_rejects_malformed_text(text: str) -> None:
    with pytest.raises(InvalidChecksumError):
        Checksum.parse(text)
