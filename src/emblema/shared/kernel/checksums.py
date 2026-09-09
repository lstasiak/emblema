"""Content checksums: the hash algorithm vocabulary and the checksum value object."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from emblema.shared.kernel.exceptions import InvalidChecksumError


class HashAlgorithm(StrEnum):
    """Hash functions a checksum may be computed with.

    Attributes:
        SHA256: The only algorithm in use; the member exists so that a checksum names its
            algorithm explicitly and a future change of function is a new member, not a
            reinterpretation of stored digests.
    """

    SHA256 = "sha256"

    @property
    def hex_length(self) -> int:
        return hashlib.new(self.value).digest_size * 2


@dataclass(frozen=True)
class Checksum:
    """Checksum of a blob of content, in canonical ``<algorithm>:<hex digest>`` form.

    Invariants: the digest is lowercase hexadecimal of exactly the length the algorithm
    produces.

    Attributes:
        algorithm: Hash function the digest was computed with.
        digest: Lowercase hexadecimal digest.
    """

    algorithm: HashAlgorithm
    digest: str

    def __post_init__(self) -> None:
        expected = self.algorithm.hex_length
        if len(self.digest) != expected:
            raise InvalidChecksumError(
                f"{self.algorithm} digest must have {expected} hex characters, "
                f"got {len(self.digest)}"
            )
        if not all(char in "0123456789abcdef" for char in self.digest):
            raise InvalidChecksumError("digest must be lowercase hexadecimal")

    @classmethod
    def of_bytes(cls, data: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> Self:
        return cls(algorithm, hashlib.new(algorithm.value, data).hexdigest())

    @classmethod
    def parse(cls, text: str) -> Self:
        """Build a checksum from its canonical string form.

        Raises:
            InvalidChecksumError: If the text lacks the ``algorithm:digest`` shape, names an
                unknown algorithm or carries a malformed digest.
        """
        algorithm_name, separator, digest = text.partition(":")
        if not separator:
            raise InvalidChecksumError(f"expected '<algorithm>:<digest>', got {text!r}")
        try:
            algorithm = HashAlgorithm(algorithm_name)
        except ValueError as error:
            raise InvalidChecksumError(f"unknown hash algorithm {algorithm_name!r}") from error
        return cls(algorithm, digest)

    def matches(self, data: bytes) -> bool:
        return self == Checksum.of_bytes(data, self.algorithm)

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.digest}"
