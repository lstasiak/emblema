from dataclasses import dataclass

from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.exceptions import InvalidArtifactRefError


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a stored artifact: where it lives and what its content must hash to.

    The checksum travels with the key so that a consumer can verify the bytes it fetched are
    the bytes that were registered, whichever store serves them.

    Attributes:
        key: Location of the artifact within the artifact store; opaque to the domain.
        checksum: Checksum of the artifact's content.
    """

    key: str
    checksum: Checksum

    def __post_init__(self) -> None:
        if not self.key or self.key != self.key.strip():
            raise InvalidArtifactRefError(
                "artifact key must be non-empty without surrounding whitespace"
            )
