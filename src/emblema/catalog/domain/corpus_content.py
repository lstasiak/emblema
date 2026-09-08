from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidCorpusContentError
from emblema.shared.kernel.checksums import Checksum


@dataclass(frozen=True)
class CorpusContent:
    """What validation established about the data of a corpus version.

    Attributes:
        checksum: Checksum of the raw data as read by the corpus reader.
        record_count: Number of records the reader found; at least one.
    """

    checksum: Checksum
    record_count: int

    def __post_init__(self) -> None:
        if self.record_count < 1:
            raise InvalidCorpusContentError(
                f"record count must be positive, got {self.record_count}"
            )
