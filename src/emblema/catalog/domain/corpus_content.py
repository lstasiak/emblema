from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidCorpusContentError
from emblema.shared.kernel.checksums import Checksum


@dataclass(frozen=True)
class CorpusContent:
    """What validation established about the data of a corpus version.

    Invariants: at least one unit and at least one observation overall. A single unit may hold
    no observation at all, which is why nothing ties the two counts together.

    Attributes:
        checksum: Checksum of the raw data as read by the corpus reader.
        unit_count: Independent units the data holds, such as engines, machines or stays; the
            level at which the data is split, so the count that bounds generalisation.
        observation_count: Observed values, each one channel at one instant. Overlapping windows
            never multiply this count.
    """

    checksum: Checksum
    unit_count: int
    observation_count: int

    def __post_init__(self) -> None:
        if self.unit_count < 1:
            raise InvalidCorpusContentError(f"unit count must be positive, got {self.unit_count}")
        if self.observation_count < 1:
            raise InvalidCorpusContentError(
                f"observation count must be positive, got {self.observation_count}"
            )
