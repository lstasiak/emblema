"""Published reference to a frozen corpus version and the channel description it carries."""

from collections import Counter
from dataclasses import dataclass

from emblema.catalog.contracts.exceptions import (
    InvalidChannelSpecError,
    InvalidCorpusVersionRefError,
)
from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime


@dataclass(frozen=True)
class ChannelSpec:
    """One channel of a published corpus version.

    Attributes:
        name: Identifier of the channel within its corpus, non-blank.
        unit: Physical unit of the values, when the source documents one.
        timeless: Whether the channel is a static feature of the corpus's units rather than a
            quantity measured over time.
    """

    name: str
    unit: str | None = None
    timeless: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise InvalidChannelSpecError(
                "channel name must be non-blank without surrounding whitespace"
            )


@dataclass(frozen=True)
class CorpusVersionRef:
    """What another context knows about a frozen corpus version: enough to pin the data.

    The reference carries no corpus, licence or record count; a consumer stores it as an opaque
    pair of identifier and checksum and treats the channels and regime as facts about the data.
    Channels are in name order: the Catalog's schema is a set, and one canonical order makes two
    references to the same data compare equal.

    Attributes:
        version_id: Identity of the version in the Catalog.
        checksum: Checksum of the data the version was frozen with.
        channels: Channels the data carries, sorted by name, names unique, at least one.
        sampling_regime: How the data is spaced in time.
    """

    version_id: CorpusVersionId
    checksum: Checksum
    channels: tuple[ChannelSpec, ...]
    sampling_regime: SamplingRegime

    def __post_init__(self) -> None:
        if not self.channels:
            raise InvalidCorpusVersionRefError("a reference must carry at least one channel")
        names = [channel.name for channel in self.channels]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise InvalidCorpusVersionRefError(f"duplicate channel names: {duplicates}")
        if names != sorted(names):
            raise InvalidCorpusVersionRefError("channels must be sorted by name")
