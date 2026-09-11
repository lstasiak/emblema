"""Channel schema of a corpus version: which quantities its tokens come from."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidChannelSchemaError


@dataclass(frozen=True)
class Channel:
    """One source of tokens in a corpus: a measured quantity, or a static feature of its units.

    Attributes:
        name: Identifier of the channel within its corpus, non-blank.
        unit: Physical unit of the values, when the source documents one.
        timeless: Whether the channel describes a whole unit rather than instants of it, such as a
            patient's age; its values become timeless tokens and are never counted as observations.
    """

    name: str
    unit: str | None = None
    timeless: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise InvalidChannelSchemaError(
                "channel name must be non-blank without surrounding whitespace"
            )


@dataclass(frozen=True)
class ChannelSchema:
    """The set of channels a corpus version carries, with unique names.

    Declaration order carries no meaning: a channel is identified by its name, and the same bytes
    described by the same channels in another order are the same data. Iteration and ``names``
    are sorted by name so that every consumer sees one deterministic order.

    Attributes:
        channels: Declared channels; at least one, names unique.
    """

    channels: frozenset[Channel]

    def __post_init__(self) -> None:
        if not self.channels:
            raise InvalidChannelSchemaError("channel schema must declare at least one channel")
        counts = Counter(channel.name for channel in self.channels)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise InvalidChannelSchemaError(f"duplicate channel names: {duplicates}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self)

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self) -> Iterator[Channel]:
        return iter(sorted(self.channels, key=lambda channel: channel.name))
