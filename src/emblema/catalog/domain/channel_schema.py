"""Channel schema of a corpus version: which measured quantities it carries."""

from collections.abc import Iterator
from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidChannelSchemaError


@dataclass(frozen=True)
class Channel:
    """One measured quantity of a corpus.

    Attributes:
        name: Identifier of the channel within its corpus, non-blank.
        unit: Physical unit of the values, when the source documents one.
    """

    name: str
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise InvalidChannelSchemaError(
                "channel name must be non-blank without surrounding whitespace"
            )


@dataclass(frozen=True)
class ChannelSchema:
    """The channels of a corpus version, in declaration order, with unique names.

    Attributes:
        channels: Declared channels; at least one, names unique.
    """

    channels: tuple[Channel, ...]

    def __post_init__(self) -> None:
        if not self.channels:
            raise InvalidChannelSchemaError("channel schema must declare at least one channel")
        names = [channel.name for channel in self.channels]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise InvalidChannelSchemaError(f"duplicate channel names: {duplicates}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self) -> Iterator[Channel]:
        return iter(self.channels)
