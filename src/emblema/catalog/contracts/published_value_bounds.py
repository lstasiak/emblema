from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class PublishedValueBounds:
    """The bounds one channel's values were held to before a corpus was published, and their effect.

    Attributes:
        channel: Name of the channel within its corpus.
        removed_below: A value below it was removed, so it reads as never measured.
        clipped_below: A value below it, and not removed, was raised to it.
        clipped_above: A value above it, and not removed, was lowered to it.
        removed_above: A value above it was removed.
        removed: Values removed, over every unit of the corpus.
        clipped: Values clipped and kept, over every unit of the corpus.
    """

    channel: str
    removed_below: float
    clipped_below: float
    clipped_above: float
    removed_above: float
    removed: int
    clipped: int
