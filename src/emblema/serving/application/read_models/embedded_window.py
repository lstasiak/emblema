from dataclasses import dataclass
from math import isfinite

from emblema.serving.domain.exceptions import NonFiniteAnswerError


@dataclass(frozen=True, kw_only=True)
class EmbeddedWindow:
    """The representation of one window, beside what it was computed over and what was left out.

    Invariants: every component is a finite number, for the reason an answer must be.

    Attributes:
        embedding: The candidate's pooled representation of the window.
        channels_used: The channels it was computed over, sorted.
        channels_ignored: Channels the request had readings on that the model does not know.
        warnings: Every way the representation is less than what was asked, in words.
    """

    embedding: tuple[float, ...]
    channels_used: tuple[str, ...]
    channels_ignored: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in self.embedding):
            raise NonFiniteAnswerError("the model represented a window with a non-finite value")
