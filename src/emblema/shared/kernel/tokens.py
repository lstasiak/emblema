"""The token representation every context reads: a window of tokens and the token it is made of.

A window of sensor data is a set of tokens, each one channel observed at one instant, plus the
static features of the window as tokens that carry no time. The channel count is not an axis of
anything: a window of three channels and a window of thirty are the same kind of value. The Catalog
produces these windows, Pretraining trains on them and Serving infers on them, so their shape and
meaning are fixed here, where every context can see them.
"""

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Self

from emblema.shared.kernel.exceptions import InvalidTokenWindowError

# Channel identifiers are positive. Zero marks padding wherever windows are laid out in arrays, so a
# padding position can never be mistaken for a channel even where a mask is mishandled.
PADDING_CHANNEL_ID = 0


@dataclass(frozen=True)
class Token:
    """One row of a token window.

    A token is valid because its window is: the window checks every invariant over its columns
    once, and this row view carries no check of its own.

    Attributes:
        channel_id: Entry of the channel vocabulary the token comes from; positive.
        value: Observed value after per-channel normalisation; finite.
        time: Position within the window as a fraction of its length, in ``[0, 1]``; ``0`` for a
            timeless token.
        gap: Time since the previous token of the same channel in the window, or since the start of
            the window for the first one, as a fraction of the window length; never above ``time``;
            ``0`` for a timeless token.
        timeless: Whether the token is a static feature of the window rather than an observation. A
            timeless token has no time to encode.
    """

    channel_id: int
    value: float
    time: float
    gap: float
    timeless: bool = False


@dataclass(frozen=True)
class TokenWindow:
    """The tokens of one window, in canonical order, one column per token attribute.

    Invariants: every column has one entry per token; at least one token is timed, because a
    window of static features alone holds no measurement; channel identifiers are positive; values
    are finite; a timed token has ``time`` in ``[0, 1]`` and ``gap`` in ``[0, time]``; a timeless
    token has both at ``0``. Tokens are in canonical order — timeless tokens first, by channel and
    value, then timed tokens by time, channel and value — so two windows built from the same tokens
    in any order are equal. ``of`` puts tokens in that order; the constructor rejects any other.

    Attributes:
        channel_ids: Channel vocabulary entry per token.
        values: Normalised value per token.
        times: Position within the window per token.
        gaps: Time since the previous token of the same channel per token.
        timeless: Whether each token is a static feature of the window.
    """

    channel_ids: tuple[int, ...]
    values: tuple[float, ...]
    times: tuple[float, ...]
    gaps: tuple[float, ...]
    timeless: tuple[bool, ...]

    def __post_init__(self) -> None:
        count = len(self.channel_ids)
        if not (
            len(self.values) == len(self.times) == len(self.gaps) == len(self.timeless) == count
        ):
            raise InvalidTokenWindowError("every column must hold one entry per token")
        if all(self.timeless):
            raise InvalidTokenWindowError("a window must hold at least one timed token")
        rows = zip(self.channel_ids, self.values, self.times, self.gaps, self.timeless, strict=True)
        previous: tuple[int, float, int, float, float] | None = None
        for index, (channel_id, value, time, gap, timeless) in enumerate(rows):
            if channel_id <= PADDING_CHANNEL_ID:
                raise InvalidTokenWindowError(
                    f"token {index}: channel id must be positive, got {channel_id}"
                )
            if not math.isfinite(value):
                raise InvalidTokenWindowError(f"token {index}: value must be finite, got {value}")
            if timeless:
                if time != 0.0 or gap != 0.0:
                    raise InvalidTokenWindowError(
                        f"token {index}: a timeless token carries time 0 and gap 0"
                    )
            elif not 0.0 <= time <= 1.0:
                raise InvalidTokenWindowError(
                    f"token {index}: time must lie within the window, in [0, 1], got {time}"
                )
            elif not 0.0 <= gap <= time:
                raise InvalidTokenWindowError(
                    f"token {index}: gap must lie in [0, time], got {gap} at time {time}"
                )
            key = (0 if timeless else 1, time, channel_id, value, gap)
            if previous is not None and key < previous:
                raise InvalidTokenWindowError(f"token {index} is out of canonical order")
            previous = key

    @classmethod
    def of(cls, tokens: Iterable[Token]) -> Self:
        """The window holding ``tokens``, whatever order they arrive in."""
        ordered = sorted(tokens, key=_canonical_key)
        return cls(
            channel_ids=tuple(token.channel_id for token in ordered),
            values=tuple(token.value for token in ordered),
            times=tuple(token.time for token in ordered),
            gaps=tuple(token.gap for token in ordered),
            timeless=tuple(token.timeless for token in ordered),
        )

    def __len__(self) -> int:
        return len(self.channel_ids)

    def __iter__(self) -> Iterator[Token]:
        rows = zip(self.channel_ids, self.values, self.times, self.gaps, self.timeless, strict=True)
        return (Token(*row) for row in rows)


def _canonical_key(token: Token) -> tuple[int, float, int, float, float]:
    return (0 if token.timeless else 1, token.time, token.channel_id, token.value, token.gap)
