"""Windows the block tests and the archive contract share, and what a block makes of them."""

import numpy as np

from emblema.shared.kernel.tokens import Token, TokenWindow


def window(*tokens: Token) -> TokenWindow:
    return TokenWindow.of(tokens)


TIMED = window(
    Token(channel_id=1, value=-0.5, time=0.0, gap=0.0),
    Token(channel_id=2, value=0.25, time=0.5, gap=0.5),
    Token(channel_id=1, value=1.5, time=1.0, gap=1.0),
)
WITH_STATIC = window(
    Token(channel_id=7, value=2.0, time=0.0, gap=0.0, timeless=True),
    Token(channel_id=1, value=0.125, time=0.25, gap=0.25),
)
# Values a float32 cannot hold: what a block gives back for these is not what went in, only what
# went in rounded to the stored width. The other windows above are exact at every width.
INEXACT = window(
    Token(channel_id=1, value=1 / 3, time=1 / 3, gap=1 / 3),
    Token(channel_id=2, value=2 / 3, time=2 / 3, gap=2 / 3),
)
# Two channels observed closer together than a float32 tells apart, the later one on the lower
# identifier: a valid window that reads back out of canonical order at that width.
COLLAPSING = window(
    Token(channel_id=2, value=0.1, time=0.5, gap=0.5),
    Token(channel_id=1, value=0.2, time=0.5 + 1e-9, gap=0.5 + 1e-9),
)


def stored_at(window: TokenWindow, dtype: str = "<f4") -> TokenWindow:
    """The window as a block storing measurements at ``dtype`` gives it back."""
    return TokenWindow(
        channel_ids=window.channel_ids,
        values=_cast(window.values, dtype),
        times=_cast(window.times, dtype),
        gaps=_cast(window.gaps, dtype),
        timeless=window.timeless,
    )


def _cast(column: tuple[float, ...], dtype: str) -> tuple[float, ...]:
    return tuple(np.asarray(column, dtype=dtype).tolist())
