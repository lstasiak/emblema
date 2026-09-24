"""Windows spelled out token by token, so a feature test states its own answer."""

from collections.abc import Sequence

from emblema.shared.kernel.tokens import Token, TokenWindow


def timed(channel: int, readings: Sequence[tuple[float, float]]) -> list[Token]:
    """Tokens of one channel, each a value at a time, the gap taken from the previous one."""
    tokens, previous = [], 0.0
    for value, time in readings:
        tokens.append(Token(channel, value, time, time - previous))
        previous = time
    return tokens


def window(*channels: list[Token]) -> TokenWindow:
    return TokenWindow.of([token for channel in channels for token in channel])
