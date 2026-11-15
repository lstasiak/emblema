from enum import Enum


class MaskKind(Enum):
    """The ways a token can be hidden, told apart because each has its own trivial baseline.

    A hidden token whose channel keeps no visible token in the window cannot be interpolated,
    whatever draw hid it, so the kinds are read off the outcome rather than the draw.

    Attributes:
        CHANNEL: The whole channel is hidden; the token can only be inferred from other channels.
        BLOCK: A span of the channel's time is hidden; visible tokens of the channel remain on at
            least one side.
        TOKEN: The token is hidden on its own, between visible neighbours of its channel.
        HORIZON: The token lies in the window's hidden tail, with visible tokens of its channel
            only before it; it can only be forecast.
        INTERVAL: The token lies in an interval of the window's time hidden in every channel at
            once; its channel shows something outside the interval, but nothing measured at the
            same time is visible in any other.
    """

    CHANNEL = "channel"
    BLOCK = "block"
    TOKEN = "token"
    HORIZON = "horizon"
    INTERVAL = "interval"
