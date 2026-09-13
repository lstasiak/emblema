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
    """

    CHANNEL = "channel"
    BLOCK = "block"
    TOKEN = "token"
