from dataclasses import dataclass

from emblema.evaluation.domain.exceptions import InvalidEncoderSettingError


@dataclass(frozen=True, kw_only=True)
class EncoderShape:
    """A shape of its own for an encoder trained from no weights, in place of its backbone's.

    A network that starts from nothing need not be the size of the backbone a campaign names:
    its weights come from no artifact, so any shape builds. Stated whole, because a width without
    its heads or its feed-forward width is not a shape.

    Invariants: every count is positive; the width is a multiple of the heads.

    Attributes:
        width: Size of a token's state throughout the encoder.
        heads: Attention heads per block.
        layers: Number of blocks.
        feedforward_width: Hidden size of each block's feed-forward network.
    """

    width: int
    heads: int
    layers: int
    feedforward_width: int

    def __post_init__(self) -> None:
        counts = (
            ("width", self.width),
            ("heads", self.heads),
            ("layers", self.layers),
            ("feedforward_width", self.feedforward_width),
        )
        for label, count in counts:
            if count < 1:
                raise InvalidEncoderSettingError(f"{label} must be positive, got {count}")
        if self.width % self.heads:
            raise InvalidEncoderSettingError(
                f"width must be a multiple of heads, got width {self.width} and {self.heads} heads"
            )
