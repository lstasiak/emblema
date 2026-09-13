from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidEncoderArchitectureError
from emblema.shared.kernel.tokens import N_FEATURES


@dataclass(frozen=True, kw_only=True)
class EncoderArchitecture:
    """The shape of the encoder: what a trained backbone is identified by and its weights load into.

    Invariants: every count is positive; ``width`` is a multiple of ``heads``. The vocabulary size
    is an argument of ``parameter_count`` rather than a field, because the same architecture
    trained over two vocabularies is the same architecture with two embedding tables.

    Attributes:
        width: Size of a token's representation throughout the encoder.
        heads: Attention heads per block; each attends over ``width // heads`` features.
        layers: Number of encoder blocks.
        feedforward_width: Hidden size of the feed-forward network inside each block.
        time_frequencies: Fourier frequencies the position of a token in its window is encoded
            with.
    """

    width: int
    heads: int
    layers: int
    feedforward_width: int
    time_frequencies: int

    def __post_init__(self) -> None:
        counts = (
            ("width", self.width),
            ("heads", self.heads),
            ("layers", self.layers),
            ("feedforward_width", self.feedforward_width),
            ("time_frequencies", self.time_frequencies),
        )
        for label, count in counts:
            if count < 1:
                raise InvalidEncoderArchitectureError(f"{label} must be positive, got {count}")
        if self.width % self.heads:
            raise InvalidEncoderArchitectureError(
                f"width must be a multiple of heads, got width {self.width} and {self.heads} heads"
            )

    def parameter_count(self, vocabulary_size: int) -> int:
        """Parameters of the encoder in its standard composition over ``vocabulary_size`` channels.

        Exact — weights and biases of every layer, the padding row of the channel table included —
        so that a built model is held to this number by a test rather than trusted to match it.

        Raises:
            InvalidEncoderArchitectureError: If ``vocabulary_size`` is not positive.
        """
        if vocabulary_size < 1:
            raise InvalidEncoderArchitectureError(
                f"vocabulary size must be positive, got {vocabulary_size}"
            )
        width, hidden = self.width, self.feedforward_width
        channel_table = (vocabulary_size + 1) * width
        value_projection = self._linear(N_FEATURES, width)
        time_encoding = self._linear(2 * self.time_frequencies, width)
        attention = self._linear(width, 3 * width) + self._linear(width, width)
        feedforward = self._linear(width, hidden) + self._linear(hidden, width)
        block = 2 * self._layer_norm(width) + attention + feedforward
        return (
            channel_table
            + value_projection
            + time_encoding
            + self.layers * block
            + self._layer_norm(width)
        )

    @staticmethod
    def _linear(inputs: int, outputs: int) -> int:
        return inputs * outputs + outputs

    @staticmethod
    def _layer_norm(width: int) -> int:
        return 2 * width
