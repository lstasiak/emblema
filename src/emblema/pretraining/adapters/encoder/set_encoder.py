from typing import Self

from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.encoder_block import EncoderBlock
from emblema.pretraining.adapters.encoder.fourier_time_encoding import FourierTimeEncoding
from emblema.pretraining.adapters.encoder.grown_channel_embedding import GrownChannelEmbedding
from emblema.pretraining.adapters.encoder.learned_channel_embedding import LearnedChannelEmbedding
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.pretraining.domain.exceptions import InvalidEncoderArchitectureError
from emblema.shared.kernel.tokens import N_FEATURES


class SetEncoder(nn.Module):
    """Maps a window of tokens to one state per token, indifferent to their order and their count.

    The token count is the only axis that varies, channels entering through their embedding, so
    permuting the tokens permutes their states and padding leaves the observed states unchanged.
    Every position gets a state, padding included; an objective scores the observed ones and
    ``MaskedMeanPooling`` averages them. The time and channel modules are injected, so either can be
    replaced alone; ``for_vocabulary`` builds the standard pair ``parameter_count`` describes.
    Dropout is a run's argument, not a shape.

    Attributes:
        architecture: The shape this encoder was built to.
    """

    def __init__(
        self,
        architecture: EncoderArchitecture,
        *,
        channel_embedding: nn.Module,
        time_encoding: nn.Module,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.channel_embedding = channel_embedding
        self.value_projection = nn.Linear(N_FEATURES, architecture.width)
        self.time_encoding = time_encoding
        self.blocks = nn.ModuleList(
            EncoderBlock(architecture, dropout) for _ in range(architecture.layers)
        )
        self.norm = nn.LayerNorm(architecture.width)

    @classmethod
    def for_vocabulary(
        cls, architecture: EncoderArchitecture, vocabulary_size: int, *, dropout: float = 0.0
    ) -> Self:
        """The encoder in its standard composition over a vocabulary of ``vocabulary_size``."""
        return cls(
            architecture,
            channel_embedding=LearnedChannelEmbedding(vocabulary_size, architecture.width),
            time_encoding=FourierTimeEncoding(architecture.time_frequencies, architecture.width),
            dropout=dropout,
        )

    def grown_to(self, vocabulary_size: int) -> Self:
        """This encoder over a vocabulary that continues the one its channel table covers.

        The rows the table learnt stay as they are; a channel past them gets a row drawn from
        torch's generator as this is called (``GrownChannelEmbedding``). An encoder whose table
        already covers the vocabulary is returned as it is.

        Raises:
            InvalidEncoderArchitectureError: If the encoder's channel module is not the learnt
                table this grows.
        """
        table = self.channel_embedding
        if not isinstance(table, LearnedChannelEmbedding):
            raise InvalidEncoderArchitectureError(
                f"only a learnt channel table grows, not {type(table).__name__}"
            )
        if vocabulary_size <= table.vocabulary_size:
            return self
        self.channel_embedding = GrownChannelEmbedding(table, vocabulary_size)
        return self

    def forward(
        self,
        features: Tensor,
        channel_ids: Tensor,
        timestamps: Tensor,
        timeless: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        states = (
            self.value_projection(features)
            + self.channel_embedding(channel_ids)
            + self.time_encoding(timestamps, timeless)
        )
        for block in self.blocks:
            states = block(states, padding_mask)
        return self.norm(states)
