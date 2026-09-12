from typing import Self

from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.encoder_block import EncoderBlock
from emblema.pretraining.adapters.encoder.fourier_time_encoding import FourierTimeEncoding
from emblema.pretraining.adapters.encoder.learned_channel_embedding import LearnedChannelEmbedding
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture
from emblema.shared.kernel.tokens import N_FEATURES


class SetEncoder(nn.Module):
    """Maps a window of tokens to one state per token, indifferent to their order and their count.

    The token count is the only axis that varies — channels enter through their embedding, not
    through an axis — so permuting the tokens permutes their states, and appending padding leaves
    the states of the observed tokens unchanged. The output is a state per position, padding
    positions included: an objective scores the observed ones, ``MaskedMeanPooling`` averages them
    into one embedding of the window.

    The two input modules are injected, so a variant of either — a learned time encoding, a
    channel vector derived from a description — replaces the standard one without the rest of the
    encoder noticing. ``for_vocabulary`` builds the standard pair, which is what the architecture's
    ``parameter_count`` describes. Dropout is regularisation of a run, not part of the shape, so it
    is an argument here rather than a field of the architecture.

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
