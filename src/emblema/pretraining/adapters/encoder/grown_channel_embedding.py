from collections.abc import Iterable

import torch
from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.learned_channel_embedding import LearnedChannelEmbedding
from emblema.pretraining.domain.exceptions import InvalidEncoderArchitectureError
from emblema.shared.kernel.tokens import PADDING_CHANNEL_ID


class GrownChannelEmbedding(nn.Module):
    """A learnt channel table with rows grown for channels it was never trained on.

    A corpus published under a vocabulary that continues the backbone's carries channel
    identifiers past the end of the backbone's table. The table it learnt keeps its rows and
    whatever a transfer mode does to them; the channels past it get rows of their own, drawn
    from torch's generator when this is built, which no pretraining set and which are therefore
    trained under every mode. Padding stays in the learnt table's zero row.

    Attributes:
        learnt: The table pretraining set, over the channels it saw.
        grown: One row per channel past the learnt ones.
    """

    def __init__(self, learnt: LearnedChannelEmbedding, vocabulary_size: int) -> None:
        """Grow ``learnt`` to cover ``vocabulary_size`` channels.

        Raises:
            InvalidEncoderArchitectureError: If the vocabulary is not larger than the learnt one.
        """
        super().__init__()
        if vocabulary_size <= learnt.vocabulary_size:
            raise InvalidEncoderArchitectureError(
                f"a table over {learnt.vocabulary_size} channels does not grow to {vocabulary_size}"
            )
        self.learnt = learnt
        self.grown = nn.Embedding(vocabulary_size - learnt.vocabulary_size, learnt.width)
        self._first_grown = learnt.vocabulary_size + 1

    @property
    def vocabulary_size(self) -> int:
        return self.learnt.vocabulary_size + self.grown.num_embeddings

    def grown_parameters(self) -> Iterable[nn.Parameter]:
        return self.grown.parameters()

    def forward(self, channel_ids: Tensor) -> Tensor:
        beyond = channel_ids >= self._first_grown
        # A grown identifier is read from the learnt table as padding, whose row is zero, and
        # a learnt one from the grown table at its first row; ``where`` keeps the right one.
        learnt = self.learnt(channel_ids.masked_fill(beyond, PADDING_CHANNEL_ID))
        grown = self.grown((channel_ids - self._first_grown).clamp(min=0))
        return torch.where(beyond.unsqueeze(-1), grown, learnt)
