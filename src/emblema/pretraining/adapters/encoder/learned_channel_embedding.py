from torch import Tensor, nn

from emblema.shared.kernel.tokens import PADDING_CHANNEL_ID


class LearnedChannelEmbedding(nn.Module):
    """One vector per channel of the vocabulary, learned from data.

    The only part of the encoder that knows a channel, wrapped so that a vector derived from a
    channel's description can take its place behind the same call. The table has one row more than
    the vocabulary: row zero belongs to padding, stays at zero and receives no gradient.
    """

    def __init__(self, vocabulary_size: int, width: int) -> None:
        super().__init__()
        self.table = nn.Embedding(vocabulary_size + 1, width, padding_idx=PADDING_CHANNEL_ID)

    @property
    def vocabulary_size(self) -> int:
        """Channels the table has a row for, the padding row not counted."""
        return self.table.num_embeddings - 1

    @property
    def width(self) -> int:
        return self.table.embedding_dim

    def forward(self, channel_ids: Tensor) -> Tensor:
        return self.table(channel_ids)
