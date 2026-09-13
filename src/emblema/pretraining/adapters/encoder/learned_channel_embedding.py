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

    def forward(self, channel_ids: Tensor) -> Tensor:
        return self.table(channel_ids)
