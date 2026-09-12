from torch import Tensor, nn

from emblema.pretraining.adapters.encoder.self_attention import SelfAttention
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture


class EncoderBlock(nn.Module):
    """One residual block: attention over the window, then a feed-forward network per token.

    Each sub-layer normalises its input before acting on it and adds its output to the residual
    stream, so the stream itself is never normalised away and depth costs nothing in stability.
    """

    def __init__(self, architecture: EncoderArchitecture, dropout: float) -> None:
        super().__init__()
        width = architecture.width
        self.attention_norm = nn.LayerNorm(width)
        self.attention = SelfAttention(width, architecture.heads, dropout)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, architecture.feedforward_width),
            nn.GELU(),
            nn.Linear(architecture.feedforward_width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: Tensor, padding_mask: Tensor) -> Tensor:
        states = states + self.dropout(self.attention(self.attention_norm(states), padding_mask))
        return states + self.dropout(self.feedforward(self.feedforward_norm(states)))
