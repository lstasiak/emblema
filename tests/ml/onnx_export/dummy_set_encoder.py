"""A model with the target input shape and none of the target behaviour.

The real encoder does not exist yet and its architecture is still open. This stand-in exists to put
every construct that encoder will need — a channel embedding, continuous time encoding, a flag that
skips time, a padding mask, attention over a variable number of tokens, masked pooling — through the
exporter while the architecture can still be changed cheaply.
"""

import torch
from torch import Tensor, nn

from tests.ml.onnx_export.attention import AttentionKind, build_attention
from tests.ml.onnx_export.token_batch import N_CHANNELS, N_FEATURES


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attention: AttentionKind) -> None:
        super().__init__()
        self.attention = build_attention(attention, d_model, n_heads)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )

    def forward(self, h: Tensor, padding_mask: Tensor) -> Tensor:
        h = h + self.attention(self.attention_norm(h), padding_mask)
        return h + self.feedforward(self.feedforward_norm(h))


class DummySetEncoder(nn.Module):
    """Maps a window of tokens to one embedding, invariant to their order and to padding.

    The token count is the only structural axis: channels enter through an embedding of their
    identifier, not through a dimension of the tensor, so a window of three channels and a window
    of thirty are the same shape of input.

    Attributes:
        d_model: Width of the embedding the encoder produces.
        frequencies: Fixed Fourier frequencies the timestamp is encoded with, so the encoding is
            parameter-free and extrapolates beyond the range seen in training.
    """

    frequencies: Tensor

    def __init__(
        self,
        *,
        attention: AttentionKind = AttentionKind.SDPA,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        n_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.channel_embedding = nn.Embedding(N_CHANNELS, d_model)
        self.value_projection = nn.Linear(N_FEATURES, d_model)
        self.time_projection = nn.Linear(2 * n_frequencies, d_model)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, attention) for _ in range(n_layers)
        )
        self.register_buffer("frequencies", 2.0 ** torch.arange(n_frequencies, dtype=torch.float32))

    def forward(
        self,
        features: Tensor,
        channel_ids: Tensor,
        timestamps: Tensor,
        timeless: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        h = self.value_projection(features) + self.channel_embedding(channel_ids)
        angles = timestamps.unsqueeze(-1) * self.frequencies
        encoded_time = self.time_projection(torch.cat([angles.sin(), angles.cos()], dim=-1))
        # A timeless token has no timestamp. Substituting an artificial one would make the encoder
        # spend capacity learning to ignore a vector it cannot predict, so the encoding is dropped
        # instead — branch-free, because a Python `if` would not survive the export.
        h = h + torch.where(timeless.unsqueeze(-1), torch.zeros_like(encoded_time), encoded_time)
        for block in self.blocks:
            h = block(h, padding_mask)
        observed = (~padding_mask).unsqueeze(-1).to(h.dtype)
        return (h * observed).sum(dim=1) / observed.sum(dim=1).clamp(min=1.0)
