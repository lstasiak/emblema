"""Interchangeable attention implementations, so the spike can compare how each one exports.

The three differ only in how the padding mask reaches the softmax. That detail turns out to decide
what a window with no observed token produces, which is why the spike carries all three instead of
picking one up front.
"""

from enum import StrEnum

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention


class AttentionKind(StrEnum):
    """Attention implementations the dummy encoder can be built with.

    Attributes:
        SDPA: `scaled_dot_product_attention` with a boolean mask of positions to keep.
        MODULE: `torch.nn.MultiheadAttention` with `key_padding_mask`.
        MANUAL: Explicit scores, `masked_fill` with negative infinity, then softmax.
    """

    SDPA = "sdpa"
    MODULE = "module"
    MANUAL = "manual"


class SdpaAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, h: Tensor, padding_mask: Tensor) -> Tensor:
        query, key, value = _split_heads(self.qkv(h), self.n_heads)
        keep = ~padding_mask[:, None, None, :]
        attended = scaled_dot_product_attention(query, key, value, attn_mask=keep)
        return self.projection(_merge_heads(attended))


class ModuleAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, h: Tensor, padding_mask: Tensor) -> Tensor:
        attended, _ = self.attention(h, h, h, key_padding_mask=padding_mask, need_weights=False)
        return attended


class ManualAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, h: Tensor, padding_mask: Tensor) -> Tensor:
        query, key, value = _split_heads(self.qkv(h), self.n_heads)
        scores = query @ key.transpose(-2, -1) * query.shape[-1] ** -0.5
        scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        attended = torch.softmax(scores, dim=-1) @ value
        return self.projection(_merge_heads(attended))


def build_attention(kind: AttentionKind, d_model: int, n_heads: int) -> nn.Module:
    implementations = {
        AttentionKind.SDPA: SdpaAttention,
        AttentionKind.MODULE: ModuleAttention,
        AttentionKind.MANUAL: ManualAttention,
    }
    return implementations[kind](d_model, n_heads)


def _split_heads(qkv: Tensor, n_heads: int) -> tuple[Tensor, Tensor, Tensor]:
    batch, tokens, width = qkv.shape
    parts = qkv.reshape(batch, tokens, 3, n_heads, width // (3 * n_heads))
    query, key, value = parts.permute(2, 0, 3, 1, 4)
    return query, key, value


def _merge_heads(attended: Tensor) -> Tensor:
    batch, n_heads, tokens, head_dim = attended.shape
    return attended.transpose(1, 2).reshape(batch, tokens, n_heads * head_dim)
