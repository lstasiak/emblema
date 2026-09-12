from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention


class SelfAttention(nn.Module):
    """Full self-attention over the tokens of a window, with padding kept out of the keys.

    Exact and quadratic in the window: the window length bounds the cost, not an approximation of
    the attention. ``scaled_dot_product_attention`` and nothing else, because it is the
    implementation that returns numbers rather than NaN on a window with no observed token, in
    PyTorch and in the exported graph alike. Padding positions are hidden as keys only; they still
    produce a state, which the consumer masks.
    """

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)

    def forward(self, states: Tensor, padding_mask: Tensor) -> Tensor:
        query, key, value = self.qkv(states).chunk(3, dim=-1)
        keep = ~padding_mask[:, None, None, :]
        attended = scaled_dot_product_attention(
            self._split_heads(query),
            self._split_heads(key),
            self._split_heads(value),
            attn_mask=keep,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.projection(self._merge_heads(attended))

    def _split_heads(self, states: Tensor) -> Tensor:
        """``[batch, tokens, width]`` to ``[batch, heads, tokens, head width]``."""
        batch, tokens, width = states.shape
        return states.reshape(batch, tokens, self.heads, width // self.heads).transpose(1, 2)

    @staticmethod
    def _merge_heads(attended: Tensor) -> Tensor:
        batch, heads, tokens, head_width = attended.shape
        return attended.transpose(1, 2).reshape(batch, tokens, heads * head_width)
