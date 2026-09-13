from torch import Tensor, nn


class MaskedMeanPooling(nn.Module):
    """One embedding of a window: the mean of its observed token states.

    Padding positions carry a state like any other, so they are weighted out rather than sliced
    out — the token count stays the one dynamic axis and the operation traces for export. A window
    with no observed token averages over nothing and yields zeros rather than dividing by zero;
    rejecting such a window is the caller's job, this only keeps the numbers finite.
    """

    def forward(self, states: Tensor, padding_mask: Tensor) -> Tensor:
        observed = (~padding_mask).unsqueeze(-1).to(states.dtype)
        return (states * observed).sum(dim=1) / observed.sum(dim=1).clamp(min=1.0)
