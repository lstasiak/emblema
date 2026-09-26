from typing import ClassVar

from torch import Tensor, nn


class TailPooling(nn.Module):
    """The mean over the observed states in the last share of a window.

    A remaining life is read from where a unit stands as the window ends, and the mean over the
    whole window blurs that with everything before it; the mean over the last cycles keeps the
    end and still averages out the noise of a single reading. Static features of the window are
    current at every instant and stay in the tail. Padding is weighted out, so the token count
    stays the one dynamic axis and the operation traces for export; a window with nothing in its
    tail pools to zeros rather than dividing by zero.

    Attributes:
        share: Share of the window's length the tail keeps, from its end.
    """

    # A token's time is stored in single precision, so the first instant of a tail can land a
    # hair before the share's edge; a tolerance far above that rounding and far below any
    # spacing of readings puts it back.
    TOLERANCE: ClassVar[float] = 1e-3

    def __init__(self, share: float) -> None:
        super().__init__()
        self.share = share

    def forward(
        self, states: Tensor, padding_mask: Tensor, timestamps: Tensor, timeless: Tensor
    ) -> Tensor:
        late = ~padding_mask & (timeless | (timestamps >= 1.0 - self.share - self.TOLERANCE))
        weights = late.unsqueeze(-1).to(states.dtype)
        return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
