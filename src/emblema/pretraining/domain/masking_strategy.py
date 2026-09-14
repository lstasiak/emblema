from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidMaskingStrategyError


@dataclass(frozen=True, kw_only=True)
class MaskingStrategy:
    """How much of a window is hidden from the encoder, and in what shapes.

    Three independent draws per window, and a token is hidden if any reaches it: whole channels, so
    a value must be inferred from other channels as with a sensor set never seen; blocks of a
    channel's time, so it cannot be read off its neighbours; single tokens, as the minority. The
    rates are the primitives, being what an ablation turns and a vectorised draw needs; the fraction
    they add up to is ``expected_ratio``, measured on data rather than promised.

    Invariants: every rate lies in ``[0, 1]``; ``block_span`` lies in ``(0, 1]``; at least one
    rate is positive, since a strategy that hides nothing trains nothing; and no draw hides
    everything with certainty, since a window with nothing visible leaves nothing to infer from.

    Attributes:
        channel_rate: Probability that a channel present in the window is hidden whole.
        block_rate: Probability that a channel present in the window loses one block of its time.
        block_span: Length of that block as a fraction of the window; the block lies inside the
            window. A timeless token has no time and is never inside a block.
        token_rate: Probability that a token is hidden on its own.
    """

    channel_rate: float
    block_rate: float
    block_span: float
    token_rate: float

    def __post_init__(self) -> None:
        for label, rate in (
            ("channel_rate", self.channel_rate),
            ("block_rate", self.block_rate),
            ("token_rate", self.token_rate),
        ):
            if not 0.0 <= rate <= 1.0:
                raise InvalidMaskingStrategyError(f"{label} must lie in [0, 1], got {rate}")
        if not 0.0 < self.block_span <= 1.0:
            raise InvalidMaskingStrategyError(
                f"block_span must lie in (0, 1], got {self.block_span}"
            )
        if self.expected_ratio == 0.0:
            raise InvalidMaskingStrategyError("a strategy must hide something")
        if self.channel_rate == 1.0 or self.token_rate == 1.0 or self.expected_ratio == 1.0:
            raise InvalidMaskingStrategyError("a strategy must leave something visible")

    @property
    def expected_ratio(self) -> float:
        """The fraction of a window's tokens the three draws hide together.

        Exact for a window whose channels hold equally many tokens spread evenly over its length;
        a token survives all three draws with the product of their survival probabilities.
        """
        survives = (
            (1.0 - self.channel_rate)
            * (1.0 - self.block_rate * self.block_span)
            * (1.0 - self.token_rate)
        )
        return 1.0 - survives
