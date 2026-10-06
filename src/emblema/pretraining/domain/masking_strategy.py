from dataclasses import dataclass

from emblema.pretraining.domain.exceptions import InvalidMaskingStrategyError
from emblema.pretraining.domain.mask_kind import MaskKind


@dataclass(frozen=True, kw_only=True)
class MaskingStrategy:
    """How much of a window is hidden from the encoder, and in what shapes.

    Independent draws per window, and a token is hidden if any reaches it: whole channels, so
    a value must be inferred from other channels as with a sensor set never seen; blocks of a
    channel's time, so it cannot be read off its neighbours; single tokens, as the minority; and
    the window's tail across every channel, so the hidden part is the future of what is visible
    and cannot be interpolated. The rates are the primitives, being what an ablation turns and a
    vectorised draw needs; the fraction they add up to is ``expected_ratio``, measured on data
    rather than promised.

    Invariants: every rate lies in ``[0, 1]``; ``block_span`` lies in ``(0, 1]``; a horizon that
    is drawn has spans with ``0 < horizon_min_span <= horizon_max_span < 1``, and one that is not
    states no spans, since a span nothing draws would be a setting the run does not have; at
    least one rate is positive, since a strategy that hides nothing trains nothing; and no draw
    hides everything with certainty, since a window with nothing visible leaves nothing to infer
    from.

    Attributes:
        channel_rate: Probability that a channel present in the window is hidden whole.
        block_rate: Probability that a channel present in the window loses one block of its time.
        block_span: Length of that block as a fraction of the window; the block lies inside the
            window. A timeless token has no time and is never inside a block.
        token_rate: Probability that a token is hidden on its own.
        horizon_rate: Probability that a window loses its tail, every channel at once; zero where
            no tail is drawn.
        horizon_min_span: Shortest tail, as a fraction of the window.
        horizon_max_span: Longest tail; each window's is drawn uniformly between the two, so a
            run is not taught one length of future.
    """

    channel_rate: float
    block_rate: float
    block_span: float
    token_rate: float
    horizon_rate: float = 0.0
    horizon_min_span: float = 0.0
    horizon_max_span: float = 0.0

    def __post_init__(self) -> None:
        for label, rate in (
            ("channel_rate", self.channel_rate),
            ("block_rate", self.block_rate),
            ("token_rate", self.token_rate),
            ("horizon_rate", self.horizon_rate),
        ):
            if not 0.0 <= rate <= 1.0:
                raise InvalidMaskingStrategyError(f"{label} must lie in [0, 1], got {rate}")
        if not 0.0 < self.block_span <= 1.0:
            raise InvalidMaskingStrategyError(
                f"block_span must lie in (0, 1], got {self.block_span}"
            )
        if self.horizon_rate == 0.0:
            if self.horizon_min_span != 0.0 or self.horizon_max_span != 0.0:
                raise InvalidMaskingStrategyError(
                    "a horizon that is never drawn states no spans, got "
                    f"[{self.horizon_min_span}, {self.horizon_max_span}]"
                )
        elif not 0.0 < self.horizon_min_span <= self.horizon_max_span < 1.0:
            raise InvalidMaskingStrategyError(
                "horizon spans must satisfy 0 < horizon_min_span <= horizon_max_span < 1, got "
                f"[{self.horizon_min_span}, {self.horizon_max_span}]"
            )
        if self.expected_ratio == 0.0:
            raise InvalidMaskingStrategyError("a strategy must hide something")
        if self.channel_rate == 1.0 or self.token_rate == 1.0 or self.expected_ratio == 1.0:
            raise InvalidMaskingStrategyError("a strategy must leave something visible")

    @property
    def has_horizon(self) -> bool:
        """Whether any window loses its tail."""
        return self.horizon_rate > 0.0

    def draws(self, kind: MaskKind) -> bool:
        """Whether the draw a kind of mask is named after can hide anything under this strategy.

        A channel can also be left without a visible token by the other draws; that outcome
        counts as a hidden channel, but the strategy did not ask for it.
        """
        rate = {
            MaskKind.CHANNEL: self.channel_rate,
            MaskKind.BLOCK: self.block_rate,
            MaskKind.TOKEN: self.token_rate,
            MaskKind.HORIZON: self.horizon_rate,
        }[kind]
        return rate > 0.0

    @property
    def horizon_mean_span(self) -> float:
        """The tail a window loses on average, when it loses one."""
        return (self.horizon_min_span + self.horizon_max_span) / 2.0

    @property
    def expected_ratio(self) -> float:
        """The fraction of a window's tokens the draws hide together.

        Exact for a window whose channels hold equally many tokens spread evenly over its length,
        where a token survives every draw with the product of their survival probabilities. Blocks
        and the tail both reach a token by its time, so where both are drawn the product of their
        averages over time is an approximation.
        """
        survives = (
            (1.0 - self.channel_rate)
            * (1.0 - self.block_rate * self.block_span)
            * (1.0 - self.token_rate)
            * (1.0 - self.horizon_rate * self.horizon_mean_span)
        )
        return 1.0 - survives
