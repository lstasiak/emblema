from dataclasses import dataclass

from emblema.pretraining.domain.assessment.interval import Interval, Verdict
from emblema.pretraining.domain.mask_kind import MaskKind

MATCHED_BASELINE = {
    MaskKind.CHANNEL: "ridge",
    MaskKind.BLOCK: "interpolation",
    MaskKind.TOKEN: "interpolation",
}
# Where nothing of the channel is left, the strongest linear answer is the matched one, and a
# second comparison would repeat the first.
BEYOND_LINEAR = (MaskKind.BLOCK, MaskKind.TOKEN)


@dataclass(frozen=True)
class KindSummary:
    """One kind of mask over every validation unit, with the uncertainty of its comparisons.

    Attributes:
        kind: The kind of mask.
        apart: Whether this row is the channels reported apart.
        tokens: Hidden tokens the errors are averaged over.
        units: Validation units they come from.
        model_error: Mean squared error of the model.
        matched_error: Of the baseline matched to the kind.
        linear_error: Of the strongest linear baseline on the same sources.
        mean_error: Of predicting the channel mean.
        noise_floor: Mean noise variance of the targets; ``None`` where the corpus states none.
        matched_excess: Interval of the matched baseline's error minus the model's.
        linear_excess: Interval of the linear baseline's error minus the model's.
    """

    kind: MaskKind
    apart: bool
    tokens: int
    units: int
    model_error: float
    matched_error: float
    linear_error: float
    mean_error: float
    noise_floor: float | None
    matched_excess: Interval
    linear_excess: Interval

    @property
    def excess(self) -> float:
        """How much lower the model's error is than the matched baseline's."""
        return self.matched_error - self.model_error

    @property
    def verdict(self) -> Verdict:
        return self.matched_excess.verdict

    @property
    def trivial(self) -> bool:
        return self.verdict is not Verdict.LEARNT

    def floor_multiple(self, error: float) -> float | None:
        """``error`` as a multiple of the noise floor; ``None`` without a floor."""
        if self.noise_floor is None or self.noise_floor <= 0.0:
            return None
        return error / self.noise_floor
