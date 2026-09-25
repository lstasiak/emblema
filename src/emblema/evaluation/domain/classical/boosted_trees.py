from dataclasses import dataclass, replace
from typing import ClassVar, Self

from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.exceptions import UnknownKnobError
from emblema.evaluation.domain.tuning.knob import turned


@dataclass(frozen=True, kw_only=True)
class BoostedTrees:
    """Gradient-boosted trees over a window summarised into a vector of fixed width.

    The summary decides what the method can be fitted over, so it is part of the method rather
    than of the runtime that grows the trees: a vector as wide as a corpus has channels stays in
    that corpus, and one that is the same width whatever the layout may take in others.

    Attributes:
        features: What a window is turned into before anything is fitted.
        boosting: How many trees are grown and how far each one is trusted.
    """

    features: FeatureScheme
    boosting: GradientBoostingSpec

    NAME: ClassVar[str] = "boosted_trees"
    # What a selection may turn. The threads are left out: they change the answer's last bits,
    # not how well a method fits, and choosing them by error would be choosing noise.
    KNOBS: ClassVar[tuple[str, ...]] = (
        "rounds",
        "max_depth",
        "learning_rate",
        "row_share",
        "feature_share",
        "min_leaf_weight",
        "l2_penalty",
    )

    @property
    def spans_channel_layouts(self) -> bool:
        """Whether a fit by this method may take in windows from corpora of other layouts."""
        return self.features.spans_channel_layouts

    def parameters(self) -> dict[str, str | int | float]:
        """The method flattened to scalars, in a fixed order, for whoever records a fit."""
        return {"method": self.NAME, "features": str(self.features)} | self.boosting.parameters()

    def tuned(self, knob: str, value: str) -> Self:
        """These trees with ``knob`` turned to ``value``.

        Raises:
            UnknownKnobError: If the trees have no such knob, or it cannot take that value.
        """
        if knob not in self.KNOBS:
            raise UnknownKnobError(f"{self} has no knob {knob!r}; it turns {self.KNOBS}")
        return replace(self, boosting=turned(self.boosting, knob, value))

    def __str__(self) -> str:
        return f"{self.NAME} over {self.features}"
