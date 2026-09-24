from dataclasses import dataclass
from math import isfinite

from emblema.evaluation.domain.exceptions import InvalidGradientBoostingSpecError


@dataclass(frozen=True, kw_only=True)
class GradientBoostingSpec:
    """How many trees a classical candidate grows, how deep, and how much each one is trusted.

    Stated as a value rather than left to whatever the library defaults to, for the reason every
    other knob in this project is stated: a baseline tuned by its defaults is a baseline nobody
    can repeat, and a comparison against it says as much about the release of a package as about
    the method. The counts are the whole of what a fit costs, so they are also what a campaign
    records when it records what this candidate was set to.

    The threads are here rather than in the environment that happens to run the fit, because
    they are part of the answer and not a setting of convenience: a histogram is summed in the
    order the work was split in, so two machines agree on what a recipe produced only if they
    agree on this. A campaign that recorded every other knob and left this one to whichever
    worker picked the cell up would record a recipe that does not identify its own result.

    Invariants: at least one round, one level of depth and one thread; the rate is positive and
    finite; the two shares lie in ``(0, 1]``; the leaf weight and the penalty are finite and not
    negative.

    Attributes:
        rounds: Trees grown, one per boosting round.
        max_depth: Levels a tree may reach.
        learning_rate: Share of each tree's correction that is kept.
        row_share: Rows drawn for each tree, as a share of the fitted windows.
        feature_share: Features offered to each tree, as a share of the vector's width.
        min_leaf_weight: Least total weight a leaf may hold, which for a squared-error fit is a
            count of windows.
        l2_penalty: Penalty on the leaf values, which keeps a leaf standing on few windows from
            taking a large one.
        threads: How many threads a fit may use, which is part of what it answers.
    """

    rounds: int
    max_depth: int
    learning_rate: float
    row_share: float
    feature_share: float
    min_leaf_weight: float
    l2_penalty: float
    threads: int

    def parameters(self) -> dict[str, int | float]:
        """The knobs flattened to scalars, in a fixed order, for whoever records a fit."""
        return {
            "rounds": self.rounds,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "row_share": self.row_share,
            "feature_share": self.feature_share,
            "min_leaf_weight": self.min_leaf_weight,
            "l2_penalty": self.l2_penalty,
            "threads": self.threads,
        }

    def __post_init__(self) -> None:
        for label, count in (
            ("rounds", self.rounds),
            ("max_depth", self.max_depth),
            ("threads", self.threads),
        ):
            if count < 1:
                raise InvalidGradientBoostingSpecError(f"{label} must be positive, got {count}")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise InvalidGradientBoostingSpecError(
                f"learning_rate must be positive and finite, got {self.learning_rate}"
            )
        for label, share in (("row_share", self.row_share), ("feature_share", self.feature_share)):
            if not isfinite(share) or not 0.0 < share <= 1.0:
                raise InvalidGradientBoostingSpecError(f"{label} must lie in (0, 1], got {share}")
        for label, value in (
            ("min_leaf_weight", self.min_leaf_weight),
            ("l2_penalty", self.l2_penalty),
        ):
            if not isfinite(value) or value < 0.0:
                raise InvalidGradientBoostingSpecError(
                    f"{label} must be finite and not negative, got {value}"
                )
