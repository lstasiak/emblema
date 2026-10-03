from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import ClassVar, Self

from emblema.evaluation.domain.exceptions import InvalidTrainingRegimeError, UnknownKnobError
from emblema.evaluation.domain.tuning.knob import turned


class ClassWeight(StrEnum):
    """How the two outcomes of a binary task weigh in the loss.

    Attributes:
        NONE: Every window weighs the same, as every campaign ran before this was a knob.
        RATIO: The positive outcome weighs the ratio of negatives to positives in the labels
            learnt from, so each class contributes alike; published networks on the
            intensive-care task weight this way.
    """

    NONE = "none"
    RATIO = "ratio"


@dataclass(frozen=True, kw_only=True)
class TrainingRegime:
    """How a run is stopped, weighted and perturbed while it learns, beyond its schedule.

    The schedule says how long a run may take and at what rate; the regime says what else is
    done inside that budget. The standard regime is the one every campaign ran under before
    these were knobs: every epoch of the schedule, the last weights kept, every window weighing
    the same, every channel read. The three parts here are the ones published networks on
    irregular clinical series train under and this project's did not, tried together because,
    turned one at a time inside such a network, they cost it little, and turned together a lot.

    A stop holds out a share of the labelled sample, by unit, and keeps the weights of the epoch
    that scored best on it, giving up after ``patience`` epochs without a better one. The
    validation side is never read for it, so what is reported there stays what it was: the held
    out units are taken from the labels the run was given, and the run learns from fewer. The
    schedule's epochs become a cap, so a stopped run spends at most the budget it was declared
    under, never more.

    A variant turns the stop's two knobs one at a time, so a regime may hold one of them while
    the other is being turned; the run stops only once both are stated, and a plan refuses a
    stop stated in part.

    Invariants: the share held out lies in ``[0, 0.5]``; the patience is not negative; the
    channel dropout lies in ``[0, 1)``.

    Attributes:
        stop_share: Share of the labelled units held out to stop on; zero for no stop.
        patience: Epochs without a better score on the held-out units before the run stops;
            zero for no stop.
        class_weight: How the outcomes weigh in the loss.
        channel_dropout: Probability a window's channel is withheld whole while it is learnt
            from; zero for every channel read.
    """

    stop_share: float = 0.0
    patience: int = 0
    class_weight: ClassWeight = ClassWeight.NONE
    channel_dropout: float = 0.0

    KNOBS: ClassVar[tuple[str, ...]] = ("stop_share", "patience", "class_weight", "channel_dropout")

    def __post_init__(self) -> None:
        if not isfinite(self.stop_share) or not 0.0 <= self.stop_share <= 0.5:
            raise InvalidTrainingRegimeError(
                f"stop_share must lie in [0, 0.5], got {self.stop_share}"
            )
        if self.patience < 0:
            raise InvalidTrainingRegimeError(f"patience must not be negative, got {self.patience}")
        if not isfinite(self.channel_dropout) or not 0.0 <= self.channel_dropout < 1.0:
            raise InvalidTrainingRegimeError(
                f"channel_dropout must lie in [0, 1), got {self.channel_dropout}"
            )

    @classmethod
    def standard(cls) -> Self:
        """The regime every network ran under before these were knobs."""
        return cls()

    @property
    def stops(self) -> bool:
        """Whether the run may end before the schedule's epochs, on the held-out units."""
        return self.stop_share > 0.0 and self.patience > 0

    @property
    def stop_partly_stated(self) -> bool:
        """Whether one of the stop's two knobs is turned and the other is not."""
        return (self.stop_share > 0.0) != (self.patience > 0)

    def tuned(self, knob: str, value: str) -> Self:
        """This regime with ``knob`` turned to ``value``.

        Raises:
            UnknownKnobError: If the regime has no such knob, or it cannot take that value.
        """
        if knob not in self.KNOBS:
            raise UnknownKnobError(f"a regime has no knob {knob!r}; it turns {self.KNOBS}")
        if knob == "class_weight":
            try:
                weight = ClassWeight(value)
            except ValueError as error:
                raise UnknownKnobError(f"class_weight cannot be {value}: {error}") from error
            return turned(self, knob, weight)
        return turned(self, knob, value)

    def parameters(self) -> dict[str, float | int | str]:
        """The regime flattened to scalars, in a fixed order, for whoever records a run."""
        return {
            "stop_share": self.stop_share,
            "patience": self.patience,
            "class_weight": str(self.class_weight),
            "channel_dropout": self.channel_dropout,
        }

    def turned_away(self) -> dict[str, float | int | str]:
        """Only the knobs turned away from the standard regime, for a candidate's description.

        Left unsaid where standard for the reason the encoder's setting leaves its standard
        values unsaid: the descriptions stored before these were knobs must stay what they were.
        """
        turned_knobs: dict[str, float | int | str] = {}
        if self.stop_share > 0.0:
            turned_knobs["stop_share"] = self.stop_share
        if self.patience > 0:
            turned_knobs["patience"] = self.patience
        if self.class_weight is not ClassWeight.NONE:
            turned_knobs["class_weight"] = str(self.class_weight)
        if self.channel_dropout > 0.0:
            turned_knobs["channel_dropout"] = self.channel_dropout
        return turned_knobs
