from dataclasses import dataclass, replace
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


class HeadStart(StrEnum):
    """Where the head stands when the run takes its first step.

    Attributes:
        FRESH: Drawn under the run's seed, its bias at the log-odds or mean of the sample, as
            every campaign ran before this was a knob.
        SOLVED: Solved in closed form over the states of the encoder as the run receives it,
            frozen, as the probe solved in closed form is; the steps then begin from a head
            that already answers as well as those states allow, so the first gradients that
            reach the encoder are not the ones of a head still finding its bearing.
    """

    FRESH = "fresh"
    SOLVED = "solved"


class StopDivision(StrEnum):
    """How the units a stop holds out are drawn from the labelled sample.

    Attributes:
        UNITS: Ranked by a digest of the run's seed and the unit, whatever their outcome, as
            the stop first ran.
        OUTCOMES: Ranked the same way within each outcome, each holding out its own share,
            rounded up: at a small budget a share drawn without regard to outcome often holds
            none of the rarer one, and a stop scored on one outcome reads only noise.
    """

    UNITS = "units"
    OUTCOMES = "outcomes"


@dataclass(frozen=True, kw_only=True)
class TrainingRegime:
    """How a run is stopped, weighted, perturbed and started while it learns, beyond its schedule.

    The schedule says how long a run may take and at what rate; the regime says what else is
    done inside that budget. The standard regime is the one every campaign ran under before
    these were knobs: every epoch of the schedule, the last weights kept, every window weighing
    the same, every channel read, the head drawn at random. The stop, the class weight and the
    withheld channels are what published networks on irregular clinical series train under and
    this project's did not (ADR-0047); the patience in steps, the division by outcome and the
    solved head are what learning from a few dozen labels asks of the same parts (ADR-0050).

    A stop holds out a share of the labelled sample, by unit, and keeps the weights of the epoch
    that scored best on it, giving up after ``patience`` epochs without a better one, or after
    ``patience_steps`` optimiser steps counted from the end of the warmup: at a small budget an
    epoch is a handful of steps, and a patience in epochs runs out while the rate is still
    climbing. The validation side is never read for it, so what is reported there stays what it
    was: the held-out units are taken from the labels the run was given, and the run learns from
    fewer. The schedule's epochs become a cap, so a stopped run spends at most the budget it was
    declared under, never more.

    A variant turns the stop's knobs one at a time, so a regime may hold one of them while
    another is being turned; the run stops only once a share and a patience are stated, and a
    plan refuses a stop stated in part. The head may start solved rather than drawn, which
    spends no step, so the budget is the one every other arm spends.

    Invariants: the share held out lies in ``[0, 0.5]``; both patiences are not negative, and
    at most one is stated; the channel dropout lies in ``[0, 1)``.

    Attributes:
        stop_share: Share of the labelled units held out to stop on; zero for no stop.
        patience: Epochs without a better score on the held-out units before the run stops;
            zero for a patience counted otherwise or no stop.
        patience_steps: Optimiser steps past the end of the warmup without a better score
            before the run stops; zero for a patience counted otherwise or no stop.
        stop_division: How the held-out units are drawn.
        class_weight: How the outcomes weigh in the loss.
        channel_dropout: Probability a window's channel is withheld whole while it is learnt
            from; zero for every channel read.
        head_start: Where the head stands before the first step.
    """

    stop_share: float = 0.0
    patience: int = 0
    patience_steps: int = 0
    stop_division: StopDivision = StopDivision.UNITS
    class_weight: ClassWeight = ClassWeight.NONE
    channel_dropout: float = 0.0
    head_start: HeadStart = HeadStart.FRESH

    KNOBS: ClassVar[tuple[str, ...]] = (
        "stop_share",
        "patience",
        "patience_steps",
        "stop_division",
        "class_weight",
        "channel_dropout",
        "head_start",
    )

    def __post_init__(self) -> None:
        if not isfinite(self.stop_share) or not 0.0 <= self.stop_share <= 0.5:
            raise InvalidTrainingRegimeError(
                f"stop_share must lie in [0, 0.5], got {self.stop_share}"
            )
        for label, patience in (
            ("patience", self.patience),
            ("patience_steps", self.patience_steps),
        ):
            if patience < 0:
                raise InvalidTrainingRegimeError(f"{label} must not be negative, got {patience}")
        if self.patience > 0 and self.patience_steps > 0:
            raise InvalidTrainingRegimeError(
                "a stop waits a patience in epochs or one in steps, not both"
            )
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
        return self.stop_share > 0.0 and self._waits

    @property
    def stop_partly_stated(self) -> bool:
        """Whether a stop is stated in part.

        A share or a patience without the other, or a division with no stop to divide for.
        """
        return (self.stop_share > 0.0) != self._waits or (
            self.stop_division is not StopDivision.UNITS and not self.stops
        )

    @property
    def solves_the_head_first(self) -> bool:
        """Whether the head is solved in closed form before the run's first step."""
        return self.head_start is HeadStart.SOLVED

    @property
    def _waits(self) -> bool:
        return self.patience > 0 or self.patience_steps > 0

    def tuned(self, knob: str, value: str) -> Self:
        """This regime with ``knob`` turned to ``value``.

        Raises:
            UnknownKnobError: If the regime has no such knob, or it cannot take that value.
        """
        if knob not in self.KNOBS:
            raise UnknownKnobError(f"a regime has no knob {knob!r}; it turns {self.KNOBS}")
        try:
            match knob:
                case "class_weight":
                    return replace(self, class_weight=ClassWeight(value))
                case "stop_division":
                    return replace(self, stop_division=StopDivision(value))
                case "head_start":
                    return replace(self, head_start=HeadStart(value))
        except ValueError as error:
            raise UnknownKnobError(f"{knob} cannot be {value}: {error}") from error
        return turned(self, knob, value)

    def parameters(self) -> dict[str, float | int | str]:
        """The regime flattened to scalars, in a fixed order, for whoever records a run."""
        return {
            "stop_share": self.stop_share,
            "patience": self.patience,
            "patience_steps": self.patience_steps,
            "stop_division": str(self.stop_division),
            "class_weight": str(self.class_weight),
            "channel_dropout": self.channel_dropout,
            "head_start": str(self.head_start),
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
        if self.patience_steps > 0:
            turned_knobs["patience_steps"] = self.patience_steps
        if self.stop_division is not StopDivision.UNITS:
            turned_knobs["stop_division"] = str(self.stop_division)
        if self.class_weight is not ClassWeight.NONE:
            turned_knobs["class_weight"] = str(self.class_weight)
        if self.channel_dropout > 0.0:
            turned_knobs["channel_dropout"] = self.channel_dropout
        if self.head_start is not HeadStart.FRESH:
            turned_knobs["head_start"] = str(self.head_start)
        return turned_knobs
