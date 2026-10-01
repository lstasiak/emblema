from dataclasses import dataclass, replace
from math import ceil, isfinite
from typing import ClassVar, Self

from emblema.evaluation.domain.exceptions import InvalidEncoderSettingError, UnknownKnobError


@dataclass(frozen=True, kw_only=True)
class EncoderSetting:
    """How a cell runs the encoder beyond its weights: what it drops, and what it is given.

    The arm's rather than the schedule's, because the schedule is shared by every network of a
    campaign, the patch model included, which drops by its own specification; a dropout set on
    the schedule would be one some networks read and others ignore. The standard setting is the
    one every campaign ran under before these were knobs: nothing dropped, the raw readings.

    A grid lays the readings on equal steps, as the patch model and MiniRocket see them, and
    hands the encoder one token per channel and step from the channel's first reading on,
    carrying its last reading forward. The encoder is then the same network fed the gridded
    readings, so a comparison with the standard setting measures the grid and nothing else.

    Invariants: the dropout lies in ``[0, 1)``; a grid has a positive, finite resolution.

    Attributes:
        dropout: Share of activations the encoder drops while it learns the task.
        grid_resolution: Steps of the grid per unit of the corpus's time; ``None`` for the raw
            readings.
    """

    dropout: float = 0.0
    grid_resolution: float | None = None

    KNOBS: ClassVar[tuple[str, ...]] = ("dropout", "grid_resolution")

    def __post_init__(self) -> None:
        if not isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise InvalidEncoderSettingError(f"dropout must lie in [0, 1), got {self.dropout}")
        if self.grid_resolution is not None and (
            not isfinite(self.grid_resolution) or self.grid_resolution <= 0.0
        ):
            raise InvalidEncoderSettingError(
                f"grid_resolution must be positive and finite, got {self.grid_resolution}"
            )

    @classmethod
    def standard(cls) -> Self:
        """The setting every network ran under before these were knobs."""
        return cls()

    def tuned(self, knob: str, value: str) -> Self:
        """This setting with ``knob`` turned to ``value``.

        Both knobs are read as numbers: the grid's resolution has no value to take its type
        from while the readings are raw.

        Raises:
            UnknownKnobError: If the setting has no such knob, or it cannot take that value.
        """
        if knob not in self.KNOBS:
            raise UnknownKnobError(f"an encoder has no knob {knob!r}; it turns {self.KNOBS}")
        try:
            return replace(self, **{knob: float(value)})
        except (InvalidEncoderSettingError, ValueError) as error:
            raise UnknownKnobError(f"{knob} cannot be {value}: {error}") from error

    def steps_over(self, window_length: float) -> int | None:
        """How many steps a window of ``window_length`` units of time is laid on; ``None`` raw.

        The rule the patch model and MiniRocket lay a window by, so the same resolution puts
        a reading in the same step for every method that reads a grid.
        """
        if self.grid_resolution is None:
            return None
        return ceil(window_length * self.grid_resolution)

    def parameters(self) -> dict[str, float]:
        """The setting flattened to scalars, in a fixed order, for whoever records a run.

        A resolution of zero stands for the raw readings, so every run renders the same columns.
        """
        return {
            "encoder_dropout": self.dropout,
            "grid_resolution": 0.0 if self.grid_resolution is None else self.grid_resolution,
        }

    def turned_away(self) -> dict[str, float]:
        """Only the knobs turned away from the standard setting, for a candidate's description.

        A campaign checks every cell's candidate against the description it stored, whole. A
        description that named the standard values would differ from every one stored before
        these were knobs, and refuse the cells and selections of campaigns that ran under them;
        leaving the standard values unsaid keeps those descriptions what they were.
        """
        turned: dict[str, float] = {}
        if self.dropout != 0.0:
            turned["encoder_dropout"] = self.dropout
        if self.grid_resolution is not None:
            turned["grid_resolution"] = self.grid_resolution
        return turned
