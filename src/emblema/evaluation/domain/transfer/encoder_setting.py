from dataclasses import dataclass, replace
from enum import StrEnum
from math import ceil, isfinite
from typing import ClassVar, Self

from emblema.evaluation.domain.exceptions import InvalidEncoderSettingError, UnknownKnobError
from emblema.evaluation.domain.transfer.encoder_shape import EncoderShape


class ValueEmbedding(StrEnum):
    """How a token's value and gap reach the encoder's width.

    Attributes:
        LINEAR: One linear map shared by every channel, as every backbone was trained with.
        NONLINEAR: A narrow hidden layer bent by a tanh before the width; only an encoder
            trained from no weights can have it.
    """

    LINEAR = "linear"
    NONLINEAR = "nonlinear"


@dataclass(frozen=True, kw_only=True)
class EncoderSetting:
    """How a cell runs the encoder beyond its weights: what it drops, what it is given, its build.

    The arm's rather than the schedule's, because the schedule is shared by every network of a
    campaign, the patch model included, which drops by its own specification; a dropout set on
    the schedule would be one some networks read and others ignore. The standard setting is the
    one every campaign ran under before these were knobs: nothing dropped, the raw readings, the
    backbone's shape and its linear value.

    A grid lays the readings on equal steps, as the patch model and MiniRocket see them, and
    hands the encoder one token per channel and step from the channel's first reading on,
    carrying its last reading forward. The encoder is then the same network fed the gridded
    readings, so a comparison with the standard setting measures the grid and nothing else.

    The value's embedding and the shape change what the encoder is built as, so they hold only
    for an encoder trained from no weights; the plan refuses them elsewhere. A variant turns the
    shape's four counts one knob at a time, so a setting may hold some of them while it is being
    turned; ``shape`` is the whole of them or nothing, and a plan takes only a whole one.

    Invariants: the dropout lies in ``[0, 1)``; a grid has a positive, finite resolution; every
    count of the shape stated is positive.

    Attributes:
        dropout: Share of activations the encoder drops while it learns the task.
        grid_resolution: Steps of the grid per unit of the corpus's time; ``None`` for the raw
            readings.
        value_embedding: How a token's value and gap reach the width.
        width: The encoder's own width; ``None`` for its backbone's.
        heads: The encoder's own heads per block; ``None`` for its backbone's.
        layers: The encoder's own number of blocks; ``None`` for its backbone's.
        feedforward_width: The encoder's own feed-forward width; ``None`` for its backbone's.
    """

    dropout: float = 0.0
    grid_resolution: float | None = None
    value_embedding: ValueEmbedding = ValueEmbedding.LINEAR
    width: int | None = None
    heads: int | None = None
    layers: int | None = None
    feedforward_width: int | None = None

    KNOBS: ClassVar[tuple[str, ...]] = (
        "dropout",
        "grid_resolution",
        "value_embedding",
        "width",
        "heads",
        "layers",
        "feedforward_width",
    )
    _SHAPE: ClassVar[tuple[str, ...]] = ("width", "heads", "layers", "feedforward_width")

    def __post_init__(self) -> None:
        if not isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise InvalidEncoderSettingError(f"dropout must lie in [0, 1), got {self.dropout}")
        if self.grid_resolution is not None and (
            not isfinite(self.grid_resolution) or self.grid_resolution <= 0.0
        ):
            raise InvalidEncoderSettingError(
                f"grid_resolution must be positive and finite, got {self.grid_resolution}"
            )
        for label in self._SHAPE:
            count = getattr(self, label)
            if count is not None and count < 1:
                raise InvalidEncoderSettingError(f"{label} must be positive, got {count}")

    @classmethod
    def standard(cls) -> Self:
        """The setting every network ran under before these were knobs."""
        return cls()

    @property
    def shape(self) -> EncoderShape | None:
        """The encoder's own shape, or ``None`` while any of its counts is the backbone's.

        Raises:
            InvalidEncoderSettingError: If the counts stated make no shape.
        """
        if self.width is None or self.heads is None or self.layers is None:
            return None
        if self.feedforward_width is None:
            return None
        return EncoderShape(
            width=self.width,
            heads=self.heads,
            layers=self.layers,
            feedforward_width=self.feedforward_width,
        )

    @property
    def shape_partly_stated(self) -> bool:
        """Whether some of the shape's counts are stated and some are left to the backbone."""
        stated = [getattr(self, label) is not None for label in self._SHAPE]
        return any(stated) and not all(stated)

    @property
    def builds_its_own_encoder(self) -> bool:
        """Whether the encoder is built otherwise than its backbone: its own shape or value."""
        return (
            any(getattr(self, label) is not None for label in self._SHAPE)
            or self.value_embedding is not ValueEmbedding.LINEAR
        )

    def tuned(self, knob: str, value: str) -> Self:
        """This setting with ``knob`` turned to ``value``.

        The dropout and the grid's resolution are read as numbers, the shape's counts as whole
        numbers and the value's embedding by its name: a knob left at ``None`` has no value to
        take its type from.

        Raises:
            UnknownKnobError: If the setting has no such knob, or it cannot take that value.
        """
        if knob not in self.KNOBS:
            raise UnknownKnobError(f"an encoder has no knob {knob!r}; it turns {self.KNOBS}")
        try:
            match knob:
                case "value_embedding":
                    return replace(self, value_embedding=ValueEmbedding(value))
                case "width":
                    return replace(self, width=int(value))
                case "heads":
                    return replace(self, heads=int(value))
                case "layers":
                    return replace(self, layers=int(value))
                case "feedforward_width":
                    return replace(self, feedforward_width=int(value))
                case "grid_resolution":
                    return replace(self, grid_resolution=float(value))
                case _:
                    return replace(self, dropout=float(value))
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

    def parameters(self) -> dict[str, float | int | str]:
        """The setting flattened to scalars, in a fixed order, for whoever records a run.

        A resolution of zero stands for the raw readings and a count of zero for the backbone's,
        so every run renders the same columns.
        """
        return {
            "encoder_dropout": self.dropout,
            "grid_resolution": 0.0 if self.grid_resolution is None else self.grid_resolution,
            "value_embedding": str(self.value_embedding),
            **{f"encoder_{label}": getattr(self, label) or 0 for label in self._SHAPE},
        }

    def turned_away(self) -> dict[str, float | int | str]:
        """Only the knobs turned away from the standard setting, for a candidate's description.

        A campaign checks every cell's candidate against the description it stored, whole. A
        description that named the standard values would differ from every one stored before
        these were knobs, and refuse the cells and selections of campaigns that ran under them;
        leaving the standard values unsaid keeps those descriptions what they were.
        """
        turned: dict[str, float | int | str] = {}
        if self.dropout != 0.0:
            turned["encoder_dropout"] = self.dropout
        if self.grid_resolution is not None:
            turned["grid_resolution"] = self.grid_resolution
        if self.value_embedding is not ValueEmbedding.LINEAR:
            turned["value_embedding"] = str(self.value_embedding)
        for label in self._SHAPE:
            count = getattr(self, label)
            if count is not None:
                turned[f"encoder_{label}"] = count
        return turned
