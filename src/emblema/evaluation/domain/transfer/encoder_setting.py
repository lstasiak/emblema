from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import ceil, isfinite
from typing import ClassVar, Self

from emblema.evaluation.domain.exceptions import InvalidEncoderSettingError, UnknownKnobError
from emblema.evaluation.domain.transfer.encoder_shape import EncoderShape
from emblema.evaluation.domain.transfer.layer_reading import LayerReading


class ValueEmbedding(StrEnum):
    """How a token's features reach the encoder's width.

    Attributes:
        LINEAR: One linear map shared by every channel, as every backbone was trained with.
        NONLINEAR: A narrow hidden layer bent by a tanh before the width; only an encoder
            trained from no weights can have it.
    """

    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class TimeEncoding(StrEnum):
    """How a token's position in its window reaches the encoder's width.

    Attributes:
        FOURIER: Sines and cosines at fixed frequencies, as every backbone was trained with.
        LEARNED: The same sines and cosines with their frequencies and phases learnt from the
            fixed ones on; only an encoder trained from no weights can have it.
    """

    FOURIER = "fourier"
    LEARNED = "learned"


class CellSet(StrEnum):
    """Which channels get a row of cells where the readings are laid out as cells per bin.

    Attributes:
        WINDOW: The channels each window holds, as the raw readings hold them.
        CORPUS: Every channel the corpus observes, so a channel never measured in a window
            stands as a row of empty cells.
    """

    WINDOW = "window"
    CORPUS = "corpus"


class SummaryReading(StrEnum):
    """Whether the head also reads the window's statistics per channel, the trees' own table.

    A diagnostic of the network trained from nothing, never a candidate: the statistics hand the
    head what an encoder is meant to learn to build, so a run that reads them answers how far the
    encoder falls short of them, not how good a network it is.

    Attributes:
        NONE: The head reads the encoder's pooled states alone, as every run before this knob.
        BESIDE: The head reads the pooled states and the statistics side by side.
        ONLY: The head reads the statistics alone; the encoder is built but never run.
    """

    NONE = "none"
    BESIDE = "beside"
    ONLY = "only"


@dataclass(frozen=True, kw_only=True)
class EncoderSetting:
    """How a cell runs the encoder beyond its weights: what it drops, what it is given, its build.

    The arm's rather than the schedule's, because the schedule is shared by every network of a
    campaign, the patch model included, which drops by its own specification; a dropout set on
    the schedule would be one some networks read and others ignore. The standard setting is the
    one every campaign ran under before these were knobs: nothing dropped, the raw readings, the
    backbone's shape, its linear value and its fixed time.

    A grid lays the readings on equal steps, as the patch model and MiniRocket see them, and
    hands the encoder one token per channel and step from the channel's first reading on,
    carrying its last reading forward. The encoder is then the same network fed the gridded
    readings, so a comparison with the standard setting measures the grid and nothing else.

    Bins lay the readings out as one token per channel and span of the window, each carrying
    what the span held of that channel and whether anything was measured, in place of one token
    per reading. The cells of the window's own channels, or of every channel the corpus
    observes, by the cell set. Bins and a grid are two layouts of one window, so a setting takes
    one of them; the cell set is a knob of the bins alone.

    The value's embedding, the time's encoding, the shape and the bins change what the encoder
    is built as, so they hold only for an encoder trained from no weights; the plan refuses them
    elsewhere. A variant turns the shape's four counts one knob at a time, so a setting may hold
    some of them while it is being turned; ``shape`` is the whole of them or nothing, and a plan
    takes only a whole one.

    A clip bounds every token's value at the encoder's input, in the standard deviations the
    readings are standardised in, and leaves the gap alone. It changes what the encoder is fed,
    not what it is built as, so any encoder takes it, pretrained ones included; a published
    network fed the same tokens lost most of its lead to the readings beyond five standard
    deviations once its value embedding could pass them.

    The layer read chooses which of the encoder's layers the head reads; it changes neither the
    encoder nor what it is fed. Reading below the last layer leaves the blocks above it out of
    what a run could train, so the plan takes it only where the encoder's weights do not train.

    The summaries put the window's statistics per channel, the table the trees read, beside the
    pooled states or in their place. They take the network from nothing apart from its encoder:
    what a head makes of the statistics, and what the encoder adds to them.

    Invariants: the dropout lies in ``[0, 1)``; a grid has a positive, finite resolution; the
    bins are positive and exclude a grid and a clip, which bounds a reading's value where a cell
    carries its values otherwise; the cell set is turned only with bins; every count of the
    shape stated is positive; a clip is positive and finite.

    Attributes:
        dropout: Share of activations the encoder drops while it learns the task.
        grid_resolution: Steps of the grid per unit of the corpus's time; ``None`` for the raw
            readings.
        bins: Spans the window is cut into, one token per channel and span; ``None`` for one
            token per reading.
        cells: Which channels get a row of cells under the bins.
        value_embedding: How a token's features reach the width.
        time_encoding: How a token's position reaches the width.
        width: The encoder's own width; ``None`` for its backbone's.
        heads: The encoder's own heads per block; ``None`` for its backbone's.
        layers: The encoder's own number of blocks; ``None`` for its backbone's.
        feedforward_width: The encoder's own feed-forward width; ``None`` for its backbone's.
        value_clip: Bound on a token's value at the input, either side of zero; ``None`` for the
            readings as they are.
        layer: Which of the encoder's layers the head reads.
        summaries: Whether the head reads the window's statistics per channel too, or instead.
    """

    dropout: float = 0.0
    grid_resolution: float | None = None
    bins: int | None = None
    cells: CellSet = CellSet.WINDOW
    value_embedding: ValueEmbedding = ValueEmbedding.LINEAR
    time_encoding: TimeEncoding = TimeEncoding.FOURIER
    width: int | None = None
    heads: int | None = None
    layers: int | None = None
    feedforward_width: int | None = None
    value_clip: float | None = None
    layer: LayerReading = field(default_factory=LayerReading.last)
    summaries: SummaryReading = SummaryReading.NONE

    KNOBS: ClassVar[tuple[str, ...]] = (
        "dropout",
        "grid_resolution",
        "bins",
        "cells",
        "value_embedding",
        "time_encoding",
        "width",
        "heads",
        "layers",
        "feedforward_width",
        "value_clip",
        "layer",
        "summaries",
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
        if self.bins is not None and self.bins < 1:
            raise InvalidEncoderSettingError(f"bins must be positive, got {self.bins}")
        if self.bins is not None and self.grid_resolution is not None:
            raise InvalidEncoderSettingError(
                "bins and a grid are two layouts of one window; a setting takes one of them"
            )
        if self.cells is not CellSet.WINDOW and self.bins is None:
            raise InvalidEncoderSettingError(
                f"the cell set {self.cells} is a knob of the bins, and none are stated"
            )
        if self.bins is not None and self.value_clip is not None:
            raise InvalidEncoderSettingError(
                "a clip bounds a reading's value, and a cell carries its values otherwise"
            )
        for label in self._SHAPE:
            count = getattr(self, label)
            if count is not None and count < 1:
                raise InvalidEncoderSettingError(f"{label} must be positive, got {count}")
        if self.value_clip is not None and (
            not isfinite(self.value_clip) or self.value_clip <= 0.0
        ):
            raise InvalidEncoderSettingError(
                f"value_clip must be positive and finite, got {self.value_clip}"
            )

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
        """Whether the encoder is built otherwise than its backbone: shape, value, time or bins."""
        return (
            any(getattr(self, label) is not None for label in self._SHAPE)
            or self.value_embedding is not ValueEmbedding.LINEAR
            or self.time_encoding is not TimeEncoding.FOURIER
            or self.bins is not None
        )

    @property
    def reads_cells(self) -> bool:
        """Whether the encoder is fed one token per channel and span rather than per reading."""
        return self.bins is not None

    @property
    def reads_summaries(self) -> bool:
        """Whether the head reads the window's statistics per channel, beside or alone."""
        return self.summaries is not SummaryReading.NONE

    @property
    def reads_tokens(self) -> bool:
        """Whether the encoder is run at all: not where the head reads the statistics alone."""
        return self.summaries is not SummaryReading.ONLY

    def tuned(self, knob: str, value: str) -> Self:
        """This setting with ``knob`` turned to ``value``.

        The dropout, the grid's resolution and the clip are read as numbers, the shape's counts
        and the bins as whole numbers, the cell set, the value's embedding, the time's encoding,
        the layer read and the summaries by their names: a knob left at ``None`` has no value to
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
                case "time_encoding":
                    return replace(self, time_encoding=TimeEncoding(value))
                case "cells":
                    return replace(self, cells=CellSet(value))
                case "bins":
                    return replace(self, bins=int(value))
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
                case "value_clip":
                    return replace(self, value_clip=float(value))
                case "layer":
                    return replace(self, layer=LayerReading.of(value))
                case "summaries":
                    return replace(self, summaries=SummaryReading(value))
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

        A resolution of zero stands for the raw readings, a count of zero for the backbone's,
        no bins for one token per reading and a clip of zero for none, so every run renders the
        same columns.
        """
        return {
            "encoder_dropout": self.dropout,
            "grid_resolution": 0.0 if self.grid_resolution is None else self.grid_resolution,
            "bins": 0 if self.bins is None else self.bins,
            "cells": str(self.cells),
            "value_embedding": str(self.value_embedding),
            "time_encoding": str(self.time_encoding),
            **{f"encoder_{label}": getattr(self, label) or 0 for label in self._SHAPE},
            "value_clip": 0.0 if self.value_clip is None else self.value_clip,
            "encoder_layer": str(self.layer),
            "summaries": str(self.summaries),
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
        if self.bins is not None:
            turned["bins"] = self.bins
        if self.cells is not CellSet.WINDOW:
            turned["cells"] = str(self.cells)
        if self.value_embedding is not ValueEmbedding.LINEAR:
            turned["value_embedding"] = str(self.value_embedding)
        if self.time_encoding is not TimeEncoding.FOURIER:
            turned["time_encoding"] = str(self.time_encoding)
        for label in self._SHAPE:
            count = getattr(self, label)
            if count is not None:
                turned[f"encoder_{label}"] = count
        if self.value_clip is not None:
            turned["value_clip"] = self.value_clip
        if not self.layer.is_last:
            turned["encoder_layer"] = str(self.layer)
        if self.reads_summaries:
            turned["summaries"] = str(self.summaries)
        return turned
