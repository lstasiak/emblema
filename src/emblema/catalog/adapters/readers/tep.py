import re
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from emblema.catalog.adapters.readers.subsets import chosen_subsets
from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import (
    CorpusDataNotFoundError,
    MalformedCorpusDataError,
    UnknownUnitError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.registry.corpus_content import CorpusContent
from emblema.catalog.domain.registry.corpus_description import CorpusDescription
from emblema.shared.adapters.storage.files import chunks_of
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.sampling import SamplingRegime

_EXTENSION = ".RData"
# The three columns that place a row: the fault simulated, the run within it, the sample within
# the run; the process variables follow.
_PLACING = ("faultNumber", "simulationRun", "sample")
_MEASUREMENTS = 41
_MANIPULATED = 11
# A run's name sorts as the publisher numbers it: two digits for the fault, three for the run.
_FAULT_WIDTH = 2
_RUN_WIDTH = 3
_RUN_NAME = re.compile(rf"^fault\d{{{_FAULT_WIDTH}}}-run\d{{{_RUN_WIDTH}}}$")


class TennesseeEastmanCorpusReader:
    """Reads the training runs of Rieth et al.'s Tennessee Eastman process simulations.

    The publisher ships one R data frame per file, a row per sampling instant: the fault being
    simulated, the run's number within that fault, the sample's number within the run, then 41
    measured and 11 manipulated process variables at three-minute steps. A run is a unit and a
    variable a channel; samples are evenly spaced, so the regime is regular, and a run of ``L``
    samples spans ``[1, L + 1)``. The fault number is the label the benchmark scores detectors
    by, so it names the unit and is no channel.

    The two training files are read, each a part of the corpus; the testing files are the
    benchmark's evaluation data and stay out. Units are keyed ``<file>/fault<nn>-run<nnn>``,
    since run numbers restart per fault. The checksum covers the selected files' bytes in subset
    order. The file last read is kept, so that its runs are served from memory rather than by
    parsing half a gigabyte each.
    """

    SUBSETS: ClassVar[tuple[str, ...]] = ("TEP_FaultFree_Training", "TEP_Faulty_Training")
    MEASUREMENTS: ClassVar[tuple[Channel, ...]] = tuple(
        Channel(f"xmeas_{number}") for number in range(1, _MEASUREMENTS + 1)
    )
    MANIPULATED: ClassVar[tuple[Channel, ...]] = tuple(
        Channel(f"xmv_{number}") for number in range(1, _MANIPULATED + 1)
    )
    SCHEMA: ClassVar[ChannelSchema] = ChannelSchema(frozenset((*MEASUREMENTS, *MANIPULATED)))
    # The publisher's column order: the three that place a row, then the variables.
    COLUMNS: ClassVar[tuple[str, ...]] = (
        *_PLACING,
        *(channel.name for channel in MEASUREMENTS),
        *(channel.name for channel in MANIPULATED),
    )

    def __init__(self, root: Path, subsets: Iterable[str] = SUBSETS) -> None:
        self._root = root
        self._subsets = chosen_subsets(subsets, self.SUBSETS, "Tennessee Eastman file")
        self._loaded: _Runs | None = None

    def describe(self) -> CorpusDescription:
        units = 0
        observations = 0

        def contents() -> Iterator[bytes]:
            nonlocal units, observations
            for subset in self._subsets:
                yield from chunks_of(self._path(subset))
                runs = self._runs(subset)
                units += len(runs.bounds)
                observations += runs.observation_count

        # The checksum consumes the files a chunk at a time, so no file is ever held whole for it.
        checksum = Checksum.of_chunks(contents())
        return CorpusDescription(
            channel_schema=self.SCHEMA,
            sampling_regime=SamplingRegime.REGULAR,
            content=CorpusContent(
                checksum=checksum, unit_count=units, observation_count=observations
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        for subset in self._subsets:
            runs = self._runs(subset)
            for key, (lower, upper) in runs.bounds.items():
                yield CorpusUnit(key, TimeExtent(1.0, float(upper - lower) + 1.0))

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        subset = unit.part
        if subset is None or subset not in self._subsets or not _RUN_NAME.match(unit.name):
            raise UnknownUnitError(f"{unit} is not a run of a selected file")
        runs = self._runs(subset)
        bounds = runs.bounds.get(unit)
        if bounds is None:
            raise UnknownUnitError(f"{subset} has no run {unit.name}")
        return runs.observations_of(*bounds)

    def _path(self, subset: str) -> Path:
        path = self._root / f"{subset}{_EXTENSION}"
        if not path.is_file():
            raise CorpusDataNotFoundError(f"{path} is missing")
        return path

    def _runs(self, subset: str) -> "_Runs":
        """The runs of a file, parsed unless it is the file read last."""
        if self._loaded is None or self._loaded.subset != subset:
            self._loaded = _Runs.parse(subset, self._path(subset), self.COLUMNS)
        return self._loaded


@dataclass(frozen=True)
class _Runs:
    """One file as read: its rows sorted by fault, run and sample, and where each run's rows lie.

    Attributes:
        subset: The file the rows came from, which keys its units.
        channels: The variables, in the column order ``values`` holds them.
        samples: Sample number of every row.
        values: Every row's variables.
        bounds: Row interval ``[lower, upper)`` of each run, in the order the units are read.
    """

    subset: str
    channels: tuple[str, ...]
    samples: NDArray[np.int64]
    values: NDArray[np.float64]
    bounds: dict[UnitKey, tuple[int, int]]

    @classmethod
    def parse(cls, subset: str, path: Path, columns: tuple[str, ...]) -> "_Runs":
        """The runs the file holds, checked against the publisher's format.

        Raises:
            MalformedCorpusDataError: If the file is not one R data frame with exactly the
                publisher's columns, a value is not a finite number, a fault or run number is
                not a whole number, or a run's samples are not numbered ``1..L`` without a gap.
        """
        # Imported here: the extra that provides the R reader is needed by this reader alone, and
        # a process publishing another corpus must not need it.
        import rdata

        name = path.name
        try:
            # The library guesses at a file it does not recognise before it fails on it; the
            # guess is refused below as malformed data, so it has nothing to warn about.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                objects = rdata.read_rda(path)
        # The parser raises whatever the half-read stream tripped on, and any of it means this is
        # not the publisher's file.
        except Exception as error:
            raise MalformedCorpusDataError(f"{name}: not an R data file: {error}") from error
        if len(objects) != 1:
            raise MalformedCorpusDataError(
                f"{name}: expected a file of one object, got {sorted(objects)}"
            )
        (frame,) = objects.values()
        if tuple(getattr(frame, "columns", ())) != columns:
            raise MalformedCorpusDataError(
                f"{name}: expected a data frame of the columns {list(columns)}, got "
                f"{type(frame).__name__}"
                + (f" of {list(frame.columns)}" if hasattr(frame, "columns") else "")
            )
        table = frame.to_numpy(dtype=np.float64, na_value=np.nan)
        # The parsed frame is as large as the table and is not read again: let it go before the
        # table is checked and sorted, so the file is held once rather than twice.
        del frame, objects
        if not np.all(np.isfinite(table)):
            raise MalformedCorpusDataError(f"{name}: non-finite value")
        placing = len(_PLACING)
        faults, runs, samples = (table[:, column] for column in range(placing))
        for column, label in zip((faults, runs, samples), _PLACING, strict=True):
            if np.any(column != np.floor(column)) or np.any(column < 0):
                raise MalformedCorpusDataError(f"{name}: {label} is not a whole number")
        order = np.lexsort((samples, runs, faults))
        # The publisher writes the rows in that order already; sorting a copy of every value
        # would hold the file twice for nothing, so the rows are reordered only where they are not.
        ordered = bool(np.all(order == np.arange(len(order))))
        faults, runs, samples = (
            (column if ordered else column[order]).astype(np.int64)
            for column in (faults, runs, samples)
        )
        return cls(
            subset=subset,
            samples=samples,
            channels=columns[placing:],
            values=table[:, placing:] if ordered else table[order, placing:],
            bounds=cls._bounds(name, subset, faults, runs, samples),
        )

    @staticmethod
    def _bounds(
        name: str,
        subset: str,
        faults: NDArray[np.int64],
        runs: NDArray[np.int64],
        samples: NDArray[np.int64],
    ) -> dict[UnitKey, tuple[int, int]]:
        if len(faults) == 0:
            raise MalformedCorpusDataError(f"{name}: no rows")
        starts = np.flatnonzero(np.r_[True, (faults[1:] != faults[:-1]) | (runs[1:] != runs[:-1])])
        ends = np.r_[starts[1:], len(faults)]
        bounds: dict[UnitKey, tuple[int, int]] = {}
        for lower, upper in zip(starts.tolist(), ends.tolist(), strict=True):
            fault, run = int(faults[lower]), int(runs[lower])
            if fault >= 10**_FAULT_WIDTH or run >= 10**_RUN_WIDTH:
                raise MalformedCorpusDataError(
                    f"{name}: fault {fault}, run {run}: numbered beyond the publisher's counts"
                )
            if not np.array_equal(samples[lower:upper], np.arange(1, upper - lower + 1)):
                raise MalformedCorpusDataError(
                    f"{name}: fault {fault}, run {run}: samples are not numbered 1..L without a gap"
                )
            key = f"fault{fault:0{_FAULT_WIDTH}d}-run{run:0{_RUN_WIDTH}d}"
            bounds[UnitKey.within(subset, key)] = (lower, upper)
        return bounds

    @property
    def observation_count(self) -> int:
        return int(self.values.size)

    def observations_of(self, lower: int, upper: int) -> Iterator[Observation]:
        """The run's rows as observations, sample by sample, variables in column order."""
        for sample, row in zip(
            self.samples[lower:upper].tolist(), self.values[lower:upper].tolist(), strict=True
        ):
            time = float(sample)
            for channel, value in zip(self.channels, row, strict=True):
                yield Observation(channel, time, value)
