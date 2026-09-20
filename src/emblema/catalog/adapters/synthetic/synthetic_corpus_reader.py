from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import UnknownUnitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.registry.corpus_content import CorpusContent
from emblema.catalog.domain.registry.corpus_description import CorpusDescription
from emblema.shared.adapters.synthetic.draws import Draws
from emblema.shared.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.shared.adapters.synthetic.sensor_layout import SensorLayout
from emblema.shared.adapters.synthetic.sensor_signal import PRECISION, SensorSignal
from emblema.shared.kernel.checksums import Checksum

# The channel a unit's gain is reported under, beside the timed sensors. A fact about what this
# reader writes rather than a dial of the layout, so it is stated here and not there.
GAIN = "gain"


class SyntheticCorpusReader:
    """Generates a corpus from a sensor layout watching latent factors, and reads it back.

    The generator of the positive control (see ``layouts``), behind the same port as real data, so
    the control travels the same road: registered, frozen, tokenised, archived and trained on.
    Nothing is stored. A unit is generated on request from a seed addressed by its position rather
    than its key, which carries the layout's name, so two layouts differing in one dial share every
    draw and the null twin stays matched. Draws reproduce exactly; the arithmetic on them as far as
    ``PRECISION`` states.

    Time runs on whole steps, so two channels share an instant exactly or lie a step apart. Channels
    are named by position and keyed by corpus, so two layouts share no identifier.
    """

    def __init__(self, process: LatentFactorProcess, layout: SensorLayout) -> None:
        """Bind the reader to the factors and to the layout observing them.

        Raises:
            ValueError: If a channel is to respond to more factors than the process has.
        """
        self._layout = layout
        self._signal = SensorSignal(process, layout)

    def describe(self) -> CorpusDescription:
        """Generate the whole corpus, checksum it and count it, keeping none of it.

        The blocks are counted as they stream into the checksum, so the corpus is never held
        whole. The count is read afterwards because a checksum cannot be finished before every
        block has passed through it, which is what leaves the count finished too.
        """
        counts: list[int] = []

        def blocks() -> Iterator[bytes]:
            for index in range(self._layout.units):
                block, count = self._block_of(index)
                counts.append(count)
                yield block

        checksum = Checksum.of_chunks(blocks())
        return CorpusDescription(
            channel_schema=self._channel_schema(),
            sampling_regime=self._layout.sampling_regime,
            content=CorpusContent(
                checksum=checksum, unit_count=self._layout.units, observation_count=sum(counts)
            ),
        )

    def read_units(self) -> Iterator[CorpusUnit]:
        """Every unit of the layout, in the order its keys number them."""
        return (self._unit(index) for index in range(self._layout.units))

    def read_observations(self, unit: UnitKey) -> Iterator[Observation]:
        """Everything the layout reported for one unit, in time order and then channel order.

        Raises:
            UnknownUnitError: If the key does not name a unit of this layout.
        """
        index = self._locate(unit)
        return self._observations_of(index)

    def _observations_of(self, index: int) -> Iterator[Observation]:
        names = self._layout.channel_names
        for time, channel, value in self._readings_of(index):
            yield Observation(names[channel], float(time), float(value))

    def _readings_of(self, index: int) -> list[tuple[float, int, float]]:
        """One unit's observations as ``(time, channel, value)``, ordered by both in turn."""
        length = self._length(index)
        readings: list[tuple[float, int, float]] = []
        for channel in range(self._layout.channels):
            steps = self._steps(index, channel, length)
            times = steps.astype(np.float64) * self._layout.time_step
            values = self._values(index, channel, times)
            readings.extend(
                (time, channel, value)
                for time, value in zip(times.tolist(), values.tolist(), strict=True)
            )
        readings.sort(key=lambda reading: (reading[0], reading[1]))
        return readings

    def _values(self, index: int, channel: int, times: NDArray[np.float64]) -> NDArray[np.float64]:
        """What one channel reports at ``times``: the exact signal, plus measurement noise.

        The noise is added after the gain, as measurement noise is, and the reading is rounded
        to the precision of the corpus.
        """
        noise = Draws(self._layout.seed, "noise", index, channel).normal(len(times))
        values = self._signal.values_at(index, channel, times) + self._layout.noise * noise
        return np.round(values, PRECISION)

    def _steps(self, index: int, channel: int, length: int) -> NDArray[np.int64]:
        """The grid steps one channel reports on, inside ``[0, length)`` and strictly increasing.

        A synchronous layout draws its steps once per unit, so every channel reports together; an
        asynchronous one draws them per channel. What is dropped as missing is drawn per channel
        either way, so even a synchronous layout has gaps a window has to survive.
        """
        layout = self._layout
        drawn = Draws(layout.seed, "steps", index, "all" if layout.synchronous else channel)
        if layout.cadence == 1:
            steps = np.arange(length)
        else:
            # Gaps of at least one step, geometric about the cadence: an exponential waiting time
            # rounded down. Drawing as many gaps as there are steps is always enough to reach the
            # end of the unit, since no gap is shorter than a step.
            gaps = 1 + np.floor(-(layout.cadence - 1) * np.log(1.0 - drawn.uniform(length)))
            walked = np.cumsum(gaps.astype(np.int64)) - gaps[0].astype(np.int64)
            steps = walked[walked < length]
        kept = Draws(layout.seed, "missing", index, channel).uniform(len(steps)) >= layout.missing
        return steps[kept]

    def _length(self, index: int) -> int:
        """Steps this unit spans, drawn evenly between the shortest and the longest."""
        layout = self._layout
        span = layout.longest_unit - layout.shortest_unit + 1
        fraction = Draws(layout.seed, "length", index).uniform()
        return layout.shortest_unit + int(fraction * span)

    def _gain(self, index: int) -> float:
        return self._signal.gain(index)

    def _unit(self, index: int) -> CorpusUnit:
        return CorpusUnit(
            key=self._key(index),
            extent=TimeExtent(0.0, self._length(index) * self._layout.time_step),
            static_features=(StaticFeature(GAIN, self._gain(index)),),
        )

    def _block_of(self, index: int) -> tuple[bytes, int]:
        """One unit rendered as the lines it would occupy in a file, and its observation count.

        The corpus has no file, so the checksum needs a form of the data to stand for one. These
        lines are it: every fact the reader hands out, at the precision it hands it out, so two
        machines that generate the same corpus agree and a corpus generated differently does not.
        """
        unit, readings = self._unit(index), self._readings_of(index)
        names = self._layout.channel_names
        lines = [
            f"unit {unit.key} {unit.extent.start:.{PRECISION}f} {unit.extent.end:.{PRECISION}f}",
            f"static {GAIN} {unit.static_features[0].value:.{PRECISION}f}",
        ]
        lines.extend(
            f"timed {names[channel]} {time:.{PRECISION}f} {value:.{PRECISION}f}"
            for time, channel, value in readings
        )
        return "\n".join((*lines, "")).encode("ascii"), len(readings)

    def _channel_schema(self) -> ChannelSchema:
        timed = (Channel(name) for name in self._layout.channel_names)
        return ChannelSchema(frozenset((*timed, Channel(GAIN, timeless=True))))

    def _key(self, index: int) -> UnitKey:
        return UnitKey.within(self._layout.name, str(index))

    def _locate(self, unit: UnitKey) -> int:
        """The index of the unit ``unit`` names.

        A key is held against the one its own index would have produced, so a reshaped form of a
        real key — padded with a zero, say — names no unit rather than quietly aliasing onto it.
        """
        index = unit.name
        if index.isascii() and index.isdigit():
            number = int(index)
            if number < self._layout.units and unit == self._key(number):
                return number
        raise UnknownUnitError(f"{unit} is not a unit of layout {self._layout.name!r}")
