from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from emblema.catalog.adapters.synthetic.draws import Draws
from emblema.catalog.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.catalog.adapters.synthetic.sensor_layout import SensorLayout
from emblema.catalog.domain.channels.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.exceptions import UnknownUnitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.registry.corpus_content import CorpusContent
from emblema.catalog.domain.registry.corpus_description import CorpusDescription
from emblema.shared.kernel.checksums import Checksum

# Decimals a value is emitted with, so that two machines can agree on a checksum: the sine that
# produced the value need not agree with itself to the last bit across processors and library
# builds. Rounding narrows that window rather than closing it — numpy promises nothing about its
# transcendental functions across versions, builds or vector widths — which the pinned checksums
# in the tests turn into a loud failure. Where it is not loud is the registry: a frozen version
# whose bytes moved no longer matches the reader that claims to generate it.
PRECISION = 6

# The channel a unit's gain is reported under, beside the timed sensors. A fact about what this
# reader writes rather than a dial of the layout, so it is stated here and not there.
GAIN = "gain"


class SyntheticCorpusReader:
    """Generates a corpus from a sensor layout watching latent factors, and reads it back.

    The generator behind the positive control, whose purpose ``layouts`` states. An adapter of
    the corpus reader port and nothing more special than that, which is the point: the control
    travels the same road as real data — registered, frozen, tokenised, archived and trained on —
    so a failure anywhere along it is a failure the control catches.

    Nothing is stored. A unit is generated when it is asked for, from a seed addressed by its
    position in the layout, so the corpus exists on any machine that has the specification and
    needs no download. Addressed by position and not by key, because the key carries the layout's
    name: two layouts that differ only in a dial would otherwise differ in every draw as well,
    and the null pair of the control would stop being the matched twin it is meant to be. The
    randomness is reproducible exactly; the arithmetic on top of it is reproducible as far as the
    array library's transcendental functions are, which is what ``PRECISION`` is about and what
    the checksum a version is frozen over ultimately rests on.

    Time runs on a grid of whole steps. Irregularity is which steps a channel reports on, not an
    arbitrary instant, which keeps two channels either sharing an instant exactly or a whole step
    apart — far enough for a window to be archived at the precision the block is written in.

    Channels are named by position and the vocabulary keys them by corpus as well, so two layouts
    of one control share no identifier and a model must learn each from its own data.
    """

    def __init__(self, process: LatentFactorProcess, layout: SensorLayout) -> None:
        """Bind the reader to the factors and to the layout observing them.

        Raises:
            ValueError: If a channel is to respond to more factors than the process has.
        """
        if layout.factors_per_channel > process.factors:
            raise ValueError(
                f"a channel cannot respond to {layout.factors_per_channel} of "
                f"{process.factors} factors"
            )
        self._process = process
        self._layout = layout
        self._projection = self._drawn_projection(process, layout)
        # The factors a layout has to itself: as many as it has channels, one each, and drawn
        # under its own seed so that two layouts of one control share none of them. This is what
        # a channel follows when the coupling is zero, and it is built like the shared factors so
        # that an uncoupled channel differs from a coupled one in where its signal comes from
        # rather than in how it looks.
        self._private = process.with_dials(factors=layout.channels, seed=layout.seed)

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
        length, gain = self._length(index), self._gain(index)
        readings: list[tuple[float, int, float]] = []
        for channel in range(self._layout.channels):
            steps = self._steps(index, channel, length)
            times = steps.astype(np.float64) * self._layout.time_step
            values = self._values(index, channel, times, gain)
            readings.extend(
                (time, channel, value)
                for time, value in zip(times.tolist(), values.tolist(), strict=True)
            )
        readings.sort(key=lambda reading: (reading[0], reading[1]))
        return readings

    def _values(
        self, index: int, channel: int, times: NDArray[np.float64], gain: float
    ) -> NDArray[np.float64]:
        """What one channel reports at ``times``: shared signal, private signal and noise.

        The two signals are mixed by the root of the coupling so that their variances add to one
        whatever it is set to, and the unit's gain scales what they add up to rather than the
        shared part alone. Scaling one part would have left a null corpus quieter than its
        coupled twin by the spread of the gain, and a null that is also the fainter corpus cannot
        settle whether transfer failed for want of structure or for want of signal. The noise is
        added after the gain, as measurement noise is.
        """
        layout = self._layout
        shared = (
            self._process.values_at(times, trajectory_seed=layout.trajectory_seed, unit=index)
            @ self._projection[channel]
        )
        private = self._private.values_at(times, trajectory_seed=layout.seed, unit=index)[
            :, channel
        ]
        noise = Draws(layout.seed, "noise", index, channel).normal(len(times))
        values = (
            gain * (np.sqrt(layout.coupling) * shared + np.sqrt(1.0 - layout.coupling) * private)
            + layout.noise * noise
        )
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
        """How strongly this unit's sensors report, against the layout's nominal one."""
        drawn = float(Draws(self._layout.seed, "gain", index).uniform())
        return round(1.0 + self._layout.gain_spread * (2.0 * drawn - 1.0), PRECISION)

    def _unit(self, index: int) -> CorpusUnit:
        return CorpusUnit(
            key=UnitKey(self._key(index)),
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

    def _key(self, index: int) -> str:
        return f"{self._layout.name}/{index}"

    def _locate(self, unit: UnitKey) -> int:
        name, separator, index = unit.value.partition("/")
        if (
            not separator
            or name != self._layout.name
            or not (index.isascii() and index.isdigit())
            or int(index) >= self._layout.units
        ):
            raise UnknownUnitError(f"{unit} is not a unit of layout {self._layout.name!r}")
        return int(index)

    @staticmethod
    def _drawn_projection(
        process: LatentFactorProcess, layout: SensorLayout
    ) -> NDArray[np.float64]:
        """How strongly each channel responds to each factor; most of them not at all.

        A channel responds to a few factors rather than all of them, so that no single channel
        carries the whole state and a model has to put the picture together from several. The
        weights of a channel are scaled to unit norm, so every channel's shared signal has the
        same size whichever factors it was given.
        """
        draws = Draws(layout.seed, "projection")
        chosen = np.argsort(draws.uniform(layout.channels, process.factors), axis=1)
        chosen = chosen[:, : layout.factors_per_channel]
        weights = np.zeros((layout.channels, process.factors))
        rows = np.arange(layout.channels)[:, None]
        weights[rows, chosen] = draws.normal(layout.channels, layout.factors_per_channel)
        return weights / np.linalg.norm(weights, axis=1, keepdims=True)
