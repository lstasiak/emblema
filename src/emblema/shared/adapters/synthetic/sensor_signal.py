import numpy as np
from numpy.typing import NDArray

from emblema.shared.adapters.synthetic.draws import Draws
from emblema.shared.adapters.synthetic.latent_factor_process import LatentFactorProcess
from emblema.shared.adapters.synthetic.sensor_layout import SensorLayout

# Decimals a value of the corpus is emitted with, so that two machines can agree on a checksum:
# the sine that produced the value need not agree with itself to the last bit across processors
# and library builds. Rounding narrows that window rather than closing it — numpy promises
# nothing about its transcendental functions across versions, builds or vector widths — which
# the pinned checksums in the tests turn into a loud failure. Where it is not loud is the
# registry: a frozen version whose bytes moved no longer matches the reader that claims to
# generate it.
PRECISION = 6


class SensorSignal:
    """What a layout's sensors would report if they reported exactly: the signal before the noise.

    The one model of the signal behind both what the corpus reader emits and what a task's
    ground truth says about a unit: a channel follows a mixture of the shared factors, seen
    through the layout's projection, and of factors private to the layout, both scaled by the
    unit's gain. The reader adds the measurement noise, drops what the layout fails to report
    and rounds; the truth of a task reads the value here, at any instant, because the process is
    defined for every instant and not only where a sensor happened to report.

    The two signals are mixed by the root of the coupling so that their variances add to one
    whatever it is set to, and the unit's gain scales what they add up to rather than the shared
    part alone: scaling one part would have left a null corpus quieter than its coupled twin by
    the spread of the gain, and a null that is also the fainter corpus cannot settle whether
    transfer failed for want of structure or for want of signal.
    """

    def __init__(self, process: LatentFactorProcess, layout: SensorLayout) -> None:
        """Bind the signal to the factors and to the layout observing them.

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

    @property
    def layout(self) -> SensorLayout:
        return self._layout

    def gain(self, index: int) -> float:
        """How strongly the unit at ``index`` reports, against the layout's nominal one."""
        drawn = float(Draws(self._layout.seed, "gain", index).uniform())
        return round(1.0 + self._layout.gain_spread * (2.0 * drawn - 1.0), PRECISION)

    def values_at(
        self, index: int, channel: int, times: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The exact signal of ``channel`` of the unit at ``index``, at every instant of ``times``.

        Unrounded: the reader rounds what it emits, and a truth read here is rounded by whoever
        states the precision it is compared at.
        """
        layout = self._layout
        shared = (
            self._process.values_at(times, trajectory_seed=layout.trajectory_seed, unit=index)
            @ self._projection[channel]
        )
        private = self._private.values_at(times, trajectory_seed=layout.seed, unit=index)[
            :, channel
        ]
        mixture = np.sqrt(layout.coupling) * shared + np.sqrt(1.0 - layout.coupling) * private
        return np.asarray(self.gain(index) * mixture, dtype=np.float64)

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
