from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from emblema.evaluation.domain.exceptions import UnknownGroundTruthError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.shared.adapters.synthetic.sensor_signal import PRECISION, SensorSignal


class SyntheticGroundTruth:
    """The exact reading of one sensor of a generated corpus, at the instant a task asks about.

    The corpus is generated from its specification, so its truth is generated the same way: the
    signal a sensor would report without noise, at any instant, whether or not the sensor
    reported there. That is what the forecasting task of the synthetic control asks for, and
    what its label scheme names — which sensor, how far past a window's end — this adapter reads
    off the scheme, so the question is stated once. The reading is rounded to the precision of
    the corpus, as the sensor's own readings are.

    Units are named ``<layout>/<index>``, as the reader of these corpora names them.
    """

    def __init__(self, signal: SensorSignal, scheme: ForecastScheme) -> None:
        """Read the truth of ``scheme`` off ``signal``.

        Raises:
            UnknownGroundTruthError: If the layout has no sensor of the scheme's name.
        """
        names = signal.layout.channel_names
        if scheme.channel not in names:
            raise UnknownGroundTruthError(
                f"layout {signal.layout.name!r} has no channel {scheme.channel!r}"
            )
        self._signal = signal
        self._scheme = scheme
        self._channel = names.index(scheme.channel)

    def truths_of(self, windows: Sequence[TaskWindow]) -> Mapping[TaskWindow, float]:
        by_unit: dict[int, list[TaskWindow]] = defaultdict(list)
        for window in windows:
            by_unit[self._index_of(window.unit)].append(window)
        truths: dict[TaskWindow, float] = {}
        for index, of_unit in by_unit.items():
            instants = np.array([self._scheme.instant_of(w.ends_at) for w in of_unit])
            values = self._signal.values_at(index, self._channel, instants)
            for window, value in zip(of_unit, values.tolist(), strict=True):
                truths[window] = round(float(value), PRECISION)
        return truths

    def _index_of(self, unit: UnitKey) -> int:
        layout = self._signal.layout
        prefix, _, index = str(unit).partition("/")
        if (
            prefix == layout.name
            and index.isascii()
            and index.isdigit()
            and str(int(index)) == index
            and int(index) < layout.units
        ):
            return int(index)
        raise UnknownGroundTruthError(f"{unit} is not a unit of layout {layout.name!r}")
