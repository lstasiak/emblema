from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.domain.exceptions import InvalidTaughtEventsError
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow


@dataclass(frozen=True)
class TaughtRows:
    """Where the taught windows sit among the fitting rows, and which event each is of.

    A detector over features sees rows, one per fitting window, and the events it is taught
    arrive as windows; this is the join, made once by the runtime that laid the rows out. Every
    event weighs one, spread over its windows, as the register's measure weighs it, so a long
    event among the taught does not stand for them all.

    Invariants: at least one row; every position once; every event numbered from one.

    Attributes:
        positions: Index of each taught window among the fitting rows.
        events: The event each taught row overlaps, by position.
    """

    positions: tuple[int, ...]
    events: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.positions:
            raise InvalidTaughtEventsError("a detector taught events is handed their windows")
        if len(self.positions) != len(self.events):
            raise InvalidTaughtEventsError("every taught row names its event")
        if len(set(self.positions)) != len(self.positions):
            raise InvalidTaughtEventsError("a taught window is named twice")
        if any(event < 1 for event in self.events):
            raise InvalidTaughtEventsError("a taught row overlaps an event numbered from one")

    @classmethod
    def among(cls, fitting: Sequence[TaskWindow], taught: Sequence[LabelledWindow]) -> Self:
        """The taught windows found among ``fitting``, by unit and place.

        Raises:
            InvalidTaughtEventsError: If a taught window is no fitting window, or overlaps no
                event.
        """
        index = {(str(window.unit), window.position): at for at, window in enumerate(fitting)}
        positions, events = [], []
        for labelled in taught:
            place = (str(labelled.window.unit), labelled.window.position)
            if place not in index:
                raise InvalidTaughtEventsError(
                    f"the taught window of unit {place[0]} at {place[1]} is not on the fitting side"
                )
            if not AnomalyScheme.is_event(labelled.target):
                raise InvalidTaughtEventsError(
                    f"a taught window overlaps an event, not {labelled.target}"
                )
            positions.append(index[place])
            events.append(int(labelled.target))
        return cls(positions=tuple(positions), events=tuple(events))

    @property
    def event_count(self) -> int:
        return len(set(self.events))

    def weights(self) -> NDArray[np.float64]:
        """What each taught row weighs: one over its event's windows, so every event weighs one."""
        spans = {event: self.events.count(event) for event in set(self.events)}
        return np.array([1.0 / spans[event] for event in self.events], dtype=np.float64)

    def mask(self, rows: int) -> NDArray[np.bool_]:
        """Which of ``rows`` fitting rows are taught.

        Raises:
            InvalidTaughtEventsError: If a taught position lies past the rows.
        """
        if max(self.positions) >= rows:
            raise InvalidTaughtEventsError(
                f"a taught window sits at {max(self.positions)}, past the {rows} fitting rows"
            )
        taught = np.zeros(rows, dtype=np.bool_)
        taught[list(self.positions)] = True
        return taught
