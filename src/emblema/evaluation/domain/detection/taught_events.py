from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.exceptions import InvalidTaughtEventsError
from emblema.evaluation.domain.labels.anomaly_scheme import AnomalyScheme
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.shared.kernel.ordering import seeded_rank


@dataclass(frozen=True, kw_only=True)
class TaughtEvents:
    """The few events of the fitting side a detector is taught, and what drew them.

    The counterpart of a sample of labels for a task that spends no budget of them: an operator
    who has marked a handful of past incidents, each whole, however many windows it spans. The
    task, the seed and the count travel with the windows so the same draw repeats, and two
    detectors taught at one count under one seed are taught the same events. Events are ranked
    one by one under the seed rather than shuffled, so the draws nest: the events taught at a
    smaller count are among those taught at a larger one under the same seed, and a reading
    across counts reads one growing set.

    Invariants: at least one event asked; every window overlaps an event; the windows hold
    exactly the events asked.

    Attributes:
        task: Task the events were drawn from.
        windows: Every window of the taught events, by unit and place.
        events: How many events were asked for.
        seed: Seed the draw was made under.
    """

    task: TaskId
    windows: tuple[LabelledWindow, ...]
    events: int
    seed: int

    def __post_init__(self) -> None:
        if self.events < 1:
            raise InvalidTaughtEventsError(
                f"a detector is taught at least one event: {self.events}"
            )
        for labelled in self.windows:
            if not AnomalyScheme.is_event(labelled.target):
                raise InvalidTaughtEventsError(
                    f"a taught window overlaps an event, not {labelled.target}"
                )
        if len(self.numbers()) != self.events:
            raise InvalidTaughtEventsError(
                f"{self.events} events were asked and the windows hold {len(self.numbers())}"
            )

    @classmethod
    def drawn(cls, task: TaskId, pool: Sequence[LabelledWindow], *, events: int, seed: int) -> Self:
        """``events`` of the events the windows of ``pool`` overlap, the first under ``seed``.

        Raises:
            InvalidTaughtEventsError: If fewer than one event is asked, or the pool holds fewer
                events than asked.
        """
        if events < 1:
            raise InvalidTaughtEventsError(f"a detector is taught at least one event: {events}")
        held = sorted({int(w.target) for w in pool if AnomalyScheme.is_event(w.target)})
        if len(held) < events:
            raise InvalidTaughtEventsError(
                f"the fitting side holds {len(held)} events, fewer than the {events} asked"
            )
        chosen = set(sorted(held, key=lambda event: seeded_rank(seed, task, event))[:events])
        windows = sorted(
            (w for w in pool if int(w.target) in chosen),
            key=lambda w: (str(w.window.unit), w.window.position),
        )
        return cls(task=task, windows=tuple(windows), events=events, seed=seed)

    def numbers(self) -> tuple[int, ...]:
        """The taught events' numbers, ascending."""
        return tuple(sorted({int(w.target) for w in self.windows}))
