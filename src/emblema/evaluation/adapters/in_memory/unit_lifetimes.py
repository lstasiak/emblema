from collections.abc import Collection, Mapping

from emblema.evaluation.domain.exceptions import UnknownUnitLifetimeError
from emblema.evaluation.domain.identifiers import UnitKey


class InMemoryUnitLifetimes:
    """Failure times given outright, for a task whose ground truth is stated rather than read."""

    def __init__(self, failures: Mapping[UnitKey, float]) -> None:
        self._failures = dict(failures)

    def failure_times(self, units: Collection[UnitKey]) -> Mapping[UnitKey, float]:
        missing = sorted(str(unit) for unit in units if unit not in self._failures)
        if missing:
            raise UnknownUnitLifetimeError(f"no failure time known for units: {missing}")
        return {unit: self._failures[unit] for unit in units}
