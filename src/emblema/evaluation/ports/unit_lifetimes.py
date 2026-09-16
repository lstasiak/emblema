from collections.abc import Collection, Mapping
from typing import Protocol

from emblema.evaluation.domain.identifiers import UnitKey


class UnitLifetimes(Protocol):
    """Tells when each unit failed, in the time axis the corpus measures it on.

    The Catalog publishes measurements, never answers: a corpus that carried its labels would
    hand the pretraining an axis to learn the task from without anyone asking for it. So the one
    fact a supervised task needs beyond the windows enters here, from wherever the ground truth
    of that corpus is published.
    """

    def failure_times(self, units: Collection[UnitKey]) -> Mapping[UnitKey, float]:
        """When each of those units failed.

        Raises:
            UnknownUnitLifetimeError: If the ground truth says nothing about one of the units.
        """
        ...
