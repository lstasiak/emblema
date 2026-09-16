from collections.abc import Collection, Iterable, Mapping

from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.shared.kernel.artifacts import ArtifactRef


class InMemoryCorpusWindows:
    """A corpus held as its division and its window placements, with no artifact behind it.

    Windows are given per unit as the moments they end, which is all a task reads of them, and are
    numbered in the order they arrive: a block numbers its windows the same way, so a task drawn
    here and a task drawn from a block address the same windows by the same positions.
    """

    def __init__(
        self,
        sides: CorpusSides,
        ends: Mapping[UnitKey, Iterable[float]],
        manifest: ArtifactRef | None = None,
    ) -> None:
        self._sides = sides
        self._manifest = manifest
        self._windows: list[TaskWindow] = []
        for unit, moments in ends.items():
            for moment in moments:
                self._windows.append(
                    TaskWindow(unit=unit, position=len(self._windows), ends_at=moment)
                )
        self._named = frozenset(ends) | sides.training | sides.validation

    def describe(self, manifest: ArtifactRef) -> CorpusSides:
        self._known(manifest)
        return self._sides

    def windows_of(
        self, manifest: ArtifactRef, units: Collection[UnitKey]
    ) -> tuple[TaskWindow, ...]:
        self._known(manifest)
        unknown = sorted(str(unit) for unit in set(units) - self._named)
        if unknown:
            raise UnreadableTaskCorpusError(
                f"corpus {self._sides.corpus!r} does not name units: {unknown}"
            )
        wanted = set(units)
        return tuple(window for window in self._windows if window.unit in wanted)

    def _known(self, manifest: ArtifactRef) -> None:
        if self._manifest is not None and manifest != self._manifest:
            raise UnreadableTaskCorpusError(f"artifact {manifest.key!r} is not this corpus")
