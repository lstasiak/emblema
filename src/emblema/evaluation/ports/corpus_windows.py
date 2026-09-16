from collections.abc import Collection
from typing import Protocol

from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.shared.kernel.artifacts import ArtifactRef


class CorpusWindows(Protocol):
    """Reads what a published corpus holds: how it divides its units, and where its windows sit.

    Two calls at two costs, as the corpus travels as two artifacts: defining a task needs the
    division and the names, which is the manifest; drawing labels needs where each window sits and
    when it ends, which is the block behind it. Neither call returns tokens — nothing here trains
    anything, and a task that fetched hundreds of megabytes to count its labels would be paying
    for a tensor it does not read.
    """

    def describe(self, manifest: ArtifactRef) -> CorpusSides:
        """How the published corpus divides its units.

        Raises:
            UnreadableTaskCorpusError: If the artifact is not a manifest this can read.
            ArtifactNotFoundError: If the manifest is not in the store.
            ArtifactIntegrityError: If the stored manifest does not hash to its checksum.
        """
        ...

    def windows_of(
        self, manifest: ArtifactRef, units: Collection[UnitKey]
    ) -> tuple[TaskWindow, ...]:
        """Every window the corpus cut from those units, in the order the block holds them.

        Raises:
            UnreadableTaskCorpusError: If the manifest or its block is not one this reads, or a
                unit is not one the corpus names.
            ArtifactNotFoundError: If the manifest or the block is not in the store.
            ArtifactIntegrityError: If a stored artifact does not hash to its checksum.
        """
        ...
