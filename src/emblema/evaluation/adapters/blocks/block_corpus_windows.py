from collections.abc import Collection

from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.corpus_sides import CorpusSides
from emblema.shared.kernel.artifacts import ArtifactRef


class BlockCorpusWindows:
    """Reads a published corpus for a task: sides off the manifest, windows out of the block.

    Only the placement of a window is read, never its tokens: a task counts and labels windows,
    and the block holds where each one sits beside the tokens it does not need. The block is
    still fetched whole, because that is how it travels.
    """

    def __init__(self, blocks: PublishedCorpusBlocks) -> None:
        self._blocks = blocks

    def describe(self, manifest: ArtifactRef) -> CorpusSides:
        published = self._blocks.manifest_of(manifest)
        return CorpusSides(
            corpus=published.corpus,
            training=frozenset(UnitKey(key) for key in published.training_units),
            validation=frozenset(UnitKey(key) for key in published.validation_units),
        )

    def windows_of(
        self, manifest: ArtifactRef, units: Collection[UnitKey]
    ) -> tuple[TaskWindow, ...]:
        published = self._blocks.manifest_of(manifest)
        named = {UnitKey(key) for key in published.units} | {
            UnitKey(key) for key in published.empty_units
        }
        unknown = sorted(str(unit) for unit in set(units) - named)
        if unknown:
            raise UnreadableTaskCorpusError(
                f"corpus {published.corpus!r} does not name units: {unknown}"
            )
        positions = {key: index for index, key in enumerate(published.units)}
        wanted = {positions[str(unit)] for unit in units if str(unit) in positions}
        block = self._blocks.block_of(published)
        by_position = {index: key for key, index in positions.items()}
        return tuple(
            TaskWindow(
                unit=UnitKey(by_position[block.unit_of(index)]),
                position=index,
                ends_at=block.extent_of(index)[1],
            )
            for index in range(len(block))
            if block.unit_of(index) in wanted
        )
