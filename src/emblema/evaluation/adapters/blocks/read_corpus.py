from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from emblema.catalog.contracts.published_corpus_manifest import PublishedCorpusManifest
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.adapters.windows.window_block import WindowBlock
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow


@dataclass(frozen=True)
class ReadCorpus:
    """One published corpus a fit reads: its manifest and its block, each fetched once.

    Attributes:
        manifest: What the corpus names: its channels, units and windows.
        block: Its windows, mapped from the workspace.
    """

    manifest: PublishedCorpusManifest
    block: WindowBlock

    @classmethod
    def every(
        cls, blocks: PublishedCorpusBlocks, tasks: Iterable[DownstreamTask]
    ) -> dict[ArtifactRef, Self]:
        """Every corpus the tasks are defined over, fetched and verified once each.

        Once because the manifest of one corpus is what every side of a fit is read through, and
        fetching it per side pays for the same bytes several times over a network. A runtime
        calls this before it starts timing, because the seconds a cell reports are what the
        method cost and not what the network did.
        """
        read: dict[ArtifactRef, Self] = {}
        for task in tasks:
            if task.manifest not in read:
                published = blocks.manifest_of(task.manifest)
                read[task.manifest] = cls(published, blocks.block_of(published))
        return read

    @property
    def channels(self) -> int:
        """How many channels the corpus's vocabulary runs to."""
        return len(self.manifest.channels)

    def windows(self, placed: Sequence[TaskWindow]) -> Sequence[TokenWindow]:
        """The tokens of each window, in the order given."""
        return self.block.at([window.position for window in placed])
