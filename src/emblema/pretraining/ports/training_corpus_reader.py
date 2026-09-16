from typing import Protocol

from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.kernel.artifacts import ArtifactRef


class TrainingCorpusReader(Protocol):
    """Reads a corpus the Catalog published, given the reference to its manifest.

    Two calls at two costs, because two callers want different things: whoever registers a run
    needs what the corpus is — which version of which corpus, under which vocabulary — and that
    is the manifest, a few kilobytes; whoever trains needs the windows, which is the block behind
    it, hundreds of megabytes. A caller that only describes never pays for the block.
    """

    def describe(self, manifest: ArtifactRef) -> PretrainingInput:
        """What the manifest says the corpus is, without its windows.

        Raises:
            UnreadablePublishedCorpusError: If the artifact is not a manifest this can read.
            ArtifactNotFoundError: If the manifest is not in the store.
            ArtifactIntegrityError: If the stored manifest does not hash to its checksum.
        """
        ...

    def read(self, manifest: ArtifactRef) -> TrainingCorpus:
        """The windows the manifest describes, training and validation sides apart.

        Raises:
            UnreadablePublishedCorpusError: If the manifest or its block is not one this reads.
            ArtifactNotFoundError: If the manifest or the block is not in the store.
            ArtifactIntegrityError: If a stored artifact does not hash to its checksum.
        """
        ...
