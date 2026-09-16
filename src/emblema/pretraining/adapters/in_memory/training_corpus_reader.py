from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.training.training_corpus import TrainingCorpus
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.exceptions import ArtifactNotFoundError


class InMemoryTrainingCorpusReader:
    """Reader over corpora handed to it whole: the fake of the port for application tests.

    A corpus is published into it as the description and the windows a manifest would yield,
    under the reference a manifest would have; nothing is decoded, so a test that wants to know
    what a caller does with a corpus does not pay for a block.
    """

    def __init__(self) -> None:
        self._published: dict[ArtifactRef, tuple[PretrainingInput, TrainingCorpus]] = {}

    def publish(
        self, manifest: ArtifactRef, described: PretrainingInput, corpus: TrainingCorpus
    ) -> None:
        """Make ``corpus`` readable under ``manifest``, described as ``described``."""
        self._published[manifest] = (described, corpus)

    def describe(self, manifest: ArtifactRef) -> PretrainingInput:
        return self._found(manifest)[0]

    def read(self, manifest: ArtifactRef) -> TrainingCorpus:
        return self._found(manifest)[1]

    def _found(self, manifest: ArtifactRef) -> tuple[PretrainingInput, TrainingCorpus]:
        try:
            return self._published[manifest]
        except KeyError:
            raise ArtifactNotFoundError(manifest.key) from None
