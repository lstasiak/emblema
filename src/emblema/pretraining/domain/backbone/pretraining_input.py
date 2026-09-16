from dataclasses import dataclass

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.pretraining.domain.exceptions import InvalidPretrainingInputError
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum


@dataclass(frozen=True, kw_only=True)
class PretrainingInput:
    """One published corpus a backbone is trained on, as the Catalog published it.

    What the Catalog says about the corpus, kept beside the backbone so that its provenance reads
    without the manifest: which version of which corpus, the checksum of that version's source
    data, the manifest the run was pointed at and the checksum of the block of windows it
    describes. Two checksums because they name two things — the version is the data as it was
    obtained, the block is what the run read — and a version can be published more than once.

    Invariants: the corpus name is non-empty and carries no surrounding whitespace; the
    vocabulary holds at least one channel.

    Attributes:
        corpus: Name the corpus is registered under.
        corpus_version: Version of the corpus the windows were cut from.
        corpus_checksum: Checksum of that version's source data.
        manifest: The published manifest the run was pointed at.
        block_checksum: Checksum of the block of windows the manifest describes.
        vocabulary_size: Channels of the vocabulary the corpus was tokenised under.
    """

    corpus: str
    corpus_version: CorpusVersionId
    corpus_checksum: Checksum
    manifest: ArtifactRef
    block_checksum: Checksum
    vocabulary_size: int

    def __post_init__(self) -> None:
        if not self.corpus or self.corpus != self.corpus.strip():
            raise InvalidPretrainingInputError(
                "corpus must be non-empty without surrounding whitespace"
            )
        if self.vocabulary_size < 1:
            raise InvalidPretrainingInputError(
                f"vocabulary_size must be positive, got {self.vocabulary_size}"
            )
