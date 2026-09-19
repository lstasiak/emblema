from dataclasses import dataclass

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.exceptions import InvalidTokenisationManifestError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.archived_corpus import ArchivedCorpus
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.tokenisation.unit_split import UnitSplit
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum


@dataclass(frozen=True)
class TokenisationManifest:
    """Everything needed to know what a published corpus is, without reading the corpus.

    The manifest is the provenance of the block: every input that decided its bytes is here, so
    running the pipeline again either yields the same block or names what differed. It names the
    block by checksum, so the pair cannot drift apart.

    Invariants: the scheme registers the corpus's channels; no unit is both indexed by the block
    and said to be empty; the units of the split are exactly the units of the corpus.

    Attributes:
        corpus: Name of the corpus, as the vocabulary of the scheme knows it.
        corpus_version: Version of the corpus the windows were cut from.
        corpus_checksum: Checksum of the source data of that version.
        archived: What writing the windows settled: the block, its unit order and its counts.
        window: Specification the windows were laid with.
        scheme: Vocabulary and fitted statistics the tokens were produced under.
        split: Which units fitted the scheme and which were held out.
        split_seed: Seed the split was drawn with; ``None`` where its units were named or a part
            of the corpus held out.
        empty_units: Units that yielded no window at all, a fact about the corpus that the block
            itself cannot state because it holds nothing for them.
    """

    corpus: str
    corpus_version: CorpusVersionId
    corpus_checksum: Checksum
    archived: ArchivedCorpus
    window: WindowSpec
    scheme: TokenisationScheme
    split: UnitSplit
    split_seed: int | None
    empty_units: tuple[UnitKey, ...] = ()

    def __post_init__(self) -> None:
        if not self.corpus or self.corpus != self.corpus.strip():
            raise InvalidTokenisationManifestError(
                "corpus name must be non-blank without surrounding whitespace"
            )
        if not self.scheme.vocabulary.entries_of(self.corpus):
            raise InvalidTokenisationManifestError(
                f"the scheme registers no channel of corpus {self.corpus!r}"
            )
        indexed = set(self.units)
        both = sorted(str(key) for key in indexed & set(self.empty_units))
        if both:
            raise InvalidTokenisationManifestError(
                f"units both indexed by the block and said to yield nothing: {both}"
            )
        if self.split.training | self.split.validation != indexed | set(self.empty_units):
            raise InvalidTokenisationManifestError(
                "the units of the split and the units of the corpus must be the same units"
            )

    @property
    def block(self) -> ArtifactRef:
        return self.archived.block

    @property
    def units(self) -> tuple[UnitKey, ...]:
        """Unit keys in the order the block's unit column indexes them."""
        return self.archived.units

    @property
    def window_count(self) -> int:
        return self.archived.window_count

    @property
    def token_count(self) -> int:
        return self.archived.token_count
