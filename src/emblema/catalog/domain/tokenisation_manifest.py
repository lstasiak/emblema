from dataclasses import dataclass

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.catalog.domain.exceptions import InvalidTokenisationManifestError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.unit_split import UnitSplit
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum


@dataclass(frozen=True)
class TokenisationManifest:
    """Everything needed to know what a published corpus is, without reading the corpus.

    A tokenised corpus travels as two artifacts: the windows themselves, which are large, and this,
    which is not. Splitting them is what lets a run decide whether an artifact suits it — the right
    corpus version, the right window, a vocabulary its model was trained against — by fetching a
    few kilobytes rather than the whole corpus. The manifest names the windows by checksum, so the
    pair cannot drift apart: bytes that answer to that checksum are the windows this describes and
    no others.

    The manifest is also the provenance of the artifact. Every input that decided the bytes is
    here — the corpus version and its checksum, the window, the scheme, the seed that split the
    units — so running the same pipeline again either produces the same block or names a
    difference in one of these.

    Invariants: the block is named; every unit of the split is a unit of the corpus and every unit
    of the corpus is in the split; units are unique and listed in the order the block indexes them.

    Attributes:
        corpus: Name of the corpus, as the vocabulary of the scheme knows it.
        corpus_version: Version of the corpus the windows were cut from.
        corpus_checksum: Checksum of the source data of that version.
        block: Reference to the artifact holding the windows.
        window: Specification the windows were laid with.
        scheme: Vocabulary and fitted statistics the tokens were produced under.
        units: Unit keys in the order the block's unit column indexes them.
        split: Which of those units fitted the scheme and which were held out.
        split_seed: Seed the split was drawn with.
        window_count: How many windows the block holds.
        token_count: How many tokens the block holds in all.
        empty_units: Units that yielded no window at all, a fact about the corpus that the block
            itself cannot state because it holds nothing for them.
    """

    corpus: str
    corpus_version: CorpusVersionId
    corpus_checksum: Checksum
    block: ArtifactRef
    window: WindowSpec
    scheme: TokenisationScheme
    units: tuple[UnitKey, ...]
    split: UnitSplit
    split_seed: int
    window_count: int
    token_count: int
    empty_units: tuple[UnitKey, ...] = ()

    def __post_init__(self) -> None:
        if not self.corpus or self.corpus != self.corpus.strip():
            raise InvalidTokenisationManifestError(
                "corpus name must be non-blank without surrounding whitespace"
            )
        listed = set(self.units)
        if len(listed) != len(self.units):
            raise InvalidTokenisationManifestError("unit keys must be unique")
        shared = sorted(str(key) for key in listed & set(self.empty_units))
        if shared:
            raise InvalidTokenisationManifestError(
                f"units both indexed by the block and said to yield nothing: {shared}"
            )
        if self.split.training | self.split.validation != listed | set(self.empty_units):
            raise InvalidTokenisationManifestError(
                "the units of the split and the units of the corpus must be the same units"
            )
        if self.window_count < 0 or self.token_count < 0:
            raise InvalidTokenisationManifestError("counts cannot be negative")
        if bool(self.window_count) != bool(self.token_count):
            raise InvalidTokenisationManifestError(
                "a block holds windows and tokens together or neither"
            )
