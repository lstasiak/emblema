from collections.abc import Collection, Iterable, Sequence
from typing import Protocol

from emblema.catalog.domain.archived_corpus import ArchivedCorpus
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.placed_window import PlacedWindow
from emblema.catalog.domain.tokenisation_manifest import TokenisationManifest
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow


class CorpusArchive(Protocol):
    """Puts a tokenised corpus away and finds it again.

    Preprocessing happens once, on a machine that holds the raw data and has no session to run
    out of; every run afterwards reads what it produced. This is the seam between those two
    halves. Which bytes a corpus becomes, and which store they land in, is the archive's own
    business: the application hands over windows and a description of them and receives
    references, which is all it can say about an artifact without knowing how one is written.

    Windows go in as a stream, because a corpus does not fit in memory, and come back addressed
    by position, because that is what a training run asks of a dataset. A corpus is written as
    two artifacts, the windows and the manifest that describes them, so that a run can read the
    description — and decide whether the corpus suits it — without fetching the corpus.
    """

    def write_windows(self, windows: Iterable[PlacedWindow]) -> ArchivedCorpus:
        """Store every window of a corpus and report what was stored.

        Windows arrive in the order they were cut, unit by unit; the archive keeps that order.
        A corpus with no windows at all is archived as an empty block rather than refused: that
        a corpus yields nothing under a given window is a result, not a failure.
        """
        ...

    def write_manifest(self, manifest: TokenisationManifest) -> ArtifactRef:
        """Store the description of an archived corpus and return the reference to it."""
        ...

    def read_manifest(self, ref: ArtifactRef) -> TokenisationManifest:
        """The manifest stored under ``ref``.

        Raises:
            ArtifactNotFoundError: If nothing is stored under the reference's key.
            ArtifactIntegrityError: If the stored bytes do not hash to the reference's checksum.
            MalformedManifestError: If the bytes are not a manifest this can read.
        """
        ...

    def read_windows(
        self, manifest: TokenisationManifest, units: Collection[UnitKey]
    ) -> Sequence[TokenWindow]:
        """The windows an archived corpus holds for those units, in the order the block holds them.

        The units are named rather than defaulted so that reading a side of a split is a decision
        the caller states: a run that means to train must not be able to read the held-out units
        by leaving an argument out.

        Raises:
            ArtifactNotFoundError: If the manifest's block is not in the archive.
            ArtifactIntegrityError: If the stored bytes do not hash to the block's checksum.
        """
        ...
