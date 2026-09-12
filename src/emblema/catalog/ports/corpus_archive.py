from collections.abc import Collection, Iterable, Sequence
from typing import Protocol

from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.archived_corpus import ArchivedCorpus
from emblema.catalog.domain.tokenisation.placed_window import PlacedWindow
from emblema.catalog.domain.tokenisation.tokenisation_manifest import TokenisationManifest
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.tokens import TokenWindow


class CorpusArchive(Protocol):
    """Puts a tokenised corpus away and finds it again.

    Which bytes a corpus becomes, and which store they land in, is the archive's business; the
    application hands over windows and a description and receives references. A corpus is two
    artifacts, the windows and the manifest describing them, so that a run can read the
    description without fetching the corpus. The manifest is stored in the Catalog's published
    language, so other contexts read it without this port.
    """

    def write_windows(self, windows: Iterable[PlacedWindow]) -> ArchivedCorpus:
        """Store every window of a corpus, in the order given, and report what was stored.

        A corpus with no windows is archived as an empty block: yielding nothing under a window
        is a result, not a failure. Windows come back at the precision the archive stores them
        at; a window that would not be a window at that precision is refused and nothing is
        stored.

        Raises:
            WindowNotArchivableError: If a window cannot be stored at the archive's precision
                without ceasing to be a valid window.
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
            MalformedManifestError: If the bytes are not a manifest this can read, or describe a
                manifest that breaks the Catalog's rules.
        """
        ...

    def read_windows(
        self, archived: ArchivedCorpus, units: Collection[UnitKey]
    ) -> Sequence[TokenWindow]:
        """The windows of an archived corpus for those units, in the order the block holds them.

        Units are named, never defaulted: a run that means to train must not be able to read the
        held-out units by leaving an argument out. A unit the block does not index selects
        nothing, so a side of the split can be passed as it is, empty units included.

        Raises:
            ArtifactNotFoundError: If the block is not in the archive.
            ArtifactIntegrityError: If the stored bytes do not hash to the block's checksum.
            UnreadableCorpusBlockError: If the block is not one this archive reads.
        """
        ...
