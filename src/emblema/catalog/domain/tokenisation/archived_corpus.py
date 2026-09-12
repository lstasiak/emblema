from dataclasses import dataclass

from emblema.catalog.domain.exceptions import InvalidArchivedCorpusError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.shared.kernel.artifacts import ArtifactRef


@dataclass(frozen=True)
class ArchivedCorpus:
    """What the archive settled when it wrote the windows of a corpus.

    The archive decides how it indexes units, so the order comes back from it; the manifest
    repeats that order, which is what turns an index beside a window back into a unit's name.

    Invariants: units are unique; windows and tokens are counted together; a corpus with windows
    has units.

    Attributes:
        block: Reference to the artifact holding the windows.
        units: Unit keys in the order the block indexes them.
        window_count: How many windows the block holds.
        token_count: How many tokens the block holds in all.
    """

    block: ArtifactRef
    units: tuple[UnitKey, ...]
    window_count: int
    token_count: int

    def __post_init__(self) -> None:
        if len(set(self.units)) != len(self.units):
            raise InvalidArchivedCorpusError("unit keys must be unique")
        if self.window_count < 0 or self.token_count < 0:
            raise InvalidArchivedCorpusError("counts cannot be negative")
        if bool(self.window_count) != bool(self.token_count):
            raise InvalidArchivedCorpusError("a block holds windows and tokens together or neither")
        if bool(self.window_count) != bool(self.units):
            raise InvalidArchivedCorpusError("a block holds windows of units or nothing at all")
