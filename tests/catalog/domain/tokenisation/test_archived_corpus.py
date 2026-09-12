import pytest

from emblema.catalog.domain.exceptions import InvalidArchivedCorpusError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.tokenisation.archived_corpus import ArchivedCorpus
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum

BLOCK = ArtifactRef("durable/sha256/" + "0" * 64, Checksum.of_bytes(b"block"))
FIRST = UnitKey("u1")
SECOND = UnitKey("u2")


def test_a_corpus_that_yielded_nothing_is_archived_as_a_block_of_nothing() -> None:
    archived = ArchivedCorpus(BLOCK, (), 0, 0)

    assert (archived.units, archived.window_count, archived.token_count) == ((), 0, 0)


def test_unit_keys_must_be_unique() -> None:
    with pytest.raises(InvalidArchivedCorpusError, match="unique"):
        ArchivedCorpus(BLOCK, (FIRST, FIRST), 2, 8)


@pytest.mark.parametrize(("windows", "tokens"), [(-1, 8), (2, -1)])
def test_counts_cannot_be_negative(windows: int, tokens: int) -> None:
    with pytest.raises(InvalidArchivedCorpusError, match="negative"):
        ArchivedCorpus(BLOCK, (FIRST,), windows, tokens)


@pytest.mark.parametrize(("windows", "tokens"), [(2, 0), (0, 8)])
def test_windows_and_tokens_come_together_or_not_at_all(windows: int, tokens: int) -> None:
    with pytest.raises(InvalidArchivedCorpusError, match="together"):
        ArchivedCorpus(BLOCK, (FIRST,), windows, tokens)


@pytest.mark.parametrize(
    ("units", "windows", "tokens"),
    [((), 2, 8), ((FIRST, SECOND), 0, 0)],
    ids=["windows of no unit", "units without a window"],
)
def test_windows_belong_to_units_and_indexed_units_have_windows(
    units: tuple[UnitKey, ...], windows: int, tokens: int
) -> None:
    with pytest.raises(InvalidArchivedCorpusError, match="of units"):
        ArchivedCorpus(BLOCK, units, windows, tokens)
