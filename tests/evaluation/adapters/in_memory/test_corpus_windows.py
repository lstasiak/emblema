import pytest

from emblema.evaluation.adapters.in_memory.corpus_windows import InMemoryCorpusWindows
from emblema.evaluation.domain.exceptions import UnreadableTaskCorpusError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.evaluation.support import MANIFEST, sides, units

OTHER = ArtifactRef(key="durable/other", checksum=Checksum.of_bytes(b"other"))
PUBLISHED = sides(training=units("a"), validation=units("c"))
ENDS = {UnitKey("a"): [10.0, 20.0]}


def test_a_corpus_bound_to_a_manifest_refuses_another_one() -> None:
    windows = InMemoryCorpusWindows(PUBLISHED, ENDS, MANIFEST)

    with pytest.raises(UnreadableTaskCorpusError, match="is not this corpus"):
        windows.describe(OTHER)


def test_a_corpus_bound_to_no_manifest_answers_for_any_reference() -> None:
    windows = InMemoryCorpusWindows(PUBLISHED, ENDS)

    assert windows.describe(OTHER) == PUBLISHED
