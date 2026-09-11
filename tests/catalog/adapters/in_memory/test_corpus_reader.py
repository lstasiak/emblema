import pytest

from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from tests.catalog.domain.support import description, grid, unit


def test_two_units_cannot_share_a_key() -> None:
    twice = [(unit("a"), grid(["pressure"], [0.0])), (unit("a"), grid(["pressure"], [1.0]))]

    with pytest.raises(ValueError, match="duplicate unit key a"):
        InMemoryCorpusReader(description(units=2, observations=2), twice)
