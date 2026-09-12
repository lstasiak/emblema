import pytest

from emblema.catalog.domain.exceptions import InvalidCorpusContentError
from emblema.catalog.domain.registry.corpus_content import CorpusContent
from emblema.shared.kernel.checksums import Checksum


@pytest.mark.parametrize(
    ("units", "observations"), [(0, 5), (1, 0)], ids=["no-unit", "no-observation"]
)
def test_content_requires_a_unit_and_an_observation(units: int, observations: int) -> None:
    with pytest.raises(InvalidCorpusContentError, match="positive"):
        CorpusContent(Checksum.of_bytes(b"x"), units, observations)


def test_content_allows_units_without_observations() -> None:
    assert CorpusContent(Checksum.of_bytes(b"x"), 3, 2).unit_count == 3
