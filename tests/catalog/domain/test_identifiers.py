import pytest

from emblema.catalog.domain.exceptions import InvalidUnitKeyError
from emblema.catalog.domain.identifiers import UnitKey
from tests.catalog.domain.support import BLANK_OR_PADDED


@pytest.mark.parametrize("text", BLANK_OR_PADDED)
def test_unit_key_rejects_blank_or_padded_text(text: str) -> None:
    with pytest.raises(InvalidUnitKeyError):
        UnitKey(text)


def test_unit_key_prints_as_its_text() -> None:
    assert str(UnitKey("FD001/39")) == "FD001/39"
