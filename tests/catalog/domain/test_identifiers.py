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


def test_a_unit_key_names_the_unit_within_the_part_it_was_read_from() -> None:
    key = UnitKey.within("set-b", "149509")

    assert str(key) == "set-b/149509"
    assert key.belongs_to("set-b")
    assert not key.belongs_to("set-a")


def test_a_unit_key_does_not_belong_to_a_part_its_own_name_merely_starts_with() -> None:
    assert not UnitKey.within("set-bb", "1").belongs_to("set-b")
    assert not UnitKey("set-b").belongs_to("set-b")


@pytest.mark.parametrize("text", BLANK_OR_PADDED)
def test_a_unit_key_refuses_a_blank_or_padded_part_or_name(text: str) -> None:
    with pytest.raises(InvalidUnitKeyError):
        UnitKey.within(text, "149509")
    with pytest.raises(InvalidUnitKeyError):
        UnitKey.within("set-b", text)
