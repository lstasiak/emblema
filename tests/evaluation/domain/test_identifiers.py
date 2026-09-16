import pytest

from emblema.evaluation.domain.exceptions import InvalidUnitKeyError
from emblema.evaluation.domain.identifiers import UnitKey


def test_a_unit_key_reads_as_the_name_the_corpus_gave_it() -> None:
    assert str(UnitKey("FD001/39")) == "FD001/39"


@pytest.mark.parametrize("value", ["", "   ", " FD001/39", "FD001/39\n"])
def test_a_blank_or_padded_unit_key_is_refused(value: str) -> None:
    with pytest.raises(InvalidUnitKeyError, match="non-blank"):
        UnitKey(value)
