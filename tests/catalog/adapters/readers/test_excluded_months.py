import pytest

from emblema.catalog.adapters.readers.excluded_months import ExcludedMonths
from emblema.catalog.domain.exceptions import CorpusDataNotFoundError
from emblema.catalog.domain.identifiers import UnitKey


def months(*names: str) -> dict[UnitKey, str]:
    """Months by their keys, each standing for whatever a reader keeps of it."""
    return {UnitKey(name): name for name in names}


def test_months_cut_by_name_are_out_of_their_parts_months_and_no_others() -> None:
    excluded = ExcludedMonths(["a/2016-02", "b/2016-01"], ("a", "b"), "turbine")

    assert excluded.kept("a", months("a/2016-01", "a/2016-02", "a/2016-03")) == (
        "a/2016-01",
        "a/2016-03",
    )
    assert excluded.kept("b", months("b/2016-01", "b/2016-02")) == ("b/2016-02",)
    assert UnitKey("a/2016-02") in excluded
    assert UnitKey("a/2016-01") not in excluded


def test_the_statement_names_the_months_in_order_and_is_empty_for_none() -> None:
    assert ExcludedMonths([], ("a",), "turbine").statement() == b""
    assert (
        ExcludedMonths(["b/2016-01", "a/2016-02"], ("a", "b"), "turbine").statement()
        == b"excluded a/2016-02,b/2016-01\n"
    )


@pytest.mark.parametrize("name", ["2016-02", "c/2016-02", "", " a/2016-02"])
def test_a_month_not_keyed_within_a_selected_part_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match=r"excluded month must be named|selected mission"):
        ExcludedMonths([name], ("a", "b"), "mission")


def test_a_month_the_part_does_not_hold_is_refused_rather_than_missed() -> None:
    excluded = ExcludedMonths(["a/2016-07"], ("a", "b"), "turbine")

    with pytest.raises(CorpusDataNotFoundError, match=r"turbine a has no month \['a/2016-07'\]"):
        excluded.kept("a", months("a/2016-01"))
    assert excluded.kept("b", months("b/2016-01")) == ("b/2016-01",)
