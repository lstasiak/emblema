import pytest

from emblema.evaluation.domain.exceptions import InvalidPairedUnitErrorsError
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors


def error(unit: str, squared: float, windows: int = 2) -> UnitError:
    return UnitError(unit=UnitKey(unit), squared_error=squared, windows=windows)


def paired() -> PairedUnitErrors:
    return PairedUnitErrors(
        control=(error("a", 32.0), error("b", 18.0)),
        candidate=(error("a", 8.0), error("b", 2.0)),
    )


def test_each_side_is_scored_by_adding_sums_and_counts_before_the_root() -> None:
    compared = paired()

    assert compared.rmse_control == pytest.approx((50.0 / 4) ** 0.5)
    assert compared.rmse_candidate == pytest.approx((10.0 / 4) ** 0.5)
    assert compared.reduction == pytest.approx(compared.rmse_control - compared.rmse_candidate)
    assert compared.relative_reduction == pytest.approx(1.0 - (10.0 / 50.0) ** 0.5)
    assert [str(unit) for unit in compared.units] == ["a", "b"]


def test_a_resample_counts_a_unit_as_often_as_it_is_picked() -> None:
    compared = paired()

    assert compared.reduction_over([0, 0]) == pytest.approx((32.0 / 2) ** 0.5 - (8.0 / 2) ** 0.5)
    assert compared.reduction_over([1, 1]) == pytest.approx((18.0 / 2) ** 0.5 - (2.0 / 2) ** 0.5)
    assert compared.reduction_over([0, 1]) == pytest.approx(compared.reduction)


def test_repeats_pool_per_unit_before_the_pair_is_made() -> None:
    pooled = PairedUnitErrors.pooled(
        control=[(error("a", 4.0), error("b", 9.0)), (error("a", 12.0), error("b", 1.0))],
        candidate=[(error("a", 1.0), error("b", 1.0)), (error("a", 1.0), error("b", 1.0))],
    )

    assert pooled.control == (error("a", 16.0, 4), error("b", 10.0, 4))
    assert pooled.candidate == (error("a", 2.0, 4), error("b", 2.0, 4))


def test_repeats_that_disagree_about_the_units_do_not_pool() -> None:
    with pytest.raises(InvalidPairedUnitErrorsError, match="disagree about the units"):
        PairedUnitErrors.pooled(
            control=[(error("a", 4.0),), (error("b", 4.0),)], candidate=[(error("a", 1.0),)]
        )


def test_a_side_without_a_repeat_does_not_pool() -> None:
    with pytest.raises(InvalidPairedUnitErrorsError, match="candidate side has no repeat"):
        PairedUnitErrors.pooled(control=[(error("a", 4.0),)], candidate=[])


@pytest.mark.parametrize(
    ("control", "candidate", "message"),
    [
        ((), (), "at least one unit"),
        ((error("a", 1.0),), (error("a", 1.0), error("b", 1.0)), "1 control units paired with 2"),
        ((error("a", 1.0),), (error("b", 1.0),), "control unit a paired with candidate unit b"),
        ((error("a", 1.0, 2),), (error("a", 1.0, 3),), "2 control windows and 3 candidate"),
        (
            (error("a", 1.0), error("a", 1.0)),
            (error("a", 1.0), error("a", 1.0)),
            "paired twice",
        ),
    ],
)
def test_a_pair_over_different_units_or_windows_is_refused(
    control: tuple[UnitError, ...], candidate: tuple[UnitError, ...], message: str
) -> None:
    with pytest.raises(InvalidPairedUnitErrorsError, match=message):
        PairedUnitErrors(control=control, candidate=candidate)


def test_a_control_without_error_has_no_share_to_reduce() -> None:
    compared = PairedUnitErrors(control=(error("a", 0.0),), candidate=(error("a", 0.0),))

    assert compared.reduction == 0.0
    with pytest.raises(InvalidPairedUnitErrorsError, match="no error to reduce"):
        _ = compared.relative_reduction
