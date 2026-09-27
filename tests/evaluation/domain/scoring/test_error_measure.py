import pytest

from emblema.evaluation.domain.exceptions import MismatchedErrorMeasureError
from emblema.evaluation.domain.labels.target_kind import TargetKind
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure


def test_a_quantity_is_read_by_its_squared_error_and_an_outcome_by_its_ranking() -> None:
    assert ErrorMeasure.of(TargetKind.CONTINUOUS) is ErrorMeasure.RMSE
    assert ErrorMeasure.of(TargetKind.BINARY) is ErrorMeasure.AUROC_SHORTFALL


@pytest.mark.parametrize(
    ("measure", "kind"),
    [
        (ErrorMeasure.RMSE, TargetKind.BINARY),
        (ErrorMeasure.AUROC_SHORTFALL, TargetKind.CONTINUOUS),
    ],
)
def test_a_measure_that_does_not_apply_to_the_target_is_refused(
    measure: ErrorMeasure, kind: TargetKind
) -> None:
    with pytest.raises(MismatchedErrorMeasureError, match=f"{kind} target"):
        measure.accept(kind)


def test_a_measure_that_applies_is_accepted() -> None:
    ErrorMeasure.AUROC_SHORTFALL.accept(TargetKind.BINARY)
