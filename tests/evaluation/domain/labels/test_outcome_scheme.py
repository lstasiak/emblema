import pytest

from emblema.evaluation.domain.exceptions import InvalidLabelSchemeError, InvalidOutcomeError
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.evaluation.domain.labels.target_kind import TargetKind

SCHEME = OutcomeScheme("In-hospital_death")


@pytest.mark.parametrize("recorded", [0.0, 1.0])
def test_an_outcome_is_its_own_target(recorded: float) -> None:
    assert SCHEME.target(recorded=recorded) == recorded


@pytest.mark.parametrize("recorded", [-1.0, 0.5, 2.0])
def test_an_outcome_recorded_as_anything_but_zero_or_one_is_refused(recorded: float) -> None:
    with pytest.raises(InvalidOutcomeError, match="In-hospital_death"):
        SCHEME.target(recorded=recorded)


def test_an_outcome_is_binary_and_learnt_as_it_is() -> None:
    assert SCHEME.kind is TargetKind.BINARY
    assert SCHEME.scale == 1.0


@pytest.mark.parametrize("outcome", ["", " death", "death "])
def test_an_outcome_without_a_clean_name_is_refused(outcome: str) -> None:
    with pytest.raises(InvalidLabelSchemeError, match="non-blank"):
        OutcomeScheme(outcome)
