import pytest

from emblema.evaluation.domain.campaign.candidate_method import CandidateMethod, MethodParameter
from emblema.evaluation.domain.exceptions import InvalidCandidateMethodError


def test_what_a_candidate_was_set_to_is_recorded_as_the_provider_spells_it() -> None:
    method = CandidateMethod.of(transfer_mode="lora", learning_rate=1e-3)

    assert [(parameter.name, parameter.value) for parameter in method.parameters] == [
        ("learning_rate", "0.001"),
        ("transfer_mode", "lora"),
    ]


def test_two_providers_naming_the_same_knobs_in_different_orders_state_the_same_method() -> None:
    # A store that keeps its own order would otherwise read a method back as a different one.
    assert CandidateMethod.of(alpha=1, beta=2) == CandidateMethod.of(beta=2, alpha=1)


def test_a_method_naming_nothing_is_a_method_all_the_same() -> None:
    assert CandidateMethod().parameters == ()


@pytest.mark.parametrize("name", ["", " ", " rate", "rate "])
def test_a_parameter_without_a_name_of_its_own_is_refused(name: str) -> None:
    with pytest.raises(InvalidCandidateMethodError, match="named"):
        MethodParameter(name=name, value="1")


def test_a_method_stated_out_of_order_is_refused() -> None:
    # Canonicalised by the factory, not by the constructor: the value object refuses anything
    # but the one form, as every other one here does.
    with pytest.raises(InvalidCandidateMethodError, match="name order"):
        CandidateMethod(
            (
                MethodParameter(name="weight_decay", value="0.0"),
                MethodParameter(name="learning_rate", value="0.001"),
            )
        )


def test_a_parameter_named_twice_is_refused() -> None:
    with pytest.raises(InvalidCandidateMethodError, match="named twice"):
        CandidateMethod(
            (
                MethodParameter(name="learning_rate", value="0.001"),
                MethodParameter(name="learning_rate", value="0.01"),
            )
        )
