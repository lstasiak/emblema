from math import inf, nan

import pytest

from emblema.pretraining.domain.exceptions import InvalidObservationTermError
from emblema.pretraining.domain.training.observation_term import ObservationTerm


def test_no_term_asks_nothing() -> None:
    assert ObservationTerm() == ObservationTerm.none()
    assert not ObservationTerm.none().is_stated
    assert not ObservationTerm.none().asks_presence
    assert not ObservationTerm.none().asks_count


def test_a_stated_term_carries_its_weights() -> None:
    term = ObservationTerm.of(presence_weight=1.0, count_weight=0.5)

    assert term.is_stated
    assert term.asks_presence
    assert term.asks_count
    assert (term.presence_weight, term.count_weight) == (1.0, 0.5)


def test_either_weight_alone_states_the_term() -> None:
    assert ObservationTerm.of(presence_weight=1.0).is_stated
    assert not ObservationTerm.of(presence_weight=1.0).asks_count
    assert ObservationTerm.of(presence_weight=0.0, count_weight=0.5).is_stated
    assert not ObservationTerm.of(presence_weight=0.0, count_weight=0.5).asks_presence


@pytest.mark.parametrize("weight", [-0.5, nan, inf])
@pytest.mark.parametrize("label", ["presence_weight", "count_weight"])
def test_a_weight_that_is_negative_or_not_a_number_is_refused(label: str, weight: float) -> None:
    with pytest.raises(InvalidObservationTermError, match=label):
        ObservationTerm(**{label: weight})
