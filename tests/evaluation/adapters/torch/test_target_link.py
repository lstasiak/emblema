"""What a network's head means under each kind of target: its loss, its start, its answer."""

from math import log

import pytest

torch = pytest.importorskip("torch")

from torch.nn.functional import binary_cross_entropy, mse_loss  # noqa: E402

from emblema.evaluation.adapters.torch.target_link import TargetLink  # noqa: E402
from emblema.evaluation.domain.labels.target_kind import TargetKind  # noqa: E402
from tests.evaluation.support import OUTCOME, SCHEME  # noqa: E402

QUANTITY = TargetLink.of(SCHEME)
OUTCOMES = TargetLink.of(OUTCOME)


def test_a_quantity_is_learnt_in_its_scale_and_answered_in_its_unit() -> None:
    raw = torch.tensor([0.2, 0.8])

    assert QUANTITY.learnt(25.0) == 25.0 / SCHEME.ceiling
    assert QUANTITY.starting_at(62.5) == 0.5
    assert torch.equal(QUANTITY.answered(raw), raw * SCHEME.ceiling)
    assert float(QUANTITY.loss(raw, torch.tensor([0.0, 1.0]))) == pytest.approx(
        float(mse_loss(raw, torch.tensor([0.0, 1.0])))
    )


def test_an_outcome_is_learnt_as_the_log_odds_and_answered_as_a_probability() -> None:
    raw = torch.tensor([-2.0, 0.0, 3.0])
    taught = torch.tensor([0.0, 1.0, 1.0])

    assert OUTCOMES.learnt(1.0) == 1.0
    assert torch.allclose(OUTCOMES.answered(raw), torch.sigmoid(raw))
    assert float(OUTCOMES.loss(raw, taught)) == pytest.approx(
        float(binary_cross_entropy(torch.sigmoid(raw), taught)), rel=1e-6
    )


def test_an_outcome_starts_at_the_log_odds_of_its_prevalence() -> None:
    assert OUTCOMES.starting_at(0.2) == pytest.approx(log(0.2 / 0.8))
    assert OUTCOMES.answered(torch.tensor(OUTCOMES.starting_at(0.2))) == pytest.approx(0.2)


@pytest.mark.parametrize("prevalence", [0.0, 1.0])
def test_a_prevalence_of_certainty_starts_at_a_finite_log_odds(prevalence: float) -> None:
    assert abs(OUTCOMES.starting_at(prevalence)) < 10.0


def test_a_link_is_named_by_its_kind_to_be_read_back() -> None:
    assert TargetLink.named(str(TargetKind.BINARY), 1.0) == OUTCOMES
    with pytest.raises(ValueError, match="count"):
        TargetLink.named("count", 1.0)
