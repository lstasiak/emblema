import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from emblema.evaluation.adapters.torch.early_stop import EarlyStop  # noqa: E402
from emblema.evaluation.adapters.torch.target_link import TargetLink  # noqa: E402
from emblema.evaluation.domain.labels.target_kind import TargetKind  # noqa: E402

pytestmark = pytest.mark.ml

BINARY = TargetLink(kind=TargetKind.BINARY, scale=1.0)
QUANTITY = TargetLink(kind=TargetKind.CONTINUOUS, scale=10.0)


def test_the_score_of_an_outcome_is_the_two_areas_summed_and_perfect_ranking_scores_two() -> None:
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])

    assert EarlyStop.score(BINARY, torch.tensor([0.1, 0.2, 0.8, 0.9]), targets) == pytest.approx(
        2.0
    )
    reversed_score = EarlyStop.score(BINARY, torch.tensor([0.9, 0.8, 0.2, 0.1]), targets)
    assert reversed_score == pytest.approx(0.0 + (1 / 3 + 2 / 4) / 2)
    tied = EarlyStop.score(BINARY, torch.tensor([0.5, 0.5, 0.5, 0.5]), targets)
    assert tied == pytest.approx(0.5 + (1 / 3 + 2 / 4) / 2)


def test_the_score_of_a_quantity_is_the_negative_squared_error() -> None:
    score = EarlyStop.score(QUANTITY, torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0]))

    assert score == pytest.approx(-2.0)


def test_the_best_epochs_weights_come_back_and_the_patience_ends_the_run() -> None:
    model = nn.Linear(1, 1)
    stop = EarlyStop(patience=2)
    with torch.no_grad():
        model.weight.fill_(1.0)
    assert stop.observe(0, 0.5, model) is False
    with torch.no_grad():
        model.weight.fill_(2.0)
    assert stop.observe(1, 0.9, model) is False
    with torch.no_grad():
        model.weight.fill_(3.0)
    assert stop.observe(2, 0.8, model) is False
    assert stop.observe(3, 0.7, model) is True
    assert stop.best_epoch == 1

    stop.restore(model)

    assert float(model.weight) == 2.0


def test_a_model_never_scored_is_left_alone() -> None:
    model = nn.Linear(1, 1)
    before = model.weight.detach().clone()

    EarlyStop(patience=1).restore(model)

    assert torch.equal(model.weight, before)


def test_a_patience_in_steps_is_counted_from_the_end_of_the_warmup() -> None:
    # Three steps an epoch, the warmup over at step ten, the best epoch the first: the wait
    # begins at ten, not at three, so six steps run out at the epoch ending at step eighteen.
    model = nn.Linear(1, 1)
    stop = EarlyStop(6, steps_per_epoch=3, counted_from=10)

    assert stop.observe(0, 0.9, model) is False
    assert [stop.observe(epoch, 0.1, model) for epoch in range(1, 6)] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert stop.best_epoch == 0


def test_a_patience_in_steps_after_the_warmup_counts_from_the_best_epoch() -> None:
    model = nn.Linear(1, 1)
    stop = EarlyStop(6, steps_per_epoch=3, counted_from=4)

    assert stop.observe(0, 0.1, model) is False
    assert stop.observe(1, 0.9, model) is False
    assert [stop.observe(epoch, 0.1, model) for epoch in (2, 3)] == [False, True]
