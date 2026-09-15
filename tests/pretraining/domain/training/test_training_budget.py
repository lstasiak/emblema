import pytest

from emblema.pretraining.domain.exceptions import InvalidTrainingBudgetError, PretrainingError
from emblema.pretraining.domain.training.training_budget import TrainingBudget
from tests.support.experiments import budget


def test_the_effective_batch_is_what_the_accumulation_sums() -> None:
    assert budget(batch_size=8, accumulation_steps=4).effective_batch_size == 32


def test_a_trailing_group_of_micro_batches_still_steps() -> None:
    accumulating = budget(accumulation_steps=4)

    assert accumulating.steps_per_epoch(8) == 2
    assert accumulating.steps_per_epoch(9) == 3


def test_the_schedule_counts_warmup_and_total_in_optimiser_steps() -> None:
    schedule = budget(epochs=10, warmup_epochs=2, accumulation_steps=2).schedule(10)

    assert (schedule.warmup_steps, schedule.total_steps) == (10, 50)


def test_a_budget_without_batches_in_an_epoch_is_refused() -> None:
    with pytest.raises(InvalidTrainingBudgetError, match="must hold a batch"):
        budget().schedule(0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 0),
        ("batch_size", 0),
        ("accumulation_steps", 0),
        ("learning_rate", 0.0),
        ("learning_rate", float("inf")),
        ("warmup_epochs", -1),
        ("final_lr_fraction", 1.5),
    ],
)
def test_a_budget_no_run_could_follow_is_refused(field: str, value: float) -> None:
    with pytest.raises(InvalidTrainingBudgetError):
        budget(**{field: value})


def test_a_warmup_leaving_no_epoch_to_decay_over_is_refused() -> None:
    with pytest.raises(InvalidTrainingBudgetError, match="decay"):
        budget(epochs=3, warmup_epochs=3)


def test_an_invalid_budget_reads_as_a_value_error_at_the_edge() -> None:
    with pytest.raises(ValueError, match="epochs"):
        TrainingBudget(
            epochs=0,
            batch_size=1,
            accumulation_steps=1,
            learning_rate=1e-3,
            warmup_epochs=0,
            final_lr_fraction=0.1,
            seed=1,
        )
    assert issubclass(InvalidTrainingBudgetError, PretrainingError)
