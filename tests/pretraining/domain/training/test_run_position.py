import pytest

from emblema.pretraining.domain.exceptions import InvalidRunPositionError
from emblema.pretraining.domain.training.run_position import RunPosition


def test_a_run_starts_at_nothing() -> None:
    assert RunPosition.start() == RunPosition(epoch=0, batches=0, steps=0)


def test_a_micro_batch_that_does_not_step_advances_the_batches_alone() -> None:
    position = RunPosition.start().after_batch(stepped=False).after_batch(stepped=True)

    assert position == RunPosition(epoch=0, batches=2, steps=1)


def test_the_next_epoch_keeps_the_steps_and_forgets_the_batches() -> None:
    position = RunPosition(epoch=1, batches=7, steps=20).next_epoch()

    assert position == RunPosition(epoch=2, batches=0, steps=20)


@pytest.mark.parametrize("field", ["epoch", "batches", "steps"])
def test_a_position_before_the_start_is_refused(field: str) -> None:
    with pytest.raises(InvalidRunPositionError):
        RunPosition(**{"epoch": 0, "batches": 0, "steps": 0, field: -1})
