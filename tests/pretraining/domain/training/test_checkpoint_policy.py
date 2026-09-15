import pytest

from emblema.pretraining.domain.exceptions import InvalidCheckpointPolicyError
from emblema.pretraining.domain.training.checkpoint_policy import CheckpointPolicy


def test_a_checkpoint_is_due_every_interval_and_not_between() -> None:
    policy = CheckpointPolicy(every_steps=3)

    assert [policy.due_at(step) for step in range(1, 8)] == [
        False,
        False,
        True,
        False,
        False,
        True,
        False,
    ]


def test_every_step_can_be_a_checkpoint() -> None:
    assert all(CheckpointPolicy(every_steps=1).due_at(step) for step in range(1, 5))


def test_an_interval_of_nothing_is_refused() -> None:
    with pytest.raises(InvalidCheckpointPolicyError, match="positive"):
        CheckpointPolicy(every_steps=0)


def test_steps_are_counted_from_one() -> None:
    with pytest.raises(InvalidCheckpointPolicyError, match="counted from one"):
        CheckpointPolicy(every_steps=2).due_at(0)
