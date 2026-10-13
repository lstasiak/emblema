from math import inf, nan

import pytest

from emblema.pretraining.domain.exceptions import InvalidObjectiveTargetError
from emblema.pretraining.domain.training.objective_target import ObjectiveTarget, TargetKind

VALUE = ObjectiveTarget.value()
LATENT = ObjectiveTarget.latent(layers=4, momentum=0.9, final_momentum=1.0)


def test_the_value_target_states_no_layers_and_no_momentum() -> None:
    assert ObjectiveTarget(kind=TargetKind.VALUE) == VALUE
    assert not VALUE.is_latent
    assert (VALUE.layers, VALUE.momentum, VALUE.final_momentum) == (0, 0.0, 0.0)


@pytest.mark.parametrize(
    "knobs", [{"layers": 1}, {"momentum": 0.9}, {"final_momentum": 1.0}], ids=str
)
def test_the_value_target_is_refused_a_knob_nobody_would_read(knobs: dict[str, object]) -> None:
    with pytest.raises(InvalidObjectiveTargetError, match="value target"):
        ObjectiveTarget(kind=TargetKind.VALUE, **knobs)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_the_latent_target_trails_by_default_from_a_momentum_near_one_to_one() -> None:
    stated = ObjectiveTarget.latent(layers=4)

    assert stated.is_latent
    assert (stated.layers, stated.momentum, stated.final_momentum) == (4, 0.996, 1.0)


def test_the_latent_target_reads_at_least_one_block() -> None:
    with pytest.raises(InvalidObjectiveTargetError, match="at least one block"):
        ObjectiveTarget.latent(layers=0)


@pytest.mark.parametrize("momentum", [0.0, -0.5, 1.5, nan, inf])
def test_a_momentum_outside_the_unit_interval_is_refused(momentum: float) -> None:
    with pytest.raises(InvalidObjectiveTargetError, match="momentum"):
        ObjectiveTarget.latent(layers=1, momentum=momentum, final_momentum=1.0)
    with pytest.raises(InvalidObjectiveTargetError, match="momentum"):
        ObjectiveTarget.latent(layers=1, momentum=0.5, final_momentum=momentum)


def test_a_momentum_that_would_fall_over_the_run_is_refused() -> None:
    with pytest.raises(InvalidObjectiveTargetError, match="never falls"):
        ObjectiveTarget.latent(layers=1, momentum=0.999, final_momentum=0.99)


def test_the_momentum_rises_linearly_to_its_final_and_stays_there() -> None:
    assert LATENT.momentum_at(0, 100) == pytest.approx(0.9)
    assert LATENT.momentum_at(25, 100) == pytest.approx(0.925)
    assert LATENT.momentum_at(100, 100) == pytest.approx(1.0)
    assert LATENT.momentum_at(150, 100) == pytest.approx(1.0)


def test_a_momentum_of_one_from_the_start_holds_the_teacher_still() -> None:
    frozen = ObjectiveTarget.latent(layers=1, momentum=1.0, final_momentum=1.0)

    assert all(frozen.momentum_at(step, 10) == 1.0 for step in range(12))


def test_the_value_target_has_no_teacher_to_move() -> None:
    with pytest.raises(InvalidObjectiveTargetError, match="no teacher"):
        VALUE.momentum_at(1, 10)


def test_a_run_of_no_steps_has_no_schedule_and_no_step_precedes_the_first() -> None:
    with pytest.raises(InvalidObjectiveTargetError, match="no momentum schedule"):
        LATENT.momentum_at(0, 0)
    with pytest.raises(InvalidObjectiveTargetError, match="numbered -1"):
        LATENT.momentum_at(-1, 10)
