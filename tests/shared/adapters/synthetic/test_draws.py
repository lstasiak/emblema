import numpy as np

from emblema.shared.adapters.synthetic.draws import Draws


def test_the_same_address_gives_the_same_numbers() -> None:
    assert np.array_equal(Draws(1, "unit", 2).uniform(5), Draws(1, "unit", 2).uniform(5))


def test_another_part_of_the_address_gives_other_numbers() -> None:
    assert not np.array_equal(Draws(1, "unit", 2).uniform(5), Draws(1, "unit", 3).uniform(5))


def test_another_seed_gives_other_numbers() -> None:
    assert not np.array_equal(Draws(1, "unit", 2).uniform(5), Draws(2, "unit", 2).uniform(5))


def test_the_parts_of_an_address_cannot_run_into_each_other() -> None:
    assert not np.array_equal(Draws(1, "a", "bc").uniform(3), Draws(1, "ab", "c").uniform(3))


def test_uniform_draws_fill_the_unit_interval() -> None:
    values = Draws(1, "uniform").uniform(4096)

    assert values.min() >= 0.0
    assert values.max() < 1.0
    assert abs(float(values.mean()) - 0.5) < 0.02


def test_normal_draws_are_standard() -> None:
    values = Draws(1, "normal").normal(4096)

    assert abs(float(values.mean())) < 0.05
    assert abs(float(values.std()) - 1.0) < 0.05


def test_normal_draws_come_in_the_shape_asked_for_however_many() -> None:
    assert Draws(1, "odd").normal(5).shape == (5,)
    assert Draws(1, "grid").normal(3, 4).shape == (3, 4)
    assert Draws(1, "one").normal().shape == ()
