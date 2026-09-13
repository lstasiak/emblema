import numpy as np
import pytest
from pydantic import ValidationError

from emblema.catalog.adapters.synthetic.latent_factor_process import LatentFactorProcess
from tests.support.synthetic import PROCESS

DENSE = np.arange(0.0, 400.0, 0.25)


def test_every_frequency_lies_inside_the_band() -> None:
    periods = 1.0 / PROCESS.frequencies()

    assert periods.shape == (PROCESS.factors, PROCESS.harmonics)
    assert periods.min() >= PROCESS.shortest_period
    assert periods.max() <= PROCESS.longest_period


def test_the_structure_is_the_seed_and_nothing_else() -> None:
    same = PROCESS.model_copy(update={"seed": PROCESS.seed})
    other = PROCESS.model_copy(update={"seed": PROCESS.seed + 1})

    assert np.array_equal(same.frequencies(), PROCESS.frequencies())
    assert not np.array_equal(other.frequencies(), PROCESS.frequencies())


def test_factors_are_scaled_to_unit_variance() -> None:
    values = PROCESS.values_at(DENSE, trajectory_seed=1, unit="a")

    assert values.shape == (len(DENSE), PROCESS.factors)
    assert np.allclose(values.std(axis=0), 1.0, atol=0.15)


def test_a_unit_sees_the_same_trajectory_every_time_it_is_asked() -> None:
    first = PROCESS.values_at(DENSE, trajectory_seed=1, unit="a")
    second = PROCESS.values_at(DENSE, trajectory_seed=1, unit="a")

    assert np.array_equal(first, second)


def test_two_units_see_different_trajectories_of_the_same_factors() -> None:
    first = PROCESS.values_at(DENSE, trajectory_seed=1, unit="a")
    second = PROCESS.values_at(DENSE, trajectory_seed=1, unit="b")

    assert not np.allclose(first, second)


def test_two_trajectory_seeds_part_the_realisations_of_one_unit() -> None:
    first = PROCESS.values_at(DENSE, trajectory_seed=1, unit="a")
    second = PROCESS.values_at(DENSE, trajectory_seed=2, unit="a")

    assert not np.allclose(first, second)


def test_a_band_needs_two_ends() -> None:
    with pytest.raises(ValidationError, match="shortest period must precede"):
        LatentFactorProcess(
            factors=2, harmonics=2, shortest_period=40.0, longest_period=40.0, seed=1
        )


def test_a_process_needs_a_factor() -> None:
    with pytest.raises(ValidationError):
        LatentFactorProcess(
            factors=0, harmonics=2, shortest_period=4.0, longest_period=40.0, seed=1
        )
