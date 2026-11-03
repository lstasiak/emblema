import numpy as np
import pytest

from emblema.evaluation.adapters.detection.robust_scale import RobustScale


def test_the_centre_is_the_median_and_the_spread_the_scaled_deviation_about_it() -> None:
    scale = RobustScale.of(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))

    assert scale.centre == 3.0
    assert scale.spread == pytest.approx(1.4826 * 1.0)
    assert scale.read(np.array([3.0, 3.0 + scale.spread])) == pytest.approx([0.0, 1.0])


def test_a_few_values_far_out_move_neither_the_centre_nor_the_spread() -> None:
    plain = RobustScale.of(np.arange(100.0))
    with_far = RobustScale.of(np.concatenate((np.arange(100.0), [1e6, 1e6, 1e6])))

    assert with_far.centre == pytest.approx(plain.centre, abs=2.0)
    assert with_far.spread == pytest.approx(plain.spread, rel=0.05)


def test_a_statistic_that_varies_by_rounding_alone_is_centred_and_nothing_more() -> None:
    scale = RobustScale.of(np.array([7.0, 7.0, 7.0 + 1e-12, 7.0, 7.0 - 1e-12]))

    assert RobustScale.deviation_of(np.full(5, 7.0)) == 0.0
    assert scale.spread == 1.0
    assert scale.read(np.array([7.0, 8.0])) == pytest.approx([0.0, 1.0])
