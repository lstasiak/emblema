import numpy as np
import pytest

from emblema.evaluation.adapters.detection.shrunk_covariance import ShrunkCovariance


def test_the_shrunk_covariance_is_scikit_learn_s() -> None:
    sklearn_covariance = pytest.importorskip("sklearn.covariance")
    rows = np.random.default_rng(0).normal(size=(30, 6)) * np.arange(1, 7)
    theirs = sklearn_covariance.LedoitWolf().fit(rows)

    ours = ShrunkCovariance.of(rows)

    np.testing.assert_allclose(ours.matrix, theirs.covariance_, rtol=1e-10, atol=1e-12)
    assert ours.shrinkage == pytest.approx(theirs.shrinkage_, rel=1e-10)


def test_a_covariance_needs_two_rows() -> None:
    with pytest.raises(ValueError, match="two rows"):
        ShrunkCovariance.of(np.ones((1, 3)))


def test_the_inverse_root_whitens() -> None:
    rows = np.random.default_rng(1).normal(size=(20000, 4)) * np.array([10.0, 3.0, 1.0, 0.3])
    whitened = (rows - rows.mean(axis=0)) @ ShrunkCovariance.of(rows).inverse_root()

    np.testing.assert_allclose(np.cov(whitened, rowvar=False), np.eye(4), atol=0.05)


def test_rows_alike_in_every_feature_still_have_a_finite_inverse_root() -> None:
    alike = np.tile([[1.0, 2.0, 3.0]], (20, 1))

    covariance = ShrunkCovariance.of(alike)

    assert covariance.shrinkage == 0.0
    assert np.all(np.isfinite(covariance.inverse_root()))
