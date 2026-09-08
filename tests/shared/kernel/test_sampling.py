from emblema.shared.kernel.sampling import SamplingRegime


def test_regimes_are_exactly_regular_and_irregular() -> None:
    assert [regime.value for regime in SamplingRegime] == ["regular", "irregular"]
