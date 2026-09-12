import pytest
from hypothesis import given
from hypothesis import strategies as st

from emblema.catalog.domain.channels.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.exceptions import (
    ChannelAlreadyFittedError,
    InvalidTokenisationSchemeError,
    MissingChannelStatisticsError,
    UnknownChannelError,
)
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from emblema.shared.kernel.tokens import Token, TokenWindow
from tests.catalog.domain.support import (
    CORPUS,
    OTHER_SCHEMA,
    SCHEMA,
    STATIC_SCHEMA,
    identity_scheme,
    scheme_for,
)

STATISTICS = ChannelStatistics(3, 1.0, 2.0)
# The channels of SCHEMA in vocabulary order, which is name order.
PRESSURE, TEMPERATURE = 1, 2


def test_a_scheme_for_a_vocabulary_has_nothing_fitted() -> None:
    scheme = TokenisationScheme.for_vocabulary(ChannelVocabulary().extended_with(CORPUS, SCHEMA))

    assert scheme.statistics == (None, None)
    with pytest.raises(MissingChannelStatisticsError, match="not observed"):
        scheme.statistics_of(1)


def test_fitting_a_channel_records_its_statistics_and_leaves_the_others() -> None:
    scheme = scheme_for().with_statistics(2, STATISTICS)

    assert scheme.statistics_of(2) == STATISTICS
    assert scheme.statistics == (None, STATISTICS)


def test_a_channel_is_fitted_once() -> None:
    scheme = scheme_for().with_statistics(1, STATISTICS)

    with pytest.raises(ChannelAlreadyFittedError, match="fitted"):
        scheme.with_statistics(1, STATISTICS)


@pytest.mark.parametrize("channel_id", [0, 3])
def test_an_unknown_channel_cannot_be_fitted_or_looked_up(channel_id: int) -> None:
    scheme = scheme_for()

    with pytest.raises(UnknownChannelError):
        scheme.with_statistics(channel_id, STATISTICS)
    with pytest.raises(UnknownChannelError):
        scheme.statistics_of(channel_id)


def test_extending_the_vocabulary_keeps_fitted_channels_and_adds_unfitted_ones() -> None:
    scheme = scheme_for().with_statistics(1, STATISTICS)

    extended = scheme.extended_with("other", OTHER_SCHEMA)

    assert len(extended.vocabulary) == 3
    assert extended.statistics == (STATISTICS, None, None)


def test_statistics_must_align_with_the_vocabulary() -> None:
    vocabulary = ChannelVocabulary().extended_with(CORPUS, SCHEMA)

    with pytest.raises(InvalidTokenisationSchemeError, match="align"):
        TokenisationScheme(vocabulary, (None,))


def test_a_timed_token_reads_back_as_an_observation_in_raw_units_and_raw_time() -> None:
    scheme = scheme_for().with_statistics(TEMPERATURE, ChannelStatistics(3, mean=4.0, std=2.0))
    window = TokenWindow.of([Token(TEMPERATURE, value=1.5, time=0.25, gap=0.25)])

    reconstruction = scheme.reconstruct(window, TimeExtent(5.0, 15.0))

    assert reconstruction.observations == (Observation("temperature", time=7.5, value=7.0),)
    assert reconstruction.static_features == ()


def test_a_timeless_token_reads_back_as_a_static_feature() -> None:
    scheme = identity_scheme(STATIC_SCHEMA)
    age = scheme.vocabulary.id_of(CORPUS, "age")
    temperature = scheme.vocabulary.id_of(CORPUS, "temperature")
    window = TokenWindow.of(
        [
            Token(age, value=31.0, time=0.0, gap=0.0, timeless=True),
            Token(temperature, 2.0, 0.5, 0.5),
        ]
    )

    reconstruction = scheme.reconstruct(window, TimeExtent(0.0, 4.0))

    assert reconstruction.static_features == (StaticFeature("age", 31.0),)
    assert reconstruction.observations == (Observation("temperature", time=2.0, value=2.0),)


def test_time_comes_from_the_position_in_the_window_and_not_from_the_gap() -> None:
    scheme = identity_scheme()
    # The first token of a channel measures its gap from the start of the window, so its gap is
    # the same as its position and says nothing about the observation before it.
    window = TokenWindow.of(
        [Token(TEMPERATURE, 1.0, time=0.5, gap=0.5), Token(TEMPERATURE, 2.0, time=0.75, gap=0.25)]
    )

    reconstruction = scheme.reconstruct(window, TimeExtent(0.0, 8.0))

    times = [observation.time for observation in reconstruction.observations]

    assert times == [4.0, 6.0]


def test_a_channel_that_never_varied_reads_back_in_raw_units() -> None:
    scheme = scheme_for().with_statistics(PRESSURE, ChannelStatistics(3, mean=518.67, std=0.0))
    window = TokenWindow.of([Token(PRESSURE, value=0.01, time=0.0, gap=0.0)])

    reconstruction = scheme.reconstruct(window, TimeExtent(0.0, 1.0))

    assert reconstruction.observations[0].value == pytest.approx(518.68)


def test_a_token_of_a_channel_outside_the_vocabulary_cannot_be_read_back() -> None:
    window = TokenWindow.of([Token(3, 1.0, 0.0, 0.0)])

    with pytest.raises(UnknownChannelError):
        identity_scheme().reconstruct(window, TimeExtent(0.0, 1.0))


def test_a_token_of_an_unfitted_channel_cannot_be_read_back() -> None:
    window = TokenWindow.of([Token(TEMPERATURE, 1.0, 0.0, 0.0)])

    with pytest.raises(MissingChannelStatisticsError, match="not observed"):
        scheme_for().reconstruct(window, TimeExtent(0.0, 1.0))


reals = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
positions = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
# Either the channel never varied, or its spread is a spread of the values fitted on it; a spread
# far below them cannot come from those values and normalising against it overflows.
spreads = st.one_of(st.just(0.0), st.floats(min_value=1e-3, max_value=1e3, allow_nan=False))


@given(
    mean=reals,
    std=spreads,
    start=reals,
    length=st.floats(min_value=0.1, max_value=1e3, allow_nan=False, allow_infinity=False),
    measurements=st.lists(st.tuples(positions, reals), min_size=1, max_size=8),
)
def test_reading_a_window_back_returns_the_measurements_it_was_laid_from(
    mean: float,
    std: float,
    start: float,
    length: float,
    measurements: list[tuple[float, float]],
) -> None:
    statistics = ChannelStatistics(len(measurements), mean, std)
    scheme = scheme_for().with_statistics(TEMPERATURE, statistics)
    extent = TimeExtent(start, start + length)
    window = TokenWindow.of(
        Token(TEMPERATURE, statistics.normalise(value), position, gap=position)
        for position, value in measurements
    )

    reconstruction = scheme.reconstruct(window, extent)

    expected = sorted((start + position * length, value) for position, value in measurements)
    read_back = sorted(
        (observation.time, observation.value) for observation in reconstruction.observations
    )
    assert [number for pair in read_back for number in pair] == pytest.approx(
        [number for pair in expected for number in pair], abs=1e-6
    )
