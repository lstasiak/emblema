"""Contract of the Tokeniser port, run against every adapter.

The reference the adapters are held to is a brute-force reconstruction of each window from the
observations that fall inside it, computed here without streaming, so that an adapter's
bookkeeping is checked against the definition rather than against itself.
"""

import statistics
from collections.abc import Callable, Iterable, Sequence
from itertools import chain

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from emblema.catalog.adapters.in_memory.corpus_reader import InMemoryCorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.domain.channel_schema import Channel, ChannelSchema
from emblema.catalog.domain.corpus_unit import CorpusUnit
from emblema.catalog.domain.exceptions import (
    ChannelAlreadyFittedError,
    ChannelKindMismatchError,
    MissingChannelStatisticsError,
    ObservationOutOfOrderError,
    ObservationOutsideExtentError,
    UnknownChannelError,
)
from emblema.catalog.domain.observation import Observation
from emblema.catalog.domain.placed_window import PlacedWindow
from emblema.catalog.domain.static_feature import StaticFeature
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.unit_split import UnitSplit
from emblema.catalog.domain.window_spec import WindowSpec
from emblema.catalog.ports.tokeniser import Tokeniser
from emblema.shared.kernel.tokens import Token, TokenWindow
from tests.catalog.domain.support import (
    CORPUS,
    SCHEMA,
    STATIC_SCHEMA,
    description,
    grid,
    identity_scheme,
    measured,
    occupied_extents,
    scheme_for,
    unit,
)

ADAPTERS: dict[str, Callable[[], Tokeniser]] = {"sliding_window": SlidingWindowTokeniser}
CHANNELS = ("temperature", "pressure")
AGE = StaticFeature("age", 61.0)
UNIT = unit("u1", 0.0, 10.0)
STATIC_UNIT = unit("u1", 0.0, 10.0, AGE)
REGULAR = grid(CHANNELS, [float(t) for t in range(10)])
WINDOW = WindowSpec(length=4, stride=2)


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def tokeniser(request: pytest.FixtureRequest) -> Tokeniser:
    factory: Callable[[], Tokeniser] = request.param
    return factory()


def fitted(
    tokeniser: Tokeniser,
    observations: Iterable[Observation],
    statics: Iterable[StaticFeature] = (),
    schema: ChannelSchema = SCHEMA,
) -> TokenisationScheme:
    return tokeniser.fit(CORPUS, observations, statics, scheme_for(schema))


def placed_of(
    tokeniser: Tokeniser,
    corpus_unit: CorpusUnit,
    observations: Sequence[Observation],
    scheme: TokenisationScheme,
    window: WindowSpec = WINDOW,
) -> list[PlacedWindow]:
    return list(tokeniser.tokenise(CORPUS, corpus_unit, observations, scheme, window))


def windows_of(
    tokeniser: Tokeniser,
    corpus_unit: CorpusUnit,
    observations: Sequence[Observation],
    scheme: TokenisationScheme,
    window: WindowSpec = WINDOW,
) -> list[TokenWindow]:
    placed = placed_of(tokeniser, corpus_unit, observations, scheme, window)
    return [item.window for item in placed]


def expected_windows(
    corpus_unit: CorpusUnit,
    observations: Sequence[Observation],
    scheme: TokenisationScheme,
    window: WindowSpec,
) -> list[TokenWindow]:
    """Brute force: every window that holds an observation, built from the definition of a token."""
    vocabulary = scheme.vocabulary
    result = []
    for extent in window.windows_over(corpus_unit.extent):
        inside = sorted(
            (observation for observation in observations if extent.contains(observation.time)),
            key=lambda observation: (
                observation.time,
                vocabulary.id_of(CORPUS, observation.channel),
                observation.value,
            ),
        )
        if not inside:
            continue
        tokens = [
            Token(
                vocabulary.id_of(CORPUS, feature.channel),
                scheme.statistics_of(vocabulary.id_of(CORPUS, feature.channel)).normalise(
                    feature.value
                ),
                0.0,
                0.0,
                timeless=True,
            )
            for feature in corpus_unit.static_features
        ]
        previous: dict[int, float] = {}
        for observation in inside:
            channel_id = vocabulary.id_of(CORPUS, observation.channel)
            position = (observation.time - extent.start) / extent.length
            tokens.append(
                Token(
                    channel_id,
                    scheme.statistics_of(channel_id).normalise(observation.value),
                    position,
                    position - previous.get(channel_id, 0.0),
                )
            )
            previous[channel_id] = position
        result.append(TokenWindow.of(tokens))
    return result


# --- fitting -------------------------------------------------------------------------------------


def test_statistics_are_the_count_mean_and_population_spread_per_channel(
    tokeniser: Tokeniser,
) -> None:
    scheme = fitted(tokeniser, REGULAR)

    for channel in CHANNELS:
        values = [o.value for o in REGULAR if o.channel == channel]
        fitted_statistics = scheme.statistics_of(scheme.vocabulary.id_of(CORPUS, channel))
        assert fitted_statistics.count == len(values)
        assert fitted_statistics.mean == pytest.approx(statistics.fmean(values))
        assert fitted_statistics.std == pytest.approx(statistics.pstdev(values))


def test_static_features_are_fitted_on_their_own_channel(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR, [AGE, StaticFeature("age", 65.0)], STATIC_SCHEMA)

    age = scheme.statistics_of(scheme.vocabulary.id_of(CORPUS, "age"))
    assert (age.count, age.mean, age.std) == (2, 63.0, 2.0)


def test_a_channel_the_data_never_shows_stays_unfitted(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, [o for o in REGULAR if o.channel == "pressure"])

    with pytest.raises(MissingChannelStatisticsError, match="temperature"):
        scheme.statistics_of(scheme.vocabulary.id_of(CORPUS, "temperature"))


def test_fitting_rejects_a_channel_outside_the_vocabulary(tokeniser: Tokeniser) -> None:
    with pytest.raises(UnknownChannelError, match="vibration"):
        fitted(tokeniser, [Observation("vibration", 0.0, 1.0)])


def test_fitting_rejects_a_static_feature_on_a_timed_channel(tokeniser: Tokeniser) -> None:
    with pytest.raises(ChannelKindMismatchError, match="static feature"):
        fitted(tokeniser, REGULAR, [StaticFeature("pressure", 1.0)])


def test_fitting_rejects_an_observation_on_a_timeless_channel(tokeniser: Tokeniser) -> None:
    with pytest.raises(ChannelKindMismatchError, match="observation"):
        fitted(tokeniser, [Observation("age", 0.0, 1.0)], schema=STATIC_SCHEMA)


def test_a_corpus_is_fitted_once(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR)

    with pytest.raises(ChannelAlreadyFittedError):
        tokeniser.fit(CORPUS, REGULAR, (), scheme)


# --- no leakage across the unit split -----------------------------------------------------------


def corpus_of(
    training_values: Sequence[float], validation_values: Sequence[float]
) -> tuple[InMemoryCorpusReader, UnitSplit]:
    units = [
        (
            unit("train", 0.0, 4.0),
            tuple(Observation("pressure", t, v) for t, v in enumerate(training_values)),
        ),
        (
            unit("validate", 0.0, 4.0),
            tuple(Observation("pressure", t, v) for t, v in enumerate(validation_values)),
        ),
    ]
    reader = InMemoryCorpusReader(description(units=2, observations=8), units)
    split = UnitSplit.by_seed([u.key for u, _ in units], 0.5, seed=3)
    return reader, split


def fit_on_training(
    tokeniser: Tokeniser, reader: InMemoryCorpusReader, split: UnitSplit
) -> TokenisationScheme:
    training = [u for u in reader.read_units() if u.key in split.training]
    observations = chain.from_iterable(reader.read_observations(u.key) for u in training)
    return tokeniser.fit(CORPUS, observations, (), scheme_for())


def test_statistics_ignore_the_validation_units(tokeniser: Tokeniser) -> None:
    reader, split = corpus_of([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    before = fit_on_training(tokeniser, reader, split)

    perturbed, _ = corpus_of([1.0, 2.0, 3.0, 4.0], [-99.0, 0.0, 99.0, 1e6])
    after = fit_on_training(tokeniser, perturbed, split)

    assert after == before


def test_statistics_follow_the_training_units(tokeniser: Tokeniser) -> None:
    reader, split = corpus_of([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    before = fit_on_training(tokeniser, reader, split)

    perturbed, _ = corpus_of([1.0, 2.0, 3.0, 400.0], [10.0, 20.0, 30.0, 40.0])
    after = fit_on_training(tokeniser, perturbed, split)

    assert after != before


# --- tokenising ----------------------------------------------------------------------------------


def test_windows_match_the_brute_force_definition_on_a_regular_grid(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR)

    assert windows_of(tokeniser, UNIT, REGULAR, scheme) == expected_windows(
        UNIT, REGULAR, scheme, WINDOW
    )


def test_tokenising_twice_gives_equal_windows(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR)

    assert windows_of(tokeniser, UNIT, REGULAR, scheme) == windows_of(
        tokeniser, UNIT, REGULAR, scheme
    )


def test_channels_observed_at_one_instant_may_arrive_in_any_order(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR)
    swapped = tuple(
        chain.from_iterable(
            reversed(REGULAR[i : i + len(CHANNELS)]) for i in range(0, len(REGULAR), 2)
        )
    )

    assert windows_of(tokeniser, UNIT, swapped, scheme) == windows_of(
        tokeniser, UNIT, REGULAR, scheme
    )


def test_a_regular_grid_gives_the_closed_form_window_count_and_token_count(
    tokeniser: Tokeniser,
) -> None:
    windows = windows_of(tokeniser, UNIT, REGULAR, identity_scheme())

    assert len(windows) == 4
    assert all(len(window) == WINDOW.length * len(CHANNELS) for window in windows)


def test_values_are_normalised_with_the_fitted_statistics(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR)
    (first, *_) = windows_of(tokeniser, UNIT, REGULAR, scheme)

    pressure = scheme.vocabulary.id_of(CORPUS, "pressure")
    pressure_at_zero = next(o for o in REGULAR if o.channel == "pressure" and o.time == 0.0)
    assert first.values[first.channel_ids.index(pressure)] == scheme.statistics_of(
        pressure
    ).normalise(pressure_at_zero.value)


def test_time_is_the_position_within_the_window_and_the_gap_the_distance_in_channel(
    tokeniser: Tokeniser,
) -> None:
    observations = (
        Observation("pressure", 1.0, 0.0),
        Observation("temperature", 1.0, 0.0),
        Observation("pressure", 1.5, 0.0),
        Observation("pressure", 3.0, 0.0),
    )
    (window,) = windows_of(
        tokeniser, unit("u", 0.0, 4.0), observations, identity_scheme(), WindowSpec(4, 4)
    )

    pressure = window.channel_ids.index(identity_scheme().vocabulary.id_of(CORPUS, "pressure"))
    assert window.times == (0.25, 0.25, 0.375, 0.75)
    assert window.gaps[pressure] == 0.25  # first in channel: since the window start
    assert window.gaps[2:] == (0.125, 0.375)  # then the distance to the previous in channel


def test_static_features_enter_every_window_as_timeless_tokens(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR, [AGE], STATIC_SCHEMA)
    age = scheme.vocabulary.id_of(CORPUS, "age")

    windows = windows_of(tokeniser, STATIC_UNIT, REGULAR, scheme)

    assert windows == expected_windows(STATIC_UNIT, REGULAR, scheme, WINDOW)
    for window in windows:
        assert window.channel_ids[0] == age
        assert window.timeless[0]
        assert (window.times[0], window.gaps[0]) == (0.0, 0.0)
        assert window.values[0] == scheme.statistics_of(age).normalise(AGE.value)
        assert sum(window.timeless) == 1


def test_a_unit_with_static_features_only_yields_no_window(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR, [AGE], STATIC_SCHEMA)

    assert windows_of(tokeniser, STATIC_UNIT, (), scheme) == []


def test_a_unit_shorter_than_a_window_yields_no_window(tokeniser: Tokeniser) -> None:
    short = unit("short", 0.0, 3.0)

    assert windows_of(tokeniser, short, grid(CHANNELS, [0.0, 1.0, 2.0]), identity_scheme()) == []


def test_a_window_that_falls_into_a_gap_in_the_data_is_skipped(tokeniser: Tokeniser) -> None:
    sparse = grid(CHANNELS, [0.0, 9.0])

    windows = windows_of(tokeniser, UNIT, sparse, identity_scheme(), WindowSpec(2, 2))

    assert [window.times for window in windows] == [(0.0, 0.0), (0.5, 0.5)]


def test_the_tail_no_full_window_covers_is_left_out(tokeniser: Tokeniser) -> None:
    windows = windows_of(tokeniser, UNIT, REGULAR, identity_scheme(), WindowSpec(4, 3))

    assert len(windows) == 3  # starts 0, 3, 6; a window from 9 would end beyond the extent
    assert max(windows[-1].times) == 0.75  # the observation at 9 reaches the last window


def test_observations_must_not_run_backwards(tokeniser: Tokeniser) -> None:
    backwards = (Observation("pressure", 2.0, 0.0), Observation("pressure", 1.0, 0.0))

    with pytest.raises(ObservationOutOfOrderError, match="follows"):
        windows_of(tokeniser, UNIT, backwards, identity_scheme())


@pytest.mark.parametrize("time", [-0.5, 10.0])
def test_observations_must_lie_inside_the_extent(tokeniser: Tokeniser, time: float) -> None:
    with pytest.raises(ObservationOutsideExtentError, match="outside"):
        windows_of(tokeniser, UNIT, (Observation("pressure", time, 0.0),), identity_scheme())


def test_tokenising_rejects_a_channel_outside_the_vocabulary(tokeniser: Tokeniser) -> None:
    with pytest.raises(UnknownChannelError, match="vibration"):
        windows_of(tokeniser, UNIT, (Observation("vibration", 0.0, 0.0),), identity_scheme())


def test_tokenising_rejects_an_observation_on_a_timeless_channel(tokeniser: Tokeniser) -> None:
    with pytest.raises(ChannelKindMismatchError, match="timeless"):
        windows_of(tokeniser, UNIT, (Observation("age", 0.0, 0.0),), identity_scheme(STATIC_SCHEMA))


def test_tokenising_rejects_a_static_feature_on_a_timed_channel(tokeniser: Tokeniser) -> None:
    with pytest.raises(ChannelKindMismatchError, match="timed"):
        windows_of(
            tokeniser,
            unit("u", 0.0, 10.0, StaticFeature("pressure", 1.0)),
            REGULAR,
            identity_scheme(),
        )


def test_tokenising_needs_statistics_for_every_channel_it_meets(tokeniser: Tokeniser) -> None:
    unfitted = fitted(tokeniser, [o for o in REGULAR if o.channel == "pressure"])

    with pytest.raises(MissingChannelStatisticsError, match="temperature"):
        windows_of(tokeniser, UNIT, REGULAR, unfitted)


# --- placing windows on the unit ----------------------------------------------------------------


def test_every_window_comes_with_the_span_it_was_cut_from(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR)

    placed = placed_of(tokeniser, UNIT, REGULAR, scheme)

    assert [item.extent for item in placed] == occupied_extents(UNIT, REGULAR, WINDOW)


def test_the_spans_skip_those_no_observation_fell_into(tokeniser: Tokeniser) -> None:
    # A unit measured at both ends of its life and not in between: three of the five windows laid
    # over it hold nothing, so the spans that come back cannot be the spans the spec lays.
    interrupted = grid(CHANNELS, [0.0, 1.0, 8.0, 9.0])
    scheme = fitted(tokeniser, interrupted)
    back_to_back = WindowSpec(length=2, stride=2)

    placed = placed_of(tokeniser, UNIT, interrupted, scheme, back_to_back)

    assert [item.extent for item in placed] == occupied_extents(UNIT, interrupted, back_to_back)
    assert len(placed) == 2
    assert len(list(back_to_back.windows_over(UNIT.extent))) == 5


# --- reading windows back ------------------------------------------------------------------------


def test_a_window_reads_back_as_the_observations_that_fell_into_it(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR)

    placed = placed_of(tokeniser, UNIT, REGULAR, scheme)

    for item in placed:
        inside = [o for o in REGULAR if item.extent.contains(o.time)]
        read_back = scheme.reconstruct(item.window, item.extent).observations
        assert measured(read_back) == measured(inside)


def test_reading_the_windows_back_recovers_nothing_that_fell_between_them(
    tokeniser: Tokeniser,
) -> None:
    scheme = fitted(tokeniser, REGULAR)
    # Longer stride than window: the observations from 2 to 4 and from 7 on reach no window.
    sparse = WindowSpec(length=2, stride=5)

    placed = placed_of(tokeniser, UNIT, REGULAR, scheme, sparse)

    read_back = chain.from_iterable(
        scheme.reconstruct(item.window, item.extent).observations for item in placed
    )
    covered = [o for o in REGULAR if o.time in (0.0, 1.0, 5.0, 6.0)]
    assert measured(read_back) == measured(covered)


def test_a_window_reads_back_the_static_features_of_its_unit(tokeniser: Tokeniser) -> None:
    scheme = fitted(tokeniser, REGULAR, [AGE], STATIC_SCHEMA)

    placed = placed_of(tokeniser, STATIC_UNIT, REGULAR, scheme)

    for item in placed:
        features = scheme.reconstruct(item.window, item.extent).static_features
        assert [(f.channel, round(f.value, 6)) for f in features] == [(AGE.channel, AGE.value)]


# --- synthetic irregularity ---------------------------------------------------------------------


@st.composite
def irregular_units(draw: st.DrawFn) -> tuple[CorpusUnit, tuple[Observation, ...], WindowSpec]:
    """A unit sampled at random instants, on a random subset of channels each time."""
    channels = [f"c{i}" for i in range(draw(st.integers(min_value=1, max_value=4)))]
    end = draw(st.floats(min_value=1.0, max_value=50.0))
    times = draw(
        st.lists(st.floats(min_value=0.0, max_value=end, exclude_max=True), min_size=0, max_size=40)
    )
    observations: list[Observation] = []
    for time in sorted(times):
        present = draw(st.lists(st.sampled_from(channels), min_size=1, unique=True))
        observations.extend(
            Observation(channel, time, draw(st.floats(min_value=-1e3, max_value=1e3)))
            for channel in present
        )
    statics = tuple(
        StaticFeature("s", draw(st.floats(min_value=-1e3, max_value=1e3)))
        for _ in range(draw(st.integers(min_value=0, max_value=1)))
    )
    window = WindowSpec(
        length=draw(st.floats(min_value=0.5, max_value=end)),
        stride=draw(st.floats(min_value=0.25, max_value=end)),
    )
    return unit("u", 0.0, end, *statics), tuple(observations), window


IRREGULAR_SCHEMA = ChannelSchema(
    frozenset({Channel(f"c{i}") for i in range(4)} | {Channel("s", timeless=True)})
)


@settings(max_examples=60, deadline=None)
@given(case=irregular_units())
def test_irregular_units_match_the_brute_force_definition(
    case: tuple[CorpusUnit, tuple[Observation, ...], WindowSpec],
) -> None:
    corpus_unit, observations, window = case
    scheme = identity_scheme(IRREGULAR_SCHEMA)
    for factory in ADAPTERS.values():
        tokeniser = factory()

        placed = list(tokeniser.tokenise(CORPUS, corpus_unit, observations, scheme, window))

        assert [item.extent for item in placed] == occupied_extents(
            corpus_unit, observations, window
        )
        windows = [item.window for item in placed]
        assert windows == expected_windows(corpus_unit, observations, scheme, window)
        for produced in windows:
            assert any(not timeless for timeless in produced.timeless)
            assert all(0.0 <= time < 1.0 for time in produced.times)
