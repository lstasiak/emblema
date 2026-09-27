"""Contract of the service the Catalog publishes for one window, run against every adapter.

The reference is the Catalog's own tokenisation of a whole unit: whatever window that produces
for a span, the service must produce for the same readings handed to it as one window. One
adapter, so the list has one entry, for the reason the tokeniser port's has.
"""

from collections.abc import Callable
from dataclasses import replace

import pytest

from emblema.catalog.adapters.tokenisation.published_window_tokeniser import (
    PublishedWindowTokeniser,
)
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.application.assemblers.published_corpus_manifest_assembler import (
    PublishedCorpusManifestAssembler,
)
from emblema.catalog.contracts.exceptions import UntokenisableWindowError
from emblema.catalog.contracts.observed_value import ObservedValue
from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.static_value import StaticValue
from emblema.catalog.contracts.window_tokeniser import WindowTokeniser
from emblema.catalog.domain.measurements.observation import Observation
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from tests.catalog.domain.support import CORPUS, STATIC_SCHEMA, grid, scheme_for, unit
from tests.support.published import CHANNELS as PUBLISHED_WITHOUT_AGE

ADAPTERS: dict[str, Callable[[], WindowTokeniser]] = {"published": PublishedWindowTokeniser}
AGE = StaticFeature("age", 61.0)
UNIT = unit("u1", 0.0, 10.0, AGE)
READINGS = grid(("temperature", "pressure"), [float(t) for t in range(10)])
WINDOW = WindowSpec(length=4, stride=2)


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def service(request: pytest.FixtureRequest) -> WindowTokeniser:
    factory: Callable[[], WindowTokeniser] = request.param
    return factory()


@pytest.fixture(scope="module")
def channels() -> tuple[PublishedChannel, ...]:
    """The corpus's channels as a manifest publishes them, fitted on the unit's readings."""
    scheme = SlidingWindowTokeniser().fit(CORPUS, READINGS, [AGE], scheme_for(STATIC_SCHEMA))
    assembler = PublishedCorpusManifestAssembler()
    return tuple(
        assembler._channel(entry, statistics)
        for entry, statistics in zip(scheme.vocabulary.entries, scheme.statistics, strict=True)
    )


def test_a_window_of_readings_tokenises_as_the_same_span_of_the_unit_did(
    service: WindowTokeniser, channels: tuple[PublishedChannel, ...]
) -> None:
    scheme = PublishedCorpusManifestAssembler().restore_scheme(channels)
    placed = list(SlidingWindowTokeniser().tokenise(CORPUS, UNIT, READINGS, scheme, WINDOW))

    for reference in placed:
        extent = reference.extent
        window = ObservedWindow(
            start=extent.start,
            length=extent.length,
            observations=tuple(
                ObservedValue(channel=o.channel, time=o.time, value=o.value)
                for o in READINGS
                if extent.contains(o.time)
            ),
            static_features=(StaticValue(channel=AGE.channel, value=AGE.value),),
        )

        assert service.tokenise(window, CORPUS, channels) == reference.window


def test_readings_are_placed_by_their_time_whatever_order_they_arrive_in(
    service: WindowTokeniser, channels: tuple[PublishedChannel, ...]
) -> None:
    inside = tuple(
        ObservedValue(channel=o.channel, time=o.time, value=o.value)
        for o in READINGS
        if 2.0 <= o.time < 6.0
    )
    ordered = ObservedWindow(start=2.0, length=4.0, observations=inside)
    reversed_ = replace(ordered, observations=tuple(reversed(inside)))

    assert service.tokenise(reversed_, CORPUS, channels) == service.tokenise(
        ordered, CORPUS, channels
    )


def test_a_reading_on_a_channel_the_corpus_does_not_know_is_refused(
    service: WindowTokeniser, channels: tuple[PublishedChannel, ...]
) -> None:
    window = ObservedWindow(
        start=0.0,
        length=4.0,
        observations=(ObservedValue(channel="vibration", time=1.0, value=1.0),),
    )

    with pytest.raises(UntokenisableWindowError, match="vibration"):
        service.tokenise(window, CORPUS, channels)


def test_a_reading_on_a_channel_never_fitted_is_refused(service: WindowTokeniser) -> None:
    # The shared published corpus declares `age` timeless without statistics: training never
    # observed it, so there is nothing to normalise a value by.
    window = ObservedWindow(
        start=0.0,
        length=4.0,
        observations=(ObservedValue(channel="temperature", time=1.0, value=1.0),),
        static_features=(StaticValue(channel="age", value=1.0),),
    )

    with pytest.raises(UntokenisableWindowError, match="statistics"):
        service.tokenise(window, "test-corpus", PUBLISHED_WITHOUT_AGE)


def test_a_timed_reading_on_a_static_channel_is_refused(
    service: WindowTokeniser, channels: tuple[PublishedChannel, ...]
) -> None:
    window = ObservedWindow(
        start=0.0,
        length=4.0,
        observations=(
            ObservedValue(channel="temperature", time=1.0, value=1.0),
            ObservedValue(channel="age", time=2.0, value=61.0),
        ),
    )

    with pytest.raises(UntokenisableWindowError, match="timeless"):
        service.tokenise(window, CORPUS, channels)


def test_a_static_value_on_a_measured_channel_is_refused(
    service: WindowTokeniser, channels: tuple[PublishedChannel, ...]
) -> None:
    window = ObservedWindow(
        start=0.0,
        length=4.0,
        observations=(ObservedValue(channel="temperature", time=1.0, value=1.0),),
        static_features=(StaticValue(channel="pressure", value=1.0),),
    )

    with pytest.raises(UntokenisableWindowError, match="timed"):
        service.tokenise(window, CORPUS, channels)


def test_the_observation_the_window_was_built_from_is_what_the_service_placed(
    service: WindowTokeniser, channels: tuple[PublishedChannel, ...]
) -> None:
    # A single reading halfway through a window sits at position one half with the same gap.
    window = ObservedWindow(
        start=10.0,
        length=4.0,
        observations=(ObservedValue(channel="pressure", time=12.0, value=3.0),),
    )
    scheme = PublishedCorpusManifestAssembler().restore_scheme(channels)
    pressure = scheme.vocabulary.id_of(CORPUS, "pressure")

    tokens = service.tokenise(window, CORPUS, channels)

    assert tokens.channel_ids == (pressure,)
    assert tokens.times == (0.5,)
    assert tokens.gaps == (0.5,)
    assert tokens.values == (scheme.statistics_of(pressure).normalise(3.0),)


def test_an_observation_of_the_unit_reads_the_same_as_a_reading_of_the_window() -> None:
    # The two ways of stating a value are the same value: the service merely re-spells one as
    # the other, so a test may build either from the other without loss.
    assert Observation("t", 1.0, 2.0) == Observation(
        ObservedValue(channel="t", time=1.0, value=2.0).channel, 1.0, 2.0
    )
