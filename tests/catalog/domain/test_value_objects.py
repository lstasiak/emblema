"""Invariants of the small value objects: licence, source, content, unit, observation."""

import pytest

from emblema.catalog.domain.channel_schema import Channel
from emblema.catalog.domain.corpus_content import CorpusContent
from emblema.catalog.domain.corpus_source import CorpusSource
from emblema.catalog.domain.corpus_unit import CorpusUnit, TimeExtent
from emblema.catalog.domain.exceptions import (
    CatalogError,
    InvalidCorpusContentError,
    InvalidCorpusSourceError,
    InvalidCorpusUnitError,
    InvalidLicenceError,
    InvalidObservationError,
    InvalidStaticFeatureError,
    InvalidTimeExtentError,
    InvalidUnitKeyError,
)
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.licence import Licence
from emblema.catalog.domain.observation import Observation
from emblema.catalog.domain.static_feature import StaticFeature
from emblema.shared.kernel.checksums import Checksum

NON_FINITE = [float("nan"), float("inf"), float("-inf")]
BLANK_OR_PADDED = ["", "   ", " x", "x "]


@pytest.mark.parametrize("identifier", ["", "   ", " CC-BY-4.0", "CC-BY-4.0 "])
def test_licence_rejects_blank_or_padded_identifier(identifier: str) -> None:
    with pytest.raises(InvalidLicenceError):
        Licence(identifier, permits_derivatives=False)


@pytest.mark.parametrize("url", ["", " ", " https://x", "https://x "])
def test_licence_rejects_blank_or_padded_url(url: str) -> None:
    with pytest.raises(InvalidLicenceError):
        Licence("CC-BY-4.0", permits_derivatives=False, url=url)


def test_licence_without_url_is_valid() -> None:
    assert Licence("CC-BY-4.0", permits_derivatives=False).url is None


@pytest.mark.parametrize(
    ("name", "uri"),
    [("", "https://x"), ("NASA", ""), (" ", " "), (" NASA", "https://x"), ("NASA", "https://x ")],
)
def test_source_rejects_blank_or_padded_fields(name: str, uri: str) -> None:
    with pytest.raises(InvalidCorpusSourceError):
        CorpusSource(name, uri)


@pytest.mark.parametrize(
    ("units", "observations"), [(0, 5), (1, 0)], ids=["no-unit", "no-observation"]
)
def test_content_requires_a_unit_and_an_observation(units: int, observations: int) -> None:
    with pytest.raises(InvalidCorpusContentError, match="positive"):
        CorpusContent(Checksum.of_bytes(b"x"), units, observations)


def test_content_allows_units_without_observations() -> None:
    assert CorpusContent(Checksum.of_bytes(b"x"), 3, 2).unit_count == 3


def test_a_channel_is_timed_unless_declared_otherwise() -> None:
    assert not Channel("T2").timeless
    assert Channel("age", timeless=True).timeless


@pytest.mark.parametrize("text", BLANK_OR_PADDED)
def test_unit_key_rejects_blank_or_padded_text(text: str) -> None:
    with pytest.raises(InvalidUnitKeyError):
        UnitKey(text)


def test_unit_key_prints_as_its_text() -> None:
    assert str(UnitKey("FD001/39")) == "FD001/39"


@pytest.mark.parametrize("channel", BLANK_OR_PADDED)
def test_observation_rejects_blank_or_padded_channel(channel: str) -> None:
    with pytest.raises(InvalidObservationError, match="non-blank"):
        Observation(channel, 0.0, 1.0)


@pytest.mark.parametrize("time", NON_FINITE)
def test_observation_time_is_finite(time: float) -> None:
    with pytest.raises(InvalidObservationError, match="time"):
        Observation("T2", time, 1.0)


@pytest.mark.parametrize("value", NON_FINITE)
def test_observation_value_is_finite_because_a_missing_value_is_no_observation(
    value: float,
) -> None:
    with pytest.raises(InvalidObservationError, match="value"):
        Observation("T2", 0.0, value)


@pytest.mark.parametrize("channel", BLANK_OR_PADDED)
def test_static_feature_rejects_blank_or_padded_channel(channel: str) -> None:
    with pytest.raises(InvalidStaticFeatureError, match="non-blank"):
        StaticFeature(channel, 1.0)


@pytest.mark.parametrize("value", NON_FINITE)
def test_static_feature_value_is_finite(value: float) -> None:
    with pytest.raises(InvalidStaticFeatureError, match="finite"):
        StaticFeature("age", value)


@pytest.mark.parametrize(("start", "end"), [(1.0, 1.0), (2.0, 1.0)])
def test_extent_start_precedes_end(start: float, end: float) -> None:
    with pytest.raises(InvalidTimeExtentError, match="precede"):
        TimeExtent(start, end)


@pytest.mark.parametrize(("start", "end"), [(float("nan"), 1.0), (0.0, float("inf"))])
def test_extent_bounds_are_finite(start: float, end: float) -> None:
    with pytest.raises(InvalidTimeExtentError, match="finite"):
        TimeExtent(start, end)


def test_extent_is_half_open() -> None:
    extent = TimeExtent(1.0, 4.0)

    assert extent.length == 3.0
    assert extent.contains(1.0)
    assert extent.contains(3.999)
    assert not extent.contains(4.0)
    assert not extent.contains(0.999)


def test_unit_static_features_are_one_per_channel() -> None:
    with pytest.raises(InvalidCorpusUnitError, match="duplicate"):
        CorpusUnit(
            UnitKey("u"),
            TimeExtent(0.0, 1.0),
            (StaticFeature("age", 1.0), StaticFeature("age", 2.0)),
        )


def test_a_unit_has_no_static_features_unless_given() -> None:
    assert CorpusUnit(UnitKey("u"), TimeExtent(0.0, 1.0)).static_features == ()


def test_invariant_errors_are_both_catalog_and_value_errors() -> None:
    with pytest.raises(CatalogError):
        Licence("", permits_derivatives=False)
    with pytest.raises(ValueError, match="non-blank"):
        Licence("", permits_derivatives=False)
    with pytest.raises(ValueError, match="finite"):
        Observation("T2", 0.0, float("nan"))
