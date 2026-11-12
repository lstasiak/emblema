import json
from dataclasses import replace
from typing import Any

import pytest

from emblema.catalog.contracts.exceptions import MalformedManifestError
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import (
    NORMALISATION_QUANTILE,
    NORMALISATION_ROBUST,
    ROBUST_BOUND_UNSTATED,
    PublishedChannelStatistics,
)
from emblema.catalog.contracts.published_corpus_manifest import SHORT_UNITS_ONE_WINDOW
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.catalog.contracts.published_value_bounds import PublishedValueBounds
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.catalog.contracts.support import PRESSURE, TEMPERATURE, manifest

MANIFEST = manifest(empty_units=("u3",), training=("u1", "u3"))
JSON = PublishedCorpusManifestJson()


def document_of() -> dict[str, Any]:
    return json.loads(JSON.encode(MANIFEST))


def encoded(document: object) -> bytes:
    return json.dumps(document).encode("utf-8")


def test_a_manifest_comes_back_as_it_went_in() -> None:
    assert JSON.decode(JSON.encode(MANIFEST)) == MANIFEST


def test_the_bytes_are_compact_with_keys_in_order() -> None:
    # Two publications of one corpus must store one description, so nothing about the bytes may
    # depend on the order a dictionary was built in or on formatting.
    raw = JSON.encode(MANIFEST)

    document = json.loads(raw)
    assert list(document) == sorted(document)
    assert b": " not in raw
    assert b"\n" not in raw


@pytest.mark.parametrize("raw", [b"\xff\xfe", b"not json", b"[]", b'"a string"'])
def test_bytes_that_are_not_a_manifest_object_are_refused(raw: bytes) -> None:
    with pytest.raises(MalformedManifestError):
        JSON.decode(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [("format", "emblema.something-else"), ("version", 9)],
    ids=["another format", "another version"],
)
def test_another_format_or_version_is_refused(key: str, value: object) -> None:
    document = document_of()
    document[key] = value

    with pytest.raises(MalformedManifestError, match="version"):
        JSON.decode(encoded(document))


def test_a_missing_field_is_refused() -> None:
    document = document_of()
    del document["units"]

    with pytest.raises(MalformedManifestError, match="missing"):
        JSON.decode(encoded(document))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("window_count", "3"),
        ("window_count", True),
        ("units", "u1"),
        ("corpus", 5),
        ("block", "durable/sha256/abc"),
        ("channels", [1, 2]),
    ],
)
def test_a_field_of_the_wrong_type_is_refused(key: str, value: object) -> None:
    document = document_of()
    document[key] = value

    with pytest.raises(MalformedManifestError, match="must be"):
        JSON.decode(encoded(document))


@pytest.mark.parametrize(
    ("key", "value"),
    [("unit", 5), ("timeless", "yes"), ("statistics", 1.0)],
)
def test_a_channel_field_of_the_wrong_type_is_refused(key: str, value: object) -> None:
    document = document_of()
    document["channels"][0][key] = value

    with pytest.raises(MalformedManifestError, match=key):
        JSON.decode(encoded(document))


@pytest.mark.parametrize(
    ("section", "key"),
    [("window", "length"), ("window", "stride")],
)
def test_a_measurement_that_is_not_a_number_is_refused(section: str, key: str) -> None:
    document = document_of()
    document[section][key] = str(document[section][key])

    with pytest.raises(MalformedManifestError, match="must be a number"):
        JSON.decode(encoded(document))


def test_a_statistic_that_is_not_a_number_is_refused() -> None:
    document = document_of()
    document["channels"][0]["statistics"]["mean"] = "0.5"

    with pytest.raises(MalformedManifestError, match="must be a number"):
        JSON.decode(encoded(document))


def test_a_manifest_that_breaks_its_own_rules_is_refused() -> None:
    document = document_of()
    document["units"] = ["u1", "u1"]

    with pytest.raises(MalformedManifestError, match="duplicate"):
        JSON.decode(encoded(document))


def test_a_checksum_of_an_unknown_algorithm_is_refused() -> None:
    document = document_of()
    document["corpus_checksum"]["algorithm"] = "md5"

    with pytest.raises(MalformedManifestError, match="well formed"):
        JSON.decode(encoded(document))


def test_a_manifest_whose_units_were_named_records_no_seed() -> None:
    named = replace(MANIFEST, split_seed=None)

    document = json.loads(JSON.encode(named))

    assert document["split"]["seed"] is None
    assert JSON.decode(JSON.encode(named)) == named


def test_a_seed_that_is_neither_a_whole_number_nor_absent_is_refused() -> None:
    document = document_of()
    document["split"]["seed"] = "seven"

    with pytest.raises(MalformedManifestError, match="integer or null"):
        JSON.decode(encoded(document))


def test_the_policy_for_short_units_is_written_only_where_it_is_not_the_usual_one() -> None:
    # A manifest written before the policy existed keeps its bytes, and so its checksum.
    assert "short_units" not in document_of()["window"]

    stated = replace(MANIFEST, window_short_units=SHORT_UNITS_ONE_WINDOW)
    document = json.loads(JSON.encode(stated))
    assert document["window"]["short_units"] == SHORT_UNITS_ONE_WINDOW
    assert JSON.decode(JSON.encode(stated)) == stated


def test_where_the_statistics_were_adopted_from_is_written_only_where_they_were() -> None:
    # A manifest whose training side fitted its statistics keeps its bytes, and so its checksum.
    assert "statistics_from" not in document_of()

    digest = "1" * 64
    stated = replace(
        MANIFEST,
        statistics_from=ArtifactRef(f"durable/sha256/{digest}", Checksum.parse(f"sha256:{digest}")),
    )
    document = json.loads(JSON.encode(stated))
    assert document["statistics_from"]["key"] == f"durable/sha256/{digest}"
    assert JSON.decode(JSON.encode(stated)) == stated


def test_a_policy_for_short_units_the_manifest_does_not_know_is_refused() -> None:
    document = document_of()
    document["window"]["short_units"] = "two_windows"

    with pytest.raises(MalformedManifestError, match="shorter than a window"):
        JSON.decode(encoded(document))


QUANTILES = tuple(float(i) for i in range(1001))
ROBUST = replace(
    PRESSURE,
    statistics=PublishedChannelStatistics(
        8, 0.5, 0.25, normalisation=NORMALISATION_ROBUST, median=0.4, mad=0.1, bound=10.0
    ),
)
ROBUST_TO_A_HUNDRED = replace(
    PRESSURE,
    statistics=PublishedChannelStatistics(
        8, 0.5, 0.25, normalisation=NORMALISATION_ROBUST, median=0.4, mad=0.1, bound=100.0
    ),
)
BY_QUANTILE = replace(
    PRESSURE,
    statistics=PublishedChannelStatistics(
        8, 0.5, 0.25, normalisation=NORMALISATION_QUANTILE, quantiles=QUANTILES
    ),
)
BOUNDED = PublishedValueBounds(
    channel="pressure",
    removed_below=0.0,
    clipped_below=1.0,
    clipped_above=9.0,
    removed_above=10.0,
    removed=11,
    clipped=2,
)


@pytest.mark.parametrize(
    "channel",
    [ROBUST, ROBUST_TO_A_HUNDRED, BY_QUANTILE],
    ids=["robust", "robust to a hundred", "quantile"],
)
def test_statistics_of_another_normalisation_come_back_as_they_went_in(
    channel: PublishedChannel,
) -> None:
    other = replace(MANIFEST, channels=(channel, TEMPERATURE), cleaning=(BOUNDED,))

    assert JSON.decode(JSON.encode(other)) == other


def test_a_manifest_normalised_by_the_standard_score_keeps_its_version_and_fields() -> None:
    # Every manifest published before the other normalisations keeps its bytes, so its checksum.
    document = document_of()

    assert document["version"] == PublishedCorpusManifestJson.VERSION
    assert "cleaning" not in document
    assert set(document["channels"][0]["statistics"]) == {"count", "mean", "std"}


def test_a_manifest_normalised_otherwise_is_of_a_version_an_older_reader_refuses() -> None:
    other = replace(MANIFEST, channels=(ROBUST, TEMPERATURE))

    document = json.loads(JSON.encode(other))

    assert document["version"] == PublishedCorpusManifestJson.VERSION_OTHER_NORMALISATION
    assert document["channels"][0]["statistics"]["normalisation"] == NORMALISATION_ROBUST


def test_the_usual_bound_is_the_one_a_robust_manifest_states_by_stating_none() -> None:
    # Every robust manifest written before a bound could be stated keeps its bytes.
    document = json.loads(JSON.encode(replace(MANIFEST, channels=(ROBUST, TEMPERATURE))))

    assert "bound" not in document["channels"][0]["statistics"]
    read = JSON.decode(encoded(document)).channels[0].statistics
    assert read == ROBUST.statistics
    assert read is not None
    assert read.bound == ROBUST_BOUND_UNSTATED


def test_a_manifest_held_to_another_bound_is_of_a_version_an_older_reader_refuses() -> None:
    document = json.loads(
        JSON.encode(replace(MANIFEST, channels=(ROBUST_TO_A_HUNDRED, TEMPERATURE)))
    )

    assert document["version"] == PublishedCorpusManifestJson.VERSION_ROBUST_BOUND
    assert document["channels"][0]["statistics"]["bound"] == 100.0


@pytest.mark.parametrize(
    ("channel", "version"),
    [
        (ROBUST, PublishedCorpusManifestJson.VERSION_ROBUST_BOUND),
        (ROBUST_TO_A_HUNDRED, PublishedCorpusManifestJson.VERSION_OTHER_NORMALISATION),
    ],
    ids=["usual bound claiming the third", "another bound claiming the second"],
)
def test_a_manifest_whose_version_does_not_say_whether_a_bound_is_stated_is_refused(
    channel: PublishedChannel, version: int
) -> None:
    document = json.loads(JSON.encode(replace(MANIFEST, channels=(channel, TEMPERATURE))))
    document["version"] = version

    with pytest.raises(MalformedManifestError, match="held to another bound"):
        JSON.decode(encoded(document))


def test_cleaning_alone_keeps_the_first_version() -> None:
    document = json.loads(JSON.encode(replace(MANIFEST, cleaning=(BOUNDED,))))

    assert document["version"] == PublishedCorpusManifestJson.VERSION
    assert document["cleaning"][0]["removed"] == 11


def test_a_first_version_manifest_claiming_another_normalisation_is_refused() -> None:
    document = json.loads(JSON.encode(replace(MANIFEST, channels=(ROBUST, TEMPERATURE))))
    document["version"] = PublishedCorpusManifestJson.VERSION

    with pytest.raises(MalformedManifestError, match="otherwise than by the standard score"):
        JSON.decode(encoded(document))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"normalisation": "logarithm"}, "where stated"),
        ({"normalisation": "standard"}, "where stated"),
        ({"normalisation": ""}, "where stated"),
        ({"quantiles": [1.0, 2.0]}, "state"),
        ({"normalisation": "robust", "median": 0.4}, "state"),
        ({"normalisation": "quantile", "quantiles": ["a"]}, "list of numbers"),
        ({"normalisation": "quantile", "quantiles": [0.0] * 1001, "bound": 100.0}, "belongs to"),
        ({"bound": 100.0}, "belongs to"),
        ({"normalisation": "robust", "median": 0.4, "mad": 0.1, "bound": 10.0}, "never written"),
    ],
    ids=[
        "unknown",
        "standard written",
        "empty",
        "quantiles on the standard score",
        "robust without a deviation",
        "text",
        "a bound on the quantile score",
        "a bound on the standard score",
        "the usual bound written",
    ],
)
def test_statistics_that_do_not_fit_their_normalisation_are_refused(
    change: dict[str, object], message: str
) -> None:
    document = document_of()
    document["version"] = PublishedCorpusManifestJson.VERSION_OTHER_NORMALISATION
    document["channels"][0]["statistics"].update(change)

    with pytest.raises(MalformedManifestError, match=message):
        JSON.decode(encoded(document))


def test_a_second_version_manifest_normalised_by_the_standard_score_alone_is_refused() -> None:
    # Read and written again it would become a first-version manifest: other bytes, other checksum.
    document = document_of()
    document["version"] = PublishedCorpusManifestJson.VERSION_OTHER_NORMALISATION

    with pytest.raises(MalformedManifestError, match="otherwise than by the standard score"):
        JSON.decode(encoded(document))
