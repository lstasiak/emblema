import json
from dataclasses import replace
from typing import Any

import pytest

from emblema.catalog.contracts.exceptions import MalformedManifestError
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from tests.catalog.contracts.support import manifest

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
