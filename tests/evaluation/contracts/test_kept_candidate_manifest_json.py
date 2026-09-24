import json
from typing import Any

import pytest

from emblema.evaluation.contracts.exceptions import MalformedKeptCandidateManifestError
from emblema.evaluation.contracts.kept_candidate_manifest_json import KeptCandidateManifestJson
from tests.evaluation.contracts.test_kept_candidate_manifest import manifest

MANIFEST = manifest()
JSON = KeptCandidateManifestJson()


def document_of() -> dict[str, Any]:
    return json.loads(JSON.encode(MANIFEST))


def encoded(document: object) -> bytes:
    return json.dumps(document).encode("utf-8")


def test_a_manifest_comes_back_as_it_went_in() -> None:
    assert JSON.decode(JSON.encode(MANIFEST)) == MANIFEST


def test_the_bytes_are_compact_with_keys_in_order() -> None:
    # A candidate kept twice must be one manifest in a content-addressed store, so nothing about
    # the bytes may depend on the order a dictionary was built in or on formatting.
    raw = JSON.encode(MANIFEST)

    document = json.loads(raw)
    assert list(document) == sorted(document)
    assert b": " not in raw
    assert b"\n" not in raw


def test_a_reader_with_nothing_but_the_standard_library_finds_the_form_it_can_run() -> None:
    document = document_of()

    forms = {form["format"]: form for form in document["representations"]}
    assert forms["onnx"]["artifact"]["checksum"]["algorithm"] == "sha256"
    assert forms["onnx"]["deviation"] == pytest.approx(1.5e-5)
    assert forms[document["measured_as"]]["deviation"] is None


@pytest.mark.parametrize("raw", [b"\xff\xfe", b"not json", b"[]", b'"a string"'])
def test_bytes_that_are_not_a_manifest_object_are_refused(raw: bytes) -> None:
    with pytest.raises(MalformedKeptCandidateManifestError):
        JSON.decode(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [("format", "emblema.something-else"), ("version", 9)],
    ids=["another format", "another version"],
)
def test_another_format_or_version_is_refused(key: str, value: object) -> None:
    document = document_of()
    document[key] = value

    with pytest.raises(MalformedKeptCandidateManifestError, match="version"):
        JSON.decode(encoded(document))


@pytest.mark.parametrize(
    "spoil",
    [
        lambda document: document.pop("representations"),
        lambda document: document["representations"][1].update(deviation="1.5e-5"),
        lambda document: document.update(kind="quantum"),
        lambda document: document.update(measured_as="npz"),
    ],
    ids=["a field missing", "a field of the wrong type", "an unknown kind", "a rule broken"],
)
def test_a_document_that_does_not_make_a_manifest_is_refused(spoil: Any) -> None:
    document = document_of()
    spoil(document)

    with pytest.raises(MalformedKeptCandidateManifestError):
        JSON.decode(encoded(document))
