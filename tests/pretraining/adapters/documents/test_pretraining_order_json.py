import json

import pytest

from emblema.pretraining.adapters.documents.pretraining_order_json import PretrainingOrderJson
from emblema.pretraining.domain.exceptions import UnreadableHandoffDocumentError
from tests.support.handoff import order

CODEC = PretrainingOrderJson()


def test_an_order_round_trips_and_is_the_same_bytes_every_time() -> None:
    placed = order(run="stable")

    encoded = CODEC.encode(placed)

    assert CODEC.decode(encoded) == placed
    assert CODEC.encode(placed) == encoded


def test_the_document_names_its_format_and_what_a_reader_of_it_needs() -> None:
    document = json.loads(CODEC.encode(order()))

    assert document["format"] == "emblema.pretraining-order"
    assert document["version"] == 2
    assert set(document) >= {
        "backbone",
        "configuration",
        "manifest",
        "run",
        "git_commit",
        "signature",
    }


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"\xff", "not JSON"),
        (b"[]", "not an object"),
        (b'{"format": "emblema.pretraining-result", "version": 1}', "not an order"),
        (b'{"format": "emblema.pretraining-order", "version": 1}', "version 1"),
    ],
)
def test_bytes_that_are_not_an_order_are_refused(content: bytes, message: str) -> None:
    with pytest.raises(UnreadableHandoffDocumentError, match=message):
        CODEC.decode(content)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backbone", "not-a-uuid", "UUID"),
        ("run", "", "run"),
        ("signature", "", "digest"),
        ("manifest", {"key": "x"}, "checksum"),
    ],
)
def test_an_order_that_breaks_its_own_rules_is_refused(
    field: str, value: object, message: str
) -> None:
    document = json.loads(CODEC.encode(order())) | {field: value}

    with pytest.raises(UnreadableHandoffDocumentError, match=message):
        CODEC.decode(json.dumps(document).encode())
