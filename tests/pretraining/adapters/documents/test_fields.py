import pytest

from emblema.pretraining.adapters.documents.fields import Fields
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum

REF = ArtifactRef("durable/thing", Checksum.of_bytes(b"thing"))


def test_every_reader_names_the_key_it_could_not_read() -> None:
    fields = Fields({"text": 1, "count": "two", "number": True, "object": [], "list": {}})

    for reader, key in (
        (fields.text, "text"),
        (fields.integer, "count"),
        (fields.number, "number"),
        (fields.fields, "object"),
        (fields.each, "list"),
        (fields.text, "absent"),
    ):
        with pytest.raises(ValueError, match=repr(key)):
            reader(key)


def test_a_boolean_is_neither_an_integer_nor_a_number() -> None:
    fields = Fields({"flag": True})

    with pytest.raises(ValueError, match="integer"):
        fields.integer("flag")
    with pytest.raises(ValueError, match="number"):
        fields.number("flag")


def test_a_list_item_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(ValueError, match="every item"):
        Fields({"items": [1]}).each("items")


def test_references_and_checksums_round_trip_and_an_absent_one_is_none() -> None:
    document = {"ref": Fields.of_ref(REF), "none": Fields.of_optional_ref(None)}
    fields = Fields(document)

    assert fields.ref("ref") == REF
    assert fields.fields("ref").checksum("checksum") == REF.checksum
    assert fields.optional_ref("none") is None
    assert fields.optional_ref("missing") is None
    assert fields.optional_fields("none") is None
    assert fields.optional_text("none") is None


def test_a_number_may_be_written_as_an_integer() -> None:
    assert Fields({"rate": 1}).number("rate") == 1.0
