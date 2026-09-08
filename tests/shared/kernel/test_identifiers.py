from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from emblema.shared.kernel.exceptions import InvalidEntityIdError
from emblema.shared.kernel.identifiers import EntityId


@dataclass(frozen=True)
class SampleId(EntityId):
    pass


@dataclass(frozen=True)
class OtherId(EntityId):
    pass


def test_parse_round_trips_through_str() -> None:
    identifier = SampleId(uuid4())

    assert SampleId.parse(str(identifier)) == identifier


def test_parse_rejects_malformed_text() -> None:
    with pytest.raises(InvalidEntityIdError):
        SampleId.parse("not-a-uuid")


def test_same_uuid_under_different_kinds_is_not_equal() -> None:
    sample: EntityId = SampleId(UUID(int=7))
    other: EntityId = OtherId(UUID(int=7))

    assert sample != other


def test_same_kind_and_uuid_are_equal_and_hash_alike() -> None:
    value = UUID(int=7)

    assert SampleId(value) == SampleId(value)
    assert hash(SampleId(value)) == hash(SampleId(value))
