from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from emblema.shared.kernel.exceptions import InvalidEntityIdError
from emblema.shared.kernel.identifiers import EntityId

CANONICAL = str(UUID(int=0xABCDEF))


@dataclass(frozen=True)
class SampleId(EntityId):
    pass


@dataclass(frozen=True)
class OtherId(EntityId):
    pass


def test_parse_round_trips_through_str() -> None:
    identifier = SampleId(uuid4())

    assert SampleId.parse(str(identifier)) == identifier


@pytest.mark.parametrize(
    "text",
    [
        "not-a-uuid",
        CANONICAL.upper(),
        CANONICAL.replace("-", ""),
        f"{{{CANONICAL}}}",
        f"urn:uuid:{CANONICAL}",
    ],
    ids=["garbage", "uppercase", "no-hyphens", "braces", "urn"],
)
def test_parse_rejects_malformed_or_non_canonical_text(text: str) -> None:
    with pytest.raises(InvalidEntityIdError):
        SampleId.parse(text)


def test_same_uuid_under_different_kinds_is_not_equal() -> None:
    # The point of wrapping: a corpus version and a backbone minted from one UUID are two
    # identities, and a dictionary keyed by one of them never answers for the other.
    sample: EntityId = SampleId(UUID(int=7))
    other: EntityId = OtherId(UUID(int=7))

    assert sample != other
    assert {sample: "corpus"}.get(other) is None
