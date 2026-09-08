"""Contract of the IdGenerator port, run against every adapter."""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from emblema.shared.kernel.identifiers import EntityId
from emblema.shared.ports.id_generator import IdGenerator


@dataclass(frozen=True)
class SampleId(EntityId):
    pass


ADAPTERS: dict[str, Callable[[], IdGenerator]] = {
    "uuid4": Uuid4IdGenerator,
    "sequential": SequentialIdGenerator,
}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def ids(request: pytest.FixtureRequest) -> IdGenerator:
    factory: Callable[[], IdGenerator] = request.param
    return factory()


def test_generates_an_identifier_of_the_requested_kind(ids: IdGenerator) -> None:
    assert isinstance(ids.generate(SampleId), SampleId)


def test_consecutive_identifiers_differ(ids: IdGenerator) -> None:
    assert ids.generate(SampleId) != ids.generate(SampleId)


def test_sequential_generator_counts_from_one() -> None:
    ids = SequentialIdGenerator()

    assert [ids.generate(SampleId).value.int for _ in range(3)] == [1, 2, 3]
