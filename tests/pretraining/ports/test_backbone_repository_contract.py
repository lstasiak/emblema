"""Contract of the BackboneRepository port, run against every adapter.

The in-memory adapter runs everywhere; the database adapter needs the metadata database of the
local stack and is marked ``integration``. Both store a whole state and give it back whole, the
ordered backbone and the delivered one alike.
"""

import pytest
from sqlalchemy import Engine

from emblema.pretraining.adapters.in_memory.backbone_repository import InMemoryBackboneRepository
from emblema.pretraining.adapters.persistence.backbone_repository import (
    SqlAlchemyBackboneRepository,
)
from emblema.pretraining.domain.exceptions import BackboneNotFoundError
from emblema.pretraining.ports.backbone_repository import BackboneRepository
from tests.support.database import clear_pretraining, migrated_engine
from tests.support.experiments import WEIGHTS, budget, configuration
from tests.support.handoff import (
    DELIVERED_AT,
    RESULT_REF,
    backbone,
    backbone_id,
    pretraining_input,
)

ADAPTERS = [
    pytest.param("in_memory", id="in_memory"),
    pytest.param("sqlalchemy", id="sqlalchemy", marks=pytest.mark.integration),
]


@pytest.fixture(scope="session")
def database() -> Engine:
    return migrated_engine()


@pytest.fixture(params=ADAPTERS)
def backbones(request: pytest.FixtureRequest) -> BackboneRepository:
    if request.param == "in_memory":
        return InMemoryBackboneRepository()
    engine: Engine = request.getfixturevalue("database")
    clear_pretraining(engine)
    return SqlAlchemyBackboneRepository(engine)


def test_an_unknown_backbone_is_reported(backbones: BackboneRepository) -> None:
    with pytest.raises(BackboneNotFoundError):
        backbones.get(backbone_id(42))


def test_an_ordered_backbone_reads_back_whole(backbones: BackboneRepository) -> None:
    ordered = backbone(
        configuration=configuration(name="stored", budget=budget(seed=3, epochs=4)),
        input=pretraining_input(vocabulary_size=17),
        run="second",
    )

    backbones.save(ordered)

    assert backbones.get(ordered.id) == ordered


def test_saving_the_delivered_state_replaces_the_order(backbones: BackboneRepository) -> None:
    ordered = backbone()
    backbones.save(ordered)

    delivered = ordered.deliver(RESULT_REF, WEIGHTS, DELIVERED_AT)
    backbones.save(delivered)

    assert backbones.get(ordered.id) == delivered
    assert backbones.get(ordered.id).is_ready
    assert backbones.get(ordered.id).result == RESULT_REF


def test_backbones_are_kept_apart_by_identity(backbones: BackboneRepository) -> None:
    first, second = (
        backbone(id=backbone_id(1), run="first"),
        backbone(id=backbone_id(2), run="second"),
    )

    backbones.save(first)
    backbones.save(second)

    assert backbones.get(first.id).run == "first"
    assert backbones.get(second.id).run == "second"
