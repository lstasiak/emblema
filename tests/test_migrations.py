"""The migration tree and the models describe the same database.

A table a model has and the migrations do not would be found by the first query against it, in
a process, on another machine. Here it is found by comparing the two on the migrated database of
the local stack, which is what a deployment runs against: every context's schema against its own
metadata, because the tree is one and each schema is one context's. A migration that moves rows
is run both ways here as well, on rows a repository wrote.
"""

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import MetaData

from emblema.catalog.adapters.persistence.corpus_record import CorpusRecord

# Imported so that the task's tables register on the Evaluation metadata, as in migrations/env.py.
from emblema.evaluation.adapters.persistence.downstream_task_record import (  # noqa: F401
    DownstreamTaskRecord,
)
from emblema.evaluation.adapters.persistence.evaluation_campaign_record import (
    EvaluationCampaignRecord,
)
from emblema.pretraining.adapters.persistence.backbone_record import BackboneRecord
from emblema.pretraining.adapters.persistence.backbone_repository import (
    SqlAlchemyBackboneRepository,
)
from emblema.serving.adapters.persistence.promotable_artifact_record import (
    PromotableArtifactRecord,
)

# Imported so that the served model's table registers on the Serving metadata, as in
# migrations/env.py.
from emblema.serving.adapters.persistence.served_model_record import (  # noqa: F401
    ServedModelRecord,
)
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.support.database import REPO_ROOT, clear_pretraining, migrated_engine
from tests.support.handoff import backbone, pretraining_input

pytestmark = pytest.mark.integration

BEFORE_INPUTS_WERE_ORDERED = "0002"
SECOND_MANIFEST = ArtifactRef("durable/second", Checksum.of_bytes(b"second manifest"))


@pytest.mark.parametrize(
    "metadata",
    [
        CorpusRecord.metadata,
        BackboneRecord.metadata,
        EvaluationCampaignRecord.metadata,
        PromotableArtifactRecord.metadata,
    ],
    ids=lambda metadata: str(metadata.schema),
)
def test_the_migrations_produce_exactly_the_tables_the_model_describes(metadata: MetaData) -> None:
    engine = migrated_engine()

    def within_the_schema(name: str | None, kind: str, parents: object) -> bool:
        return kind != "schema" or name == metadata.schema

    with engine.connect() as connection:
        context = MigrationContext.configure(
            connection,
            opts={
                "include_schemas": True,
                "include_name": within_the_schema,
                "compare_type": True,
                "compare_server_default": True,
            },
        )
        differences = compare_metadata(context, metadata)

    assert differences == []


def test_a_backbone_over_one_corpus_survives_the_ordering_of_inputs_both_ways() -> None:
    engine = migrated_engine()
    clear_pretraining(engine)
    backbones = SqlAlchemyBackboneRepository(engine)
    tree = Config(str(REPO_ROOT / "alembic.ini"))
    single = backbone()
    backbones.save(single)

    try:
        command.downgrade(tree, BEFORE_INPUTS_WERE_ORDERED)
        command.upgrade(tree, "head")
    finally:
        command.upgrade(tree, "head")

    assert backbones.get(single.id) == single


def test_a_backbone_over_several_corpora_refuses_the_downgrade_that_would_drop_its_inputs() -> None:
    engine = migrated_engine()
    clear_pretraining(engine)
    backbones = SqlAlchemyBackboneRepository(engine)
    tree = Config(str(REPO_ROOT / "alembic.ini"))
    mixed = backbone(
        inputs=(
            pretraining_input(),
            pretraining_input(corpus="second", manifest=SECOND_MANIFEST, vocabulary_size=5),
        )
    )
    backbones.save(mixed)

    try:
        with pytest.raises(RuntimeError, match="1 input rows beyond position 0"):
            command.downgrade(tree, BEFORE_INPUTS_WERE_ORDERED)
    finally:
        command.upgrade(tree, "head")

    assert backbones.get(mixed.id) == mixed
