"""The two invocations an artifact is put into service and taken out of it by."""

import pytest

from emblema.entrypoints.cli.serving.composition_root import CompositionRoot
from emblema.entrypoints.cli.serving.serving_cli import ServingCli
from emblema.serving.adapters.in_memory.promotable_artifact_repository import (
    InMemoryPromotableArtifactRepository,
)
from emblema.serving.adapters.in_memory.served_model_repository import (
    InMemoryServedModelRepository,
)
from emblema.serving.application.use_cases.promote_artifact import PromoteArtifactCommand
from emblema.serving.application.use_cases.withdraw_served_model import (
    WithdrawServedModelCommand,
)
from emblema.serving.domain.identifiers import ServedModelId
from emblema.serving.domain.served_model_state import ServedModelState
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore
from emblema.shared.adapters.in_memory.clock import FixedClock
from emblema.shared.adapters.in_memory.id_generator import SequentialIdGenerator
from tests.serving.support import CAMPAIGN, FITTED, MODEL, PROMOTED, TREES, promotable


@pytest.fixture
def root() -> CompositionRoot:
    return CompositionRoot(
        store=InMemoryArtifactStore(),
        promotables=InMemoryPromotableArtifactRepository(),
        served=InMemoryServedModelRepository(),
        clock=FixedClock(PROMOTED),
        ids=SequentialIdGenerator(),
    )


def run(root: CompositionRoot, *argv: str) -> str:
    return ServingCli().execute(ServingCli().parse(argv), root.services)


def test_a_promotion_names_the_artifact_by_checksum_and_optionally_its_origin() -> None:
    checksum = str(promotable().artifact.checksum)

    parsed = ServingCli().parse(
        [
            "promote",
            "--checksum",
            checksum,
            "--campaign",
            str(CAMPAIGN),
            "--candidate",
            str(TREES),
        ]
    )

    assert parsed.command == PromoteArtifactCommand(
        checksum=promotable().artifact.checksum, campaign=CAMPAIGN, candidate=TREES
    )


def test_a_withdrawal_names_the_model() -> None:
    parsed = ServingCli().parse(["withdraw", "--model", str(MODEL)])

    assert parsed.command == WithdrawServedModelCommand(served_model=MODEL)


def test_a_promoted_artifact_prints_the_model_that_now_serves_it_and_can_be_withdrawn(
    root: CompositionRoot,
) -> None:
    stored = root.adapters.store.put(FITTED)
    root.adapters.promotables.save(promotable(artifact=stored))

    model = run(root, "promote", "--checksum", str(stored.checksum))
    withdrawn = run(root, "withdraw", "--model", model)

    assert withdrawn == model
    assert root.adapters.served.get(ServedModelId.parse(model)).state is (
        ServedModelState.WITHDRAWN
    )


def test_a_refusal_is_reported_as_its_reason_rather_than_a_traceback(
    root: CompositionRoot,
) -> None:
    with pytest.raises(SystemExit, match="no finished campaign kept"):
        run(root, "promote", "--checksum", str(promotable().artifact.checksum))
