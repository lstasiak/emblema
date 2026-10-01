"""The HTTP process, assembled from what the settings name.

What it must not load is the training stack: a served network is a graph, and the image the
process ships in exists to leave torch out. What it must hold is one path for both things a
model answers, so a prediction and a representation of one window stand on the same tokens.
"""

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from emblema.catalog.adapters.tokenisation.published_window_tokeniser import (
    PublishedWindowTokeniser,
)
from emblema.config.identity_settings import IdentitySettings
from emblema.entrypoints.api.api_server import ApiServer
from emblema.entrypoints.api.composition_root import CompositionRoot
from emblema.entrypoints.api.telemetry.counted_answers import CountedAnswers
from emblema.entrypoints.api.telemetry.instrumented_inference_runtime import (
    InstrumentedInferenceRuntime,
)
from emblema.entrypoints.api.telemetry.instrumented_window_tokeniser import (
    InstrumentedWindowTokeniser,
)
from emblema.entrypoints.api.telemetry.telemetry import Telemetry
from emblema.evaluation.adapters.in_memory.verdict_memo import InMemoryVerdictMemo
from emblema.evaluation.adapters.persistence.campaign_listing import SqlAlchemyCampaignListing
from emblema.evaluation.adapters.persistence.evaluation_campaign_repository import (
    SqlAlchemyEvaluationCampaignRepository,
)
from emblema.serving.adapters.persistence.promotable_artifact_repository import (
    SqlAlchemyPromotableArtifactRepository,
)
from emblema.serving.adapters.persistence.served_model_listing import SqlAlchemyServedModelListing
from emblema.serving.adapters.persistence.served_model_repository import (
    SqlAlchemyServedModelRepository,
)
from emblema.serving.adapters.routing.format_routed_inference_runtime import (
    FormatRoutedInferenceRuntime,
)
from emblema.shared.adapters.identity.jwt_identity_provider import JwtIdentityProvider
from emblema.shared.adapters.identity.static_token_identity_provider import (
    StaticTokenIdentityProvider,
)
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.adapters.system.clock import SystemClock
from emblema.shared.adapters.system.id_generator import Uuid4IdGenerator
from tests.support.settings import TELEMETRY, unreachable_store


def test_without_overrides_the_process_runs_on_what_the_settings_name() -> None:
    root = CompositionRoot(unreachable_store())

    assert isinstance(root.adapters.store, S3ArtifactStore)
    assert isinstance(root.adapters.served, SqlAlchemyServedModelRepository)
    assert isinstance(root.adapters.served_listing, SqlAlchemyServedModelListing)
    assert isinstance(root.adapters.promotables, SqlAlchemyPromotableArtifactRepository)
    assert isinstance(root.adapters.identity, StaticTokenIdentityProvider)
    assert isinstance(root.adapters.clock, SystemClock)
    assert isinstance(root.adapters.ids, Uuid4IdGenerator)
    assert isinstance(root.adapters.campaigns, SqlAlchemyEvaluationCampaignRepository)
    assert isinstance(root.adapters.campaign_listing, SqlAlchemyCampaignListing)
    assert isinstance(root.adapters.runtime, FormatRoutedInferenceRuntime)
    assert isinstance(root.adapters.tokeniser, PublishedWindowTokeniser)
    assert isinstance(root.adapters.verdicts, InMemoryVerdictMemo)


def test_an_issuer_in_the_settings_makes_the_process_verify_its_tokens() -> None:
    settings = unreachable_store().model_copy(
        update={
            "identity": IdentitySettings(
                issuer="https://issuer.example",
                audience="emblema-api",
                jwks_url="https://issuer.example/jwks",
            )
        }
    )

    root = CompositionRoot(settings)

    assert isinstance(root.adapters.identity, JwtIdentityProvider)


def test_a_process_told_no_identity_provider_refuses_to_assemble() -> None:
    settings = unreachable_store().model_copy(update={"identity": None})

    with pytest.raises(ValueError, match="EMBLEMA_IDENTITY__"):
        CompositionRoot(settings)


def test_a_process_bringing_neither_settings_nor_its_registries_is_refused() -> None:
    with pytest.raises(ValueError, match="store"):
        CompositionRoot()


def test_assembling_this_process_loads_no_training_stack() -> None:
    read = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import emblema.entrypoints.api.composition_root  # noqa: F401\n"
            "print(sorted(m for m in sys.modules if m in {'torch', 'mlflow', 'celery'}))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert read.stdout.strip() == "[]"


def test_the_server_assembles_the_application_the_settings_describe_without_binding() -> None:
    api = ApiServer(unreachable_store()).application()

    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/metrics").status_code == 200


def test_with_telemetry_the_runtime_and_tokeniser_are_timed_and_the_answers_counted() -> None:
    telemetry = Telemetry(TELEMETRY)
    try:
        root = CompositionRoot(unreachable_store(), telemetry=telemetry)
    finally:
        telemetry.shutdown()

    assert isinstance(root.adapters.runtime, InstrumentedInferenceRuntime)
    assert isinstance(root.adapters.tokeniser, InstrumentedWindowTokeniser)
    assert isinstance(root.services.predict_windows, CountedAnswers)
    assert isinstance(root.services.embed_windows, CountedAnswers)
