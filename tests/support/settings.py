"""Settings for a process assembled in a test: services named, none of them reached."""

import os
from typing import TYPE_CHECKING

from emblema.config.api_settings import ApiSettings
from emblema.config.artifact_store_settings import ArtifactStoreSettings
from emblema.config.broker_settings import BrokerSettings
from emblema.config.database_settings import DatabaseSettings
from emblema.config.settings import Settings
from emblema.config.telemetry_settings import TelemetrySettings

if TYPE_CHECKING:
    import pytest

ARTIFACT_STORE = ArtifactStoreSettings(
    endpoint_url="http://127.0.0.1:3900", region="garage", bucket="emblema", key_prefix="test"
)
DATABASE = DatabaseSettings(
    host="127.0.0.1", port=5432, name="emblema", user="nobody", password="unused"
)
BROKER = BrokerSettings(host="127.0.0.1", port=5672, user="nobody", password="unused", vhost="/")
API = ApiSettings(
    host="127.0.0.1",
    port=8000,
    workers=1,
    request_threads=4,
    max_request_bytes=1_000_000,
    max_windows_per_request=4,
    max_tokens_per_window=16,
    batch_size=4,
    onnx_providers="CPUExecutionProvider",
    onnx_threads=1,
    default_page_size=1,
    max_page_size=2,
    verdict_memo_capacity=4,
)
TELEMETRY = TelemetrySettings(service_name="emblema-test")


def unreachable_store() -> Settings:
    """Settings a composition may read while every adapter that would use them is overridden."""
    return Settings(
        artifact_store=ARTIFACT_STORE,
        database=DATABASE,
        broker=BROKER,
        api=API,
        telemetry=TELEMETRY,
    )


def only(monkeypatch: "pytest.MonkeyPatch", **named: str) -> None:
    """Leave the process holding these ``EMBLEMA_`` variables and no others.

    ``_env_file=None`` keeps this machine's file out of a test; it does nothing about what is
    already exported, and a half-named group is worse than none — every field of it is required,
    so ``Settings()`` refuses. A test about one setting should not depend on the rest.
    """
    for name in [name for name in os.environ if name.startswith("EMBLEMA_")]:
        monkeypatch.delenv(name)
    for name, value in named.items():
        monkeypatch.setenv(name, value)
