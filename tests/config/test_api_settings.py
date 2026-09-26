import pytest
from pydantic import ValidationError

from emblema.config.settings import Settings
from tests.support.settings import only

STORE = {
    "EMBLEMA_ARTIFACT_STORE__ENDPOINT_URL": "http://127.0.0.1:3900",
    "EMBLEMA_ARTIFACT_STORE__REGION": "garage",
    "EMBLEMA_ARTIFACT_STORE__BUCKET": "emblema",
    "EMBLEMA_ARTIFACT_STORE__KEY_PREFIX": "test",
}
API = {
    "EMBLEMA_API__HOST": "0.0.0.0",
    "EMBLEMA_API__PORT": "8080",
    "EMBLEMA_API__WORKERS": "2",
    "EMBLEMA_API__REQUEST_THREADS": "8",
    "EMBLEMA_API__MAX_REQUEST_BYTES": "4000000",
    "EMBLEMA_API__MAX_WINDOWS_PER_REQUEST": "64",
    "EMBLEMA_API__MAX_TOKENS_PER_WINDOW": "4096",
    "EMBLEMA_API__BATCH_SIZE": "16",
    "EMBLEMA_API__ONNX_PROVIDERS": " CoreMLExecutionProvider, CPUExecutionProvider ,",
    "EMBLEMA_API__ONNX_THREADS": "2",
    "EMBLEMA_API__DEFAULT_PAGE_SIZE": "20",
    "EMBLEMA_API__MAX_PAGE_SIZE": "100",
    "EMBLEMA_API__VERDICT_MEMO_CAPACITY": "64",
}


def test_a_process_told_nothing_of_the_api_holds_no_api_group_and_the_api_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    only(monkeypatch, **STORE)

    settings = Settings(_env_file=None)

    assert settings.api is None
    assert settings.telemetry is None
    with pytest.raises(ValueError, match="EMBLEMA_API__"):
        settings.require_api()
    with pytest.raises(ValueError, match="EMBLEMA_TELEMETRY__"):
        settings.require_telemetry()


def test_where_the_api_binds_is_never_assumed(monkeypatch: pytest.MonkeyPatch) -> None:
    named = {name: value for name, value in API.items() if name != "EMBLEMA_API__HOST"}
    only(monkeypatch, **STORE, **named)

    with pytest.raises(ValidationError, match="host"):
        Settings(_env_file=None)


def test_lists_are_read_as_comma_separated_text_and_split_by_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    only(
        monkeypatch,
        **STORE,
        **API,
        EMBLEMA_API__CORS_ORIGINS=" http://localhost:5173, https://emblema.example ,,",
        EMBLEMA_TELEMETRY__SERVICE_NAME="emblema-api",
        EMBLEMA_TELEMETRY__OTLP_ENDPOINT="http://127.0.0.1:4318",
    )

    settings = Settings(_env_file=None)

    api = settings.require_api()
    assert api.origins() == ("http://localhost:5173", "https://emblema.example")
    assert api.providers() == ("CoreMLExecutionProvider", "CPUExecutionProvider")
    assert api.port == 8080
    assert settings.require_telemetry().otlp_endpoint == "http://127.0.0.1:4318"


def test_without_origins_no_browser_is_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    only(monkeypatch, **STORE, **API)

    assert Settings(_env_file=None).require_api().origins() == ()


@pytest.mark.parametrize(
    "name",
    [
        "EMBLEMA_API__PORT",
        "EMBLEMA_API__WORKERS",
        "EMBLEMA_API__REQUEST_THREADS",
        "EMBLEMA_API__MAX_REQUEST_BYTES",
        "EMBLEMA_API__MAX_WINDOWS_PER_REQUEST",
        "EMBLEMA_API__MAX_TOKENS_PER_WINDOW",
        "EMBLEMA_API__BATCH_SIZE",
        "EMBLEMA_API__ONNX_THREADS",
        "EMBLEMA_API__DEFAULT_PAGE_SIZE",
        "EMBLEMA_API__MAX_PAGE_SIZE",
        "EMBLEMA_API__VERDICT_MEMO_CAPACITY",
    ],
)
def test_a_count_of_nothing_is_refused_as_the_process_is_configured(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    only(monkeypatch, **STORE, **(API | {name: "0"}))

    with pytest.raises(ValidationError, match=name.removeprefix("EMBLEMA_API__").lower()):
        Settings(_env_file=None)
