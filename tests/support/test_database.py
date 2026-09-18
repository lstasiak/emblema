"""The tests own a database of their own and refuse to empty any other.

None of these reach a server: what is tested is which database the support module names and
what it refuses to touch, which is what keeps the development registry safe from a test run.
"""

import os
from collections.abc import Callable

import pytest
from sqlalchemy import Engine, create_engine

from tests.support.database import (
    DATABASE_NAME_VARIABLE,
    NotATestDatabaseError,
    clear_catalog,
    clear_pretraining,
    name_test_database,
)

# A server nobody runs: a guard that let a connection through would fail here, not pass.
UNREACHABLE = "postgresql+psycopg://nobody:unused@127.0.0.1:1/"


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for field, value in (
        ("ARTIFACT_STORE__ENDPOINT_URL", "http://127.0.0.1:3900"),
        ("ARTIFACT_STORE__REGION", "garage"),
        ("ARTIFACT_STORE__BUCKET", "emblema"),
        ("ARTIFACT_STORE__KEY_PREFIX", "test"),
        ("DATABASE__HOST", "127.0.0.1"),
        ("DATABASE__PORT", "1"),
        ("DATABASE__USER", "nobody"),
        ("DATABASE__PASSWORD", "unused"),
    ):
        monkeypatch.setenv(f"EMBLEMA_{field}", value)


@pytest.mark.parametrize("clear", [clear_catalog, clear_pretraining])
def test_a_database_without_the_suffix_is_never_emptied(clear: Callable[[Engine], None]) -> None:
    engine = create_engine(f"{UNREACHABLE}emblema")

    with pytest.raises(NotATestDatabaseError, match="refusing to empty 'emblema'"):
        clear(engine)


@pytest.mark.usefixtures("configured")
def test_the_test_database_is_the_configured_one_with_the_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DATABASE_NAME_VARIABLE, "emblema")

    settings = name_test_database()

    assert settings.name == "emblema_test"
    assert settings.user == "nobody"
    assert os.environ[DATABASE_NAME_VARIABLE] == "emblema_test"


@pytest.mark.usefixtures("configured")
def test_a_name_that_carries_the_suffix_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATABASE_NAME_VARIABLE, "emblema_test")

    assert name_test_database().name == "emblema_test"


@pytest.mark.usefixtures("configured")
@pytest.mark.parametrize("name", ["Emblema", "emblema-dev", "emblema;drop", "1st"])
def test_a_name_the_tests_could_not_create_is_refused(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(DATABASE_NAME_VARIABLE, name)

    with pytest.raises(NotATestDatabaseError, match="not a database name"):
        name_test_database()
