import pytest
from pydantic import ValidationError

from emblema.config.database_settings import DatabaseSettings


def settings(password: str = "secret", user: str = "emblema") -> DatabaseSettings:
    return DatabaseSettings(
        host="db.local", port=5433, name="emblema", user=user, password=password
    )


def test_the_url_names_what_the_settings_name() -> None:
    url = settings().sqlalchemy_url()

    assert (url.drivername, url.host, url.port, url.database) == (
        "postgresql+psycopg",
        "db.local",
        5433,
        "emblema",
    )
    assert (url.username, url.password) == ("emblema", "secret")


def test_credentials_reach_the_url_escaped_not_formatted() -> None:
    # A password with URL syntax in it must stay one password; built by parts it cannot be split
    # into a host or a port the way a formatted string would be.
    url = settings(password="p@ss:w/rd#1", user="em@blema").sqlalchemy_url()

    assert url.password == "p@ss:w/rd#1"
    assert "p@ss:w/rd#1" not in url.render_as_string(hide_password=False)
    assert url.render_as_string(hide_password=False).endswith("@db.local:5433/emblema")


def test_the_password_does_not_render_by_default() -> None:
    assert "secret" not in str(settings().sqlalchemy_url())


@pytest.mark.parametrize("field", ["host", "port", "name", "user", "password"])
def test_nothing_about_the_database_has_a_default(field: str) -> None:
    values = {"host": "db.local", "port": 5433, "name": "emblema", "user": "e", "password": "s"}
    del values[field]

    with pytest.raises(ValidationError):
        DatabaseSettings(**values)
