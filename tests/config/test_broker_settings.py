from urllib.parse import unquote, urlsplit

import pytest
from pydantic import ValidationError

from emblema.config.broker_settings import BrokerSettings


def settings(user: str = "emblema", password: str = "secret", vhost: str = "/") -> BrokerSettings:
    return BrokerSettings(host="queue.local", port=5672, user=user, password=password, vhost=vhost)


def test_the_url_names_what_the_settings_name() -> None:
    url = urlsplit(settings(vhost="emblema").url())

    assert (url.scheme, url.hostname, url.port, url.path) == (
        "amqp",
        "queue.local",
        5672,
        "/emblema",
    )


def test_the_broker_s_own_default_virtual_host_survives_being_named() -> None:
    # It is called "/", which is a path separator until it is encoded, so an unencoded one would
    # reach the broker as the empty vhost and the connection would be refused.
    url = urlsplit(settings().url())

    assert url.path == "/%2F"
    assert unquote(url.path.lstrip("/")) == "/"


@pytest.mark.parametrize("credential", ["p@ss:w/rd#1", "a?b", "pa ss", "per%20cent", "hasło"])
def test_a_credential_of_any_bytes_reaches_the_broker_as_it_was_set(credential: str) -> None:
    url = urlsplit(settings(user=credential, password=credential).url())

    assert url.hostname == "queue.local"
    assert unquote(url.username or "") == credential
    assert unquote(url.password or "") == credential


def test_the_password_is_not_read_off_the_settings_by_accident() -> None:
    assert "secret" not in str(settings())


@pytest.mark.parametrize("field", ["host", "port", "user", "password", "vhost"])
def test_nothing_naming_the_broker_has_a_default(field: str) -> None:
    values = {
        "host": "queue.local",
        "port": 5672,
        "user": "emblema",
        "password": "secret",
        "vhost": "/",
    }
    del values[field]

    with pytest.raises(ValidationError):
        BrokerSettings(**values)
