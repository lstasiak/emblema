"""Whom the HTTP process trusts about who a caller is, read through the environment."""

import pytest
from pydantic import ValidationError

from emblema.config.named_issuer import NamedIssuer
from emblema.config.settings import Settings
from emblema.shared.kernel.identity.principal import Principal
from emblema.shared.kernel.identity.scope import Scope
from tests.config.test_api_settings import STORE
from tests.support.settings import only

ISSUER = {
    "EMBLEMA_IDENTITY__ISSUER": "https://issuer.example",
    "EMBLEMA_IDENTITY__AUDIENCE": "emblema-api",
    "EMBLEMA_IDENTITY__JWKS_URL": "https://issuer.example/.well-known/jwks.json",
}


def test_a_process_told_nothing_of_identity_holds_no_group_and_the_api_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    only(monkeypatch, **STORE)

    settings = Settings(_env_file=None)

    assert settings.identity is None
    with pytest.raises(ValueError, match="EMBLEMA_IDENTITY__"):
        settings.require_identity()


def test_static_tokens_are_one_string_split_by_the_process_into_principals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    only(
        monkeypatch,
        **STORE,
        EMBLEMA_IDENTITY__STATIC_TOKENS=(
            "abc123:operator:serving:promote serving:withdraw, def456:auditor: ,"
        ),
    )

    identity = Settings(_env_file=None).require_identity()

    assert identity.named_issuer() is None
    assert identity.principals() == {
        "abc123": Principal(
            subject="operator",
            scopes=frozenset({Scope("serving:promote"), Scope("serving:withdraw")}),
        ),
        "def456": Principal(subject="auditor"),
    }


def test_the_tokens_are_a_secret_the_settings_never_print(monkeypatch: pytest.MonkeyPatch) -> None:
    only(monkeypatch, **STORE, EMBLEMA_IDENTITY__STATIC_TOKENS="abc123:operator:serving:promote")

    assert "abc123" not in repr(Settings(_env_file=None).require_identity())


@pytest.mark.parametrize("entry", ["abc123", "abc123:operator", "abc123:operator:Promote"])
def test_a_static_token_not_written_as_one_is_refused_when_read(
    monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    only(monkeypatch, **STORE, EMBLEMA_IDENTITY__STATIC_TOKENS=entry)

    with pytest.raises(ValueError, match=r"token:subject:scopes|resource:action"):
        Settings(_env_file=None).require_identity().principals()


def test_a_token_listed_twice_is_refused_rather_than_resolved_to_one_of_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    only(
        monkeypatch,
        **STORE,
        EMBLEMA_IDENTITY__STATIC_TOKENS="abc123:operator:serving:promote,abc123:auditor:",
    )

    with pytest.raises(ValueError, match="listed twice"):
        Settings(_env_file=None).require_identity().principals()


def test_an_issuer_is_named_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    only(monkeypatch, **STORE, **ISSUER)

    identity = Settings(_env_file=None).require_identity()

    assert identity.named_issuer() == NamedIssuer(
        issuer="https://issuer.example",
        audience="emblema-api",
        jwks_url="https://issuer.example/.well-known/jwks.json",
    )
    assert identity.principals() == {}


@pytest.mark.parametrize("missing", list(ISSUER))
def test_an_issuer_named_in_part_is_refused(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    only(monkeypatch, **STORE, **{name: value for name, value in ISSUER.items() if name != missing})

    with pytest.raises(ValidationError, match="all of issuer, audience and jwks_url"):
        Settings(_env_file=None)


def test_naming_both_providers_or_neither_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    only(monkeypatch, **STORE, **ISSUER, EMBLEMA_IDENTITY__STATIC_TOKENS="abc123:operator:")
    with pytest.raises(ValidationError, match="not both"):
        Settings(_env_file=None)

    only(monkeypatch, **STORE, EMBLEMA_IDENTITY__STATIC_TOKENS=" ")
    with pytest.raises(ValidationError, match="not neither"):
        Settings(_env_file=None)
