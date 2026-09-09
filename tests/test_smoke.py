import importlib

import pytest
from pydantic import ValidationError

from emblema.config.artifact_store_settings import ArtifactStoreSettings
from emblema.config.settings import Settings
from emblema.shared.kernel.compute import ComputeTier

CONTEXTS = ("catalog", "pretraining", "evaluation", "serving")
# Every process must name its bucket; these tests are about the rest of the settings.
ARTIFACT_STORE = ArtifactStoreSettings(
    endpoint_url="http://localhost:3900", region="garage", bucket="emblema", key_prefix="dev"
)


def test_package_imports() -> None:
    import emblema

    assert emblema.__name__ == "emblema"


@pytest.mark.parametrize("context", CONTEXTS)
def test_every_bounded_context_package_exists(context: str) -> None:
    module = importlib.import_module(f"emblema.{context}")

    assert module.__doc__


def test_compute_tiers_are_exactly_s_m_l() -> None:
    assert [tier.value for tier in ComputeTier] == ["S", "M", "L"]


def test_settings_default_tier_is_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBLEMA_DEFAULT_COMPUTE_TIER", raising=False)

    settings = Settings(_env_file=None, artifact_store=ARTIFACT_STORE)

    assert settings.default_compute_tier is ComputeTier.S
    assert settings.environment == "dev"


def test_settings_read_default_tier_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBLEMA_DEFAULT_COMPUTE_TIER", "M")

    assert (
        Settings(_env_file=None, artifact_store=ARTIFACT_STORE).default_compute_tier
        is ComputeTier.M
    )


def test_settings_reject_unknown_default_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBLEMA_DEFAULT_COMPUTE_TIER", "XL")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, artifact_store=ARTIFACT_STORE)
