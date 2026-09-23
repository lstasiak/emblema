"""The adapters every command-line process builds from its settings, stated once.

Two adapters come from the environment rather than from a caller: the artifact store and the
database engine. Every process that reaches either builds it the same way, so the connection is
written here and a composition root names the function, and stays the place that says which
adapter each use case got. Functions rather than a class, because there is no state to hold: the
settings go in, the adapter comes out.
"""

from sqlalchemy import Engine, create_engine

from emblema.config.settings import Settings
from emblema.shared.adapters.storage.s3 import S3ArtifactStore
from emblema.shared.ports.artifact_store import ArtifactStore


def configured_store(settings: Settings) -> ArtifactStore:
    """The S3-compatible bucket the settings name, connected."""
    config = settings.artifact_store
    return S3ArtifactStore.connect(
        endpoint_url=config.endpoint_url,
        region=config.region,
        access_key=config.access_key.get_secret_value() if config.access_key else None,
        secret_key=config.secret_key.get_secret_value() if config.secret_key else None,
        bucket=config.bucket,
        key_prefix=config.key_prefix,
    )


def configured_engine(settings: Settings) -> Engine:
    """An engine on the metadata database the settings name.

    The engine opens no connection until the first query, so assembling a process costs no
    network.

    Raises:
        ValueError: If the settings name no database.
    """
    return create_engine(settings.require_database().sqlalchemy_url())


def settings_for(settings: Settings | None, adapter: str) -> Settings:
    """The settings an adapter is built from, refused where a process brought none.

    Raises:
        ValueError: If there are no settings; the adapter had to be given instead.
    """
    if settings is None:
        raise ValueError(f"without settings the process needs {adapter} given, not built")
    return settings
