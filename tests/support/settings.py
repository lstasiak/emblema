"""Settings for a process assembled in a test or a report: services named, none of them reached."""

from emblema.config.artifact_store_settings import ArtifactStoreSettings
from emblema.config.database_settings import DatabaseSettings
from emblema.config.settings import Settings

ARTIFACT_STORE = ArtifactStoreSettings(
    endpoint_url="http://127.0.0.1:3900", region="garage", bucket="emblema", key_prefix="test"
)
DATABASE = DatabaseSettings(
    host="127.0.0.1", port=5432, name="emblema", user="nobody", password="unused"
)


def unreachable_store() -> Settings:
    """Settings a composition may read while every adapter that would use them is overridden."""
    return Settings(artifact_store=ARTIFACT_STORE, database=DATABASE)
