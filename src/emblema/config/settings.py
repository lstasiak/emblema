from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from emblema.config.artifact_store_settings import ArtifactStoreSettings
from emblema.shared.kernel.compute import ComputeTier

Environment = Literal["dev", "test", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Runtime settings of one process: API, worker or notebook.

    Read from the environment by the composition root, which passes the values a use case needs
    into its constructor; nothing below the adapters sees this class.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBLEMA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = "dev"
    log_level: LogLevel = "INFO"
    default_compute_tier: ComputeTier = Field(
        default=ComputeTier.S,
        description=(
            "Tier assumed when an experiment configuration declares none. "
            "The declared tier is authoritative."
        ),
    )
    artifact_store: ArtifactStoreSettings
