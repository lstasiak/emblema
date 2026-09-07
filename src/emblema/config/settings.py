from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from emblema.shared.kernel.compute import ComputeTier

Environment = Literal["dev", "test", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Runtime settings of one process: API, worker or notebook."""

    model_config = SettingsConfigDict(
        env_prefix="EMBLEMA_",
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
