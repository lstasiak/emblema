from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from emblema.config.api_settings import ApiSettings
from emblema.config.artifact_store_settings import ArtifactStoreSettings
from emblema.config.broker_settings import BrokerSettings
from emblema.config.database_settings import DatabaseSettings
from emblema.config.identity_settings import IdentitySettings
from emblema.config.telemetry_settings import TelemetrySettings
from emblema.config.worker_settings import WorkerSettings
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
    database: DatabaseSettings | None = Field(
        default=None,
        description=(
            "The metadata database the process keeps its registries in. A process that only "
            "trains — a notebook fulfilling an order — has none to reach and leaves it unset."
        ),
    )

    broker: BrokerSettings | None = Field(
        default=None,
        description=(
            "The queue background work passes through. A process that runs everything where it "
            "was asked — a report, a notebook — has none to reach and leaves it unset."
        ),
    )

    worker: WorkerSettings | None = Field(
        default=None,
        description=(
            "What the campaign worker serves and under which schedule. Every other process "
            "leaves it unset: only the one that runs cells is told these."
        ),
    )

    api: ApiSettings | None = Field(
        default=None,
        description=(
            "Where the HTTP process listens and how much one request may ask of it. Every "
            "other process leaves it unset."
        ),
    )

    telemetry: TelemetrySettings | None = Field(
        default=None,
        description=(
            "Where a process reports what it does. A process that reports nothing — a command "
            "run once from a shell — leaves it unset."
        ),
    )

    identity: IdentitySettings | None = Field(
        default=None,
        description=(
            "Who the HTTP process trusts to say who a caller is. Every other process leaves it "
            "unset: a command line acts as the operator of the machine it runs on."
        ),
    )

    def require_api(self) -> ApiSettings:
        """What the HTTP process was told about where it listens and what it allows.

        Raises:
            ValueError: If nothing is configured; the process fails as it is assembled rather
                than binding where nobody said.
        """
        if self.api is None:
            raise ValueError("the process serves HTTP and EMBLEMA_API__* is not set")
        return self.api

    def require_identity(self) -> IdentitySettings:
        """Whom the HTTP process was told to trust about who a caller is.

        Raises:
            ValueError: If nothing is configured; the process fails as it is assembled rather
                than serving routes that change what is served to whoever asks.
        """
        if self.identity is None:
            raise ValueError("the process protects routes and EMBLEMA_IDENTITY__* is not set")
        return self.identity

    def require_telemetry(self) -> TelemetrySettings:
        """How a process that reports itself was told to name itself and where to report.

        Raises:
            ValueError: If nothing is configured; the process fails as it is assembled rather
                than reporting under a name nobody gave it.
        """
        if self.telemetry is None:
            raise ValueError("the process reports itself and EMBLEMA_TELEMETRY__* is not set")
        return self.telemetry

    def require_worker(self) -> WorkerSettings:
        """What a process that runs campaign cells was told to run them with.

        Raises:
            ValueError: If nothing is configured; the worker fails as it is assembled rather
                than running a grid under a schedule nobody declared.
        """
        if self.worker is None:
            raise ValueError("the process runs campaign cells and EMBLEMA_WORKER__* is not set")
        return self.worker

    def require_broker(self) -> BrokerSettings:
        """The broker settings of a process that cannot run without a queue.

        Raises:
            ValueError: If none are configured; the process fails as it is assembled rather
                than reaching for a broker nobody named.
        """
        if self.broker is None:
            raise ValueError("the process needs the queue and EMBLEMA_BROKER__* is not set")
        return self.broker

    def require_database(self) -> DatabaseSettings:
        """The database settings of a process that cannot run without a registry.

        Raises:
            ValueError: If none are configured; the process fails as it is assembled rather
                than reaching for a database nobody named.
        """
        if self.database is None:
            raise ValueError(
                "the process needs the metadata database and EMBLEMA_DATABASE__* is not set"
            )
        return self.database
