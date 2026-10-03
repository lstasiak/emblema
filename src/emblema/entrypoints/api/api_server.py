import uvicorn
from fastapi import FastAPI

from emblema.config.settings import Settings
from emblema.entrypoints.api.composition_root import CompositionRoot
from emblema.entrypoints.api.emblema_api import EmblemaApi
from emblema.entrypoints.api.telemetry.telemetry import Telemetry


class ApiServer:
    """The HTTP process, as it is started.

    ::

        uv run python -m emblema.entrypoints.api

    Each server process builds its own application from the environment, because a model, a
    memo and a connection pool are not shared across processes. Assembling and serving are two
    steps, so that what the process is made of can be read without binding a port.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def application(self) -> EmblemaApi:
        """The application over the adapters the settings name, telemetry composed in.

        Raises:
            ValueError: If the settings name no API or no telemetry.
        """
        return self._assembled(Telemetry(self._settings.require_telemetry()))

    def run(self) -> None:  # pragma: no cover - environment
        api = self._settings.require_api()
        uvicorn.run(
            f"{__name__}:{ApiServer.__name__}.{ApiServer.served.__name__}",
            factory=True,
            host=api.host,
            port=api.port,
            workers=api.workers,
        )

    @classmethod
    def served(cls) -> FastAPI:  # pragma: no cover - environment
        """The application one server process serves, with the process itself instrumented."""
        server = cls(Settings())
        telemetry = Telemetry(server._settings.require_telemetry())
        telemetry.instrument_process(server._settings.log_level)
        return server._assembled(telemetry).app

    def _assembled(self, telemetry: Telemetry) -> EmblemaApi:
        root = CompositionRoot(self._settings, telemetry=telemetry)
        return EmblemaApi(
            root.services,
            root.readiness,
            self._settings.require_api(),
            telemetry,
            identity=root.adapters.identity,
        )
