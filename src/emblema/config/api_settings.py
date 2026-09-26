from pydantic import BaseModel, Field


class ApiSettings(BaseModel):
    """What the HTTP process is told before it starts, read as ``EMBLEMA_API__<FIELD>``.

    Where it listens, what it runs on, and how much one caller may ask of it. Nothing but the
    browser origins has a default: where a process binds depends on where it is deployed, and a
    limit nobody stated is a limit nobody can defend.
    """

    host: str
    port: int = Field(ge=1, le=65535)
    workers: int = Field(
        ge=1, description="Server processes, each holding its own models, memo and connections."
    )
    request_threads: int = Field(
        ge=1,
        description=(
            "Requests one process answers at once, and the database connections it pools for "
            "them, so that no request waits for a connection another thread holds."
        ),
    )
    cors_origins: str = Field(
        default="",
        description=(
            "Origins a browser may call the service from, comma-separated; none by default, "
            "which admits no browser. Kept as text and split by the process, so that a list "
            "survives every shell and file it is written in."
        ),
    )
    max_request_bytes: int = Field(
        ge=1, description="Largest body a request may carry; a larger one is refused unread."
    )
    max_windows_per_request: int = Field(ge=1)
    max_tokens_per_window: int = Field(ge=1)
    batch_size: int = Field(
        ge=1, description="Windows run through a graph at once, padded to the longest."
    )
    onnx_providers: str = Field(
        description=(
            "ONNX Runtime execution providers a graph runs on, comma-separated, in order of "
            "preference, e.g. CPUExecutionProvider."
        )
    )
    onnx_threads: int = Field(
        ge=1, description="Threads one graph call may use; the rest are other requests'."
    )
    default_page_size: int = Field(ge=1)
    max_page_size: int = Field(ge=1)
    verdict_memo_capacity: int = Field(
        ge=1,
        description="Campaign verdicts one process keeps once read, the least recent forgotten.",
    )

    def origins(self) -> tuple[str, ...]:
        """The origins the browser may call from, one per entry, blanks dropped."""
        return self._split(self.cors_origins)

    def providers(self) -> tuple[str, ...]:
        """The execution providers a graph runs on, in order of preference, blanks dropped."""
        return self._split(self.onnx_providers)

    @staticmethod
    def _split(text: str) -> tuple[str, ...]:
        return tuple(entry.strip() for entry in text.split(",") if entry.strip())
