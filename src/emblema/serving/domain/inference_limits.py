from dataclasses import dataclass

from emblema.serving.domain.exceptions import (
    EmptyRequestError,
    InvalidInferenceLimitsError,
    TooManyWindowsError,
    WindowTooLongError,
)


@dataclass(frozen=True, kw_only=True)
class InferenceLimits:
    """How much one request may ask of the service: windows at once, tokens per window.

    Neither is a property of the model. A graph runs a longer window and returns correct
    numbers, so what bounds a window is what the service is willing to spend on one caller, and
    the ceiling is the service's to set, by configuration.

    Invariants: both limits are positive.

    Attributes:
        max_windows_per_request: How many windows one request may carry.
        max_tokens_per_window: How many tokens one window may tokenise to.
    """

    max_windows_per_request: int
    max_tokens_per_window: int

    def __post_init__(self) -> None:
        for label, count in (
            ("max_windows_per_request", self.max_windows_per_request),
            ("max_tokens_per_window", self.max_tokens_per_window),
        ):
            if count < 1:
                raise InvalidInferenceLimitsError(f"{label} must be positive, got {count}")

    def admit_count(self, windows: int) -> None:
        """Let a request of ``windows`` windows through.

        Raises:
            EmptyRequestError: If it carries no window.
            TooManyWindowsError: If it asks for more than the service admits at once.
        """
        if windows < 1:
            raise EmptyRequestError("a request carries at least one window")
        if windows > self.max_windows_per_request:
            raise TooManyWindowsError(
                f"a request carries at most {self.max_windows_per_request} windows, got {windows}"
            )

    def admit_length(self, tokens: int) -> None:
        """Let a window that tokenises to ``tokens`` tokens through.

        Raises:
            WindowTooLongError: If it is longer than the service admits.
        """
        if tokens > self.max_tokens_per_window:
            raise WindowTooLongError(
                f"a window tokenises to at most {self.max_tokens_per_window} tokens, got {tokens}"
            )
