import time
from collections.abc import Sequence

from emblema.catalog.contracts.observed_window import ObservedWindow
from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.window_tokeniser import WindowTokeniser
from emblema.entrypoints.api.telemetry.telemetry import Telemetry
from emblema.shared.kernel.tokens import TokenWindow


class InstrumentedWindowTokeniser:
    """The Catalog's tokeniser as the process composes it, each window traced and timed.

    Tokenisation is pure Python and the one step of an answer whose cost grows with what the
    caller sent, so it is timed on its own: a latency budget that tokenisation outgrows is the
    signal to vectorise it.
    """

    def __init__(self, tokeniser: WindowTokeniser, telemetry: Telemetry) -> None:
        self._tokeniser = tokeniser
        self._telemetry = telemetry

    def tokenise(
        self, window: ObservedWindow, corpus: str, channels: Sequence[PublishedChannel]
    ) -> TokenWindow:
        with self._telemetry.tracer.start_as_current_span(
            "tokenisation.window",
            attributes={"corpus": corpus, "readings": len(window.observations)},
        ):
            started = time.perf_counter()
            tokens = self._tokeniser.tokenise(window, corpus, channels)
            self._telemetry.tokenisation_seconds.record(
                time.perf_counter() - started, {"corpus": corpus}
            )
            return tokens
