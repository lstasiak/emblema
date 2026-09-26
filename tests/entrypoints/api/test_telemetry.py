"""What the process reports about itself, read the way a scraper reads it."""

from emblema.entrypoints.api.telemetry.telemetry import Telemetry
from tests.support.settings import TELEMETRY


def test_metrics_are_exposed_in_the_format_a_scraper_reads() -> None:
    telemetry = Telemetry(TELEMETRY)
    try:
        telemetry.windows_answered.add(3, {"answer": "prediction"})

        body, media_type = telemetry.exposition()
    finally:
        telemetry.shutdown()

    assert media_type.startswith("text/plain")
    assert b"emblema_windows_answered_total" in body
    assert b'answer="prediction"' in body
