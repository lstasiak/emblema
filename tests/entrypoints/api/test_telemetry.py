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


def test_the_inference_budget_is_reported_as_it_is_when_scraped() -> None:
    telemetry = Telemetry(TELEMETRY)
    held = [30]
    try:
        telemetry.observe_budget(capacity=100, in_use=lambda: held[0], waiting=lambda: 2)

        first, _ = telemetry.exposition()
        held[0] = 60
        second, _ = telemetry.exposition()
    finally:
        telemetry.shutdown()

    assert b"emblema_inference_budget_capacity" in first
    assert b"emblema_inference_budget_in_use" in first
    assert b"emblema_inference_batches_waiting" in first
    assert b"} 30.0" in first
    assert b"} 60.0" in second
