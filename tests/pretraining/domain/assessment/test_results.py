from dataclasses import replace

from emblema.pretraining.domain.assessment.curve import Curve
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.assessment.spectrum import Spectrum
from emblema.pretraining.domain.masking_strategy import MaskingStrategy

STRATEGY = MaskingStrategy(channel_rate=0.15, block_rate=0.6, block_span=0.5, token_rate=0.1)


def results(**settings: str) -> Results:
    return Results(
        settings={"corpus": "control-a", "code": "abc", "date": "2026-09-15 23:04:11", **settings},
        strategy=STRATEGY,
        realised_ratio=0.46,
        curve=Curve((0.6, 0.5), (0.6, 0.5), (1.0, 1.0)),
        tallies=(),
        spectrum=Spectrum((1.0,), (0.5,), (0.5,), fitted=4, skipped=0),
    )


def test_runs_differing_only_in_when_they_ran_and_what_they_are_called_are_one() -> None:
    first = results(tracked_run="control-a-20260915-230411", backbone_checksum="sha256:aa")
    second = results(
        date="2026-09-16 08:00:00",
        tracked_run="control-a-20260916-080000",
        backbone_checksum="sha256:bb",
    )

    assert first.configuration_of(second)


def test_a_run_stored_before_it_named_its_tracked_run_is_still_a_run_of_its_configuration() -> None:
    assert results(tracked_run="control-a-20260915-230411").configuration_of(results())


def test_runs_that_differ_in_anything_they_were_configured_with_are_not_one_configuration() -> None:
    assert not results(code="abc").configuration_of(results(code="def"))
    assert not results().configuration_of(
        replace(results(), strategy=replace(STRATEGY, token_rate=0.2))
    )
