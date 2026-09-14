"""The assessment decides what a run is worth, so each of its rules is checked from both sides.

A rule that can only pass is decoration and a rule that can only fail is noise. Every rule is
exercised here on numbers that should earn its verdict and on numbers that should not, and the
decision is held to the precedence it states. The tallies are summarised by the bootstrap the
report uses, so the intervals the rules read are the ones a real run gets.
"""

from dataclasses import replace

import pytest

from emblema.pretraining.adapters.diagnostics.unit_bootstrap import UnitBootstrap
from emblema.pretraining.application.use_cases.assess_reconstruction_run import (
    AssessReconstructionRun,
)
from emblema.pretraining.domain.assessment.assessment import Assessment
from emblema.pretraining.domain.assessment.check import Area, Check, Status
from emblema.pretraining.domain.assessment.decision import Outcome
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.exceptions import IncomparableRunsError
from emblema.pretraining.domain.mask_kind import MaskKind
from tests.support.reconstruction_runs import (
    LEVELLED,
    SETTINGS,
    STRATEGY,
    epochs_of,
    half_of,
    learnt_everywhere,
    results,
    tallies,
    with_token,
)

assess = AssessReconstructionRun(UnitBootstrap())


def converged(run: Results) -> Assessment:
    return assess(run, half_of(run))


def check(found: Assessment, name: str) -> Check:
    return next(item for item in found.checks if item.name == name)


def test_a_clean_run_that_doubling_did_not_change_is_accepted_with_nothing_to_do() -> None:
    found = converged(results())

    assert {item.status for item in found.checks} == {Status.PASS}
    assert found.decision.outcome is Outcome.ACCEPT
    assert found.decision.actions == ()
    assert found.triviality_negative


def test_a_model_below_the_noise_floor_is_a_leak() -> None:
    found = converged(results(unit_tallies=with_token(model=0.001, matched=0.04)))

    assert check(found, "above-floor-token").status is Status.FAIL
    assert check(found, "above-floor-block").status is Status.PASS
    assert found.decision.outcome is Outcome.CHANGE_LOGIC


def test_a_model_at_the_mean_on_every_kind_has_collapsed_but_on_one_kind_only_warns() -> None:
    collapsed = converged(results(unit_tallies=learnt_everywhere(mean=0.01)))
    partly = converged(results(unit_tallies=with_token(model=0.05, matched=0.3, mean=0.05)))

    assert check(collapsed, "collapsed").status is Status.FAIL
    assert collapsed.decision.outcome is Outcome.CHANGE_LOGIC
    assert check(partly, "collapsed").status is Status.PASS
    assert check(partly, "beats-mean-token").status is Status.WARN
    assert check(partly, "beats-mean-token").area is Area.MASKING
    assert partly.decision.outcome is Outcome.ACCEPT


def test_a_loss_that_does_not_fall_is_a_fault() -> None:
    found = assess(results(validation=(0.5, 0.6, 0.6, 0.6)))

    assert check(found, "loss-falls").status is Status.FAIL
    assert found.decision.outcome is Outcome.CHANGE_LOGIC


def test_the_hidden_share_is_held_to_the_strategy() -> None:
    assert check(assess(results(realised=0.45)), "hidden-share").status is Status.PASS
    assert check(assess(results(realised=0.40)), "hidden-share").status is Status.WARN
    assert check(assess(results(realised=0.30)), "hidden-share").status is Status.FAIL


def test_a_run_that_tallied_nothing_is_never_accepted() -> None:
    found = converged(results(unit_tallies=[]))

    assert check(found, "kinds-present").status is Status.FAIL
    assert check(found, "collapsed").status is Status.SKIP
    assert found.decision.outcome is Outcome.CHANGE_LOGIC
    assert not found.triviality_negative


def test_a_kind_the_strategy_does_not_draw_is_not_missed() -> None:
    no_tokens = replace(STRATEGY, token_rate=0.0)
    without_token = [tally for tally in learnt_everywhere() if tally.kind is not MaskKind.TOKEN]

    drawn = converged(results(unit_tallies=without_token))
    undrawn = converged(results(unit_tallies=without_token, strategy=no_tokens))

    assert check(drawn, "kinds-present").status is Status.FAIL
    assert "token" in check(drawn, "kinds-present").measured
    assert check(undrawn, "kinds-present").status is Status.PASS
    assert undrawn.decision.outcome is Outcome.ACCEPT


def test_one_run_alone_cannot_show_it_stopped_learning() -> None:
    found = assess(results())

    assert check(found, "converged").status is Status.WARN
    assert "--epochs 4" in check(found, "converged").reading
    assert found.decision.outcome is Outcome.ACCEPT
    assert "not shown" in found.decision.headline


def test_doubling_that_still_lowers_the_loss_is_not_convergence() -> None:
    run = results(validation=(0.5, 0.4, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08))

    found = assess(run, epochs_of(run, 4))

    assert check(found, "converged").status is Status.WARN
    assert "(60.0% lower)" in check(found, "converged").measured
    assert "changed no verdict but still lowered the loss" in check(found, "converged").reading


def test_doubling_that_changes_a_verdict_is_not_convergence() -> None:
    run = results()
    shorter = replace(half_of(run), tallies=tuple(with_token(model=0.06, matched=0.04)))

    found = assess(run, shorter)

    assert check(found, "converged").status is Status.WARN
    assert "changed for token" in check(found, "converged").measured
    assert "changed the verdict for token" in check(found, "converged").reading


def test_only_a_shorter_run_of_the_same_configuration_is_compared_with() -> None:
    run = results()
    other_seed = replace(half_of(run), settings={**SETTINGS, "seed": "2"})
    other_day = replace(half_of(run), settings={**SETTINGS, "date": "2026-09-12 10:00:00"})

    with pytest.raises(IncomparableRunsError, match="same configuration"):
        assess(run, other_seed)
    with pytest.raises(IncomparableRunsError, match="at most half the epochs"):
        assess(run, epochs_of(run, 5))
    assert check(assess(run, other_day), "converged").status is Status.PASS


def test_a_last_epoch_well_above_the_best_warns_without_advising_the_best_epoch() -> None:
    noisy = converged(results(validation=(0.5, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.13)))

    assert check(noisy, "validation-stable").status is Status.WARN
    assert (
        "Do not read the verdicts off the best epoch" in check(noisy, "validation-stable").reading
    )
    assert check(converged(results()), "validation-stable").status is Status.PASS


def test_a_validation_loss_far_above_training_is_flagged() -> None:
    memorised = tuple(value / 3 for value in LEVELLED)

    found = assess(results(training=memorised))

    assert check(found, "generalisation-gap").status is Status.WARN


def test_a_trivial_kind_not_shown_converged_asks_for_a_longer_run() -> None:
    found = assess(results(unit_tallies=with_token(model=0.06, matched=0.04)))

    assert check(found, "trivial-token").status is Status.FAIL
    assert check(found, "room-token").status is Status.PASS
    assert found.decision.outcome is Outcome.RECALIBRATE
    assert "not shown to have stopped learning" in found.decision.headline
    assert not found.triviality_negative


def test_a_kind_its_units_cannot_tell_from_the_baseline_is_trivial_too() -> None:
    scattered = [
        *tallies(MaskKind.CHANNEL, model=0.1, matched=0.3),
        *tallies(MaskKind.BLOCK, model=0.05, matched=0.25),
        *tallies(MaskKind.TOKEN, model=0.04, matched=0.04, spread=0.03),
    ]

    found = converged(results(unit_tallies=scattered))

    assert check(found, "trivial-token").status is Status.FAIL
    assert "cannot be told apart" in check(found, "trivial-token").reading
    assert found.decision.outcome is Outcome.CHANGE_MASKING


def test_a_trivial_kind_in_a_converged_run_condemns_the_kind() -> None:
    found = converged(results(unit_tallies=with_token(model=0.06, matched=0.04)))

    assert found.decision.outcome is Outcome.CHANGE_MASKING
    assert "trivial-token" in found.decision.headline


def test_a_baseline_at_the_floor_condemns_the_kind_however_long_the_training() -> None:
    found = assess(results(unit_tallies=with_token(model=0.013, matched=0.012)))

    assert check(found, "room-token").status is Status.FAIL
    assert found.decision.outcome is Outcome.CHANGE_MASKING


def test_a_fault_in_the_implementation_outranks_everything_else() -> None:
    broken = [
        *tallies(MaskKind.CHANNEL, model=0.001, matched=0.3),
        *tallies(MaskKind.BLOCK, model=0.05, matched=0.25),
        *tallies(MaskKind.TOKEN, model=0.013, matched=0.012),
    ]

    found = assess(results(unit_tallies=broken))

    assert found.decision.outcome is Outcome.CHANGE_LOGIC
    assert found.decision.actions[0].startswith("[implementation] above-floor-channel")


def test_without_a_noise_floor_the_rules_needing_one_step_aside() -> None:
    found = converged(results(unit_tallies=learnt_everywhere(floor=None)))

    assert check(found, "above-floor-token").status is Status.SKIP
    assert check(found, "room-token").status is Status.SKIP
    assert found.decision.outcome is Outcome.ACCEPT


def test_few_units_are_too_little_evidence() -> None:
    thin = [
        *tallies(MaskKind.CHANNEL, model=0.1, matched=0.3, units=3),
        *tallies(MaskKind.BLOCK, model=0.05, matched=0.25),
        *tallies(MaskKind.TOKEN, model=0.02, matched=0.04),
    ]

    found = assess(results(unit_tallies=thin))

    assert check(found, "evidence-channel").status is Status.WARN
    assert check(found, "evidence-block").status is Status.PASS


def test_channels_apart_are_summarised_but_never_judged() -> None:
    aside = [*learnt_everywhere(), *tallies(MaskKind.CHANNEL, model=1.3, matched=0.3, apart=True)]

    found = converged(results(unit_tallies=aside))

    assert [summary.apart for summary in found.summaries] == [False, False, False, True]
    assert found.decision.outcome is Outcome.ACCEPT


def test_a_model_no_better_than_linear_algebra_on_the_same_inputs_is_trivial_too() -> None:
    linear = with_token(model=0.02, matched=0.04, linear=0.02)

    converged_run = converged(results(unit_tallies=linear))
    single_run = assess(results(unit_tallies=linear))

    assert check(converged_run, "trivial-token").status is Status.PASS
    assert check(converged_run, "beyond-linear-token").status is Status.FAIL
    assert check(converged_run, "beyond-linear-block").status is Status.PASS
    assert not any(item.name == "beyond-linear-channel" for item in converged_run.checks)
    assert converged_run.decision.outcome is Outcome.CHANGE_MASKING
    assert "beyond-linear-token" in converged_run.decision.headline
    assert not converged_run.triviality_negative
    assert single_run.decision.outcome is Outcome.RECALIBRATE


def test_a_linear_baseline_at_the_floor_leaves_nothing_beyond_linear_to_judge() -> None:
    # The linear baseline already sits at the noise floor: nothing is left above it for a model
    # to take, so falling short of it condemns neither the model nor the strategy.
    cramped = with_token(model=0.013, matched=0.04, linear=0.012)

    found = converged(results(unit_tallies=cramped))

    assert check(found, "beyond-linear-token").status is Status.SKIP
    assert "1.2× the floor" in check(found, "beyond-linear-token").measured
    assert found.decision.outcome is Outcome.ACCEPT
    assert found.triviality_negative


def test_without_a_noise_floor_the_linear_baseline_informs_and_does_not_decide() -> None:
    unstated = [
        *tallies(MaskKind.CHANNEL, model=0.1, matched=0.3, floor=None),
        *tallies(MaskKind.BLOCK, model=0.05, matched=0.25, floor=None),
        *tallies(MaskKind.TOKEN, model=0.02, matched=0.04, linear=0.02, floor=None),
    ]

    found = converged(results(unit_tallies=unstated))

    assert check(found, "beyond-linear-token").status is Status.WARN
    assert found.decision.outcome is Outcome.ACCEPT


def test_a_spectrum_with_energy_at_one_frequency_cannot_tell_smoothness_from_structure() -> None:
    one = assess(results(truth=(1000.0, 3.0, 1.0)))
    several = assess(results(truth=(100.0, 20.0, 5.0)))
    empty = assess(results(fitted=0))

    assert check(one, "spectrum-informative").area is Area.DIAGNOSTIC
    assert check(one, "spectrum-informative").status is Status.WARN
    assert check(several, "spectrum-informative").status is Status.PASS
    assert check(empty, "spectrum-informative").status is Status.FAIL


def test_a_spectrum_that_skips_most_channels_is_flagged() -> None:
    found = assess(results(fitted=10, skipped=40))

    assert check(found, "spectrum-fitted").status is Status.WARN
