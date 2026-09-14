"""The rules a masked-reconstruction run is held to, one function per question.

Each rule belongs to the area a failure sends someone to fix (``Area``). A kind of mask is judged
against its matched baseline and against the strongest linear baseline on the same inputs, since
beating each source alone is not beating both at once. The linear baseline decides only where the
noise floor — the measurement noise no predictor can go below — leaves room above it: at the floor
nothing beyond linear is left to learn, and without a stated floor it only informs.
"""

from collections.abc import Sequence

from emblema.pretraining.domain.assessment.check import Check, Rule, Status
from emblema.pretraining.domain.assessment.curve import Curve
from emblema.pretraining.domain.assessment.interval import CONFIDENCE, Verdict
from emblema.pretraining.domain.assessment.kind_summary import MATCHED_BASELINE, KindSummary
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.assessment.spectrum import Spectrum
from emblema.pretraining.domain.assessment.summarised_run import SummarisedRun
from emblema.pretraining.domain.mask_kind import MaskKind
from emblema.pretraining.domain.masking_strategy import MaskingStrategy

# Thresholds are conventions, fixed here before the runs they judge and changed only by editing
# them — a threshold turned until a run passes is a verdict written in advance.
#
# How far the realised share of hidden tokens may stray from the strategy's expectation before it
# is worth a look, and before it is a fault. The expectation is exact only for channels holding
# equally many tokens spread evenly, so a sparse, irregular layout strays a little by design.
HIDDEN_SHARE_TOLERANCE = 0.03
HIDDEN_SHARE_LIMIT = 0.10
# A model's error below this fraction of the noise floor is not a good model but a leak. Not one,
# because the floor is an expectation and the noise of a few thousand tokens scatters about it.
FLOOR_TOLERANCE = 0.8
# A baseline within this factor of the floor has nothing left above it for a model to take.
ROOM_FACTOR = 1.5
# Doubling the epochs lowered the final validation loss by at least this share: the shorter run
# stopped while the model was still learning.
BUDGET_GAIN = 0.05
# Validation loss ending this far above its best: the noise of a learning rate that does not decay
# far enough, or the model beginning to memorise.
ABOVE_BEST = 0.10
# Validation loss this many times the training loss: the model fits windows it will not see again.
GENERALISATION_GAP = 1.5
# Fewer units than this make the bootstrap a formality, fewer tokens make the error a handful.
MIN_UNITS = 10
MIN_TOKENS = 1000
# Channel-windows too short to fit, as a share of all channel-windows hidden whole.
MAX_SKIPPED = 0.25


def loss_falls(curve: Curve) -> Check:
    first, last = curve.validation[0], curve.validation[-1]
    falls = last < first
    return Check(
        Rule.LOSS_FALLS,
        None,
        Status.PASS if falls else Status.FAIL,
        f"{first:.4f} → {last:.4f}",
        "last validation loss below the first",
        "the loss falls"
        if falls
        else "the objective does not learn: check that the loss scores the hidden tokens, that "
        "the gradient reaches the encoder and that the one-batch overfit test passes",
    )


def hidden_share(results: Results) -> Check:
    expected = results.strategy.expected_ratio
    drift = abs(results.realised_ratio - expected)
    status = (
        Status.PASS
        if drift <= HIDDEN_SHARE_TOLERANCE
        else Status.WARN
        if drift <= HIDDEN_SHARE_LIMIT
        else Status.FAIL
    )
    return Check(
        Rule.HIDDEN_SHARE,
        None,
        status,
        f"{results.realised_ratio:.1%} hidden against {expected:.1%} expected",
        f"within {HIDDEN_SHARE_TOLERANCE:.0%}",
        "the masks hide what the strategy prescribes"
        if status is Status.PASS
        else "the masks hide a different share than the strategy prescribes: check the draw "
        "against the layout — blocks over irregular sampling and short channels move it — before "
        "comparing this run with one on another corpus",
    )


def kinds_present(strategy: MaskingStrategy, judged: Sequence[KindSummary]) -> Check:
    drawn = {
        MaskKind.CHANNEL: strategy.channel_rate > 0.0,
        MaskKind.BLOCK: strategy.block_rate > 0.0,
        MaskKind.TOKEN: strategy.token_rate > 0.0,
    }
    seen = {summary.kind for summary in judged}
    missing = [kind.value for kind in MaskKind if drawn[kind] and kind not in seen]
    return Check(
        Rule.KINDS_PRESENT,
        None,
        Status.FAIL if missing else Status.PASS,
        f"no hidden token of {', '.join(missing)}" if missing else "every drawn kind tallied",
        "a tally for every kind the strategy draws",
        "the strategy draws these kinds but the verdict saw none of them: the masks were not "
        "drawn, the tallies were lost on the way, or the layout leaves the kind nothing to hide — "
        "an absent kind cannot be pronounced learnt"
        if missing
        else "every kind the strategy draws reaches the verdict",
    )


def collapsed(judged: Sequence[KindSummary]) -> Check:
    if not judged:
        return Check(Rule.COLLAPSED, None, Status.SKIP, "no kind tallied", "a tallied kind", "—")
    at_the_mean = all(summary.model_error >= summary.mean_error for summary in judged)
    return Check(
        Rule.COLLAPSED,
        None,
        Status.FAIL if at_the_mean else Status.PASS,
        "model at or above the channel mean on every kind"
        if at_the_mean
        else "model below the channel mean on some kind",
        "below the channel mean on at least one kind",
        "the model reads nothing from the window on any kind: the gradient does not reach the "
        "encoder, or the placeholder of a hidden token carries nothing to tell positions apart"
        if at_the_mean
        else "the model uses the window",
    )


def above_floor(summary: KindSummary) -> Check:
    multiple = summary.floor_multiple(summary.model_error)
    if multiple is None:
        return Check(
            Rule.ABOVE_FLOOR,
            summary.kind,
            Status.SKIP,
            "no noise floor",
            "a stated noise level",
            "—",
        )
    below = multiple < FLOOR_TOLERANCE
    return Check(
        Rule.ABOVE_FLOOR,
        summary.kind,
        Status.FAIL if below else Status.PASS,
        f"model at {multiple:.2f}× the floor ({summary.model_error:.4f} against "
        f"{summary.noise_floor:.4f})",
        f"at least {FLOOR_TOLERANCE:g}× the floor",
        "the model predicts measurement noise, which nothing can: hidden values reach it — "
        "through the encoder's input, a feature computed after masking or the decoder — or the "
        "loss scores tokens that were not hidden"
        if below
        else "no sign of a leak",
    )


def converged(
    results: Results, judged: Sequence[KindSummary], shorter: SummarisedRun | None
) -> Check:
    """Whether doubling the budget changed what the run shows.

    Not read off one run's curve: a decaying learning rate flattens the last epochs by
    construction, and a flat end would pass a run that stopped learning because its schedule
    stopped it. It is read off two runs of one configuration instead — the same code, corpus,
    model, strategy, schedule and seed, one with at most half the epochs of the other — and a run
    passes when doubling its budget neither changed a verdict nor lowered the final validation loss
    by more than a few per cent.
    """
    epochs = results.epochs
    if shorter is None:
        half = epochs // 2
        advice = (
            f"run the report with --epochs {half} as well and assess this run again"
            if half
            else f"run the report with --epochs {2 * epochs} and assess that run"
        )
        return Check(
            Rule.CONVERGED,
            None,
            Status.WARN,
            f"no stored run of this configuration with at most {half} epochs"
            if half
            else "one epoch, nothing shorter to compare with",
            "a run of at most half the epochs to compare with",
            "a single run cannot tell a model that stopped learning from a schedule that stopped "
            f"it: {advice}",
        )
    before, after = shorter.results.curve.validation[-1], results.curve.validation[-1]
    gain = (before - after) / before if before > 0.0 else 0.0
    earlier = {summary.kind: summary.verdict for summary in shorter.summaries if not summary.apart}
    changed = [
        summary.kind.value for summary in judged if earlier.get(summary.kind) is not summary.verdict
    ]
    falling = gain >= BUDGET_GAIN
    if changed:
        reading = (
            f"doubling the budget changed the verdict for {', '.join(changed)}: neither run can be "
            "read as final — train longer again, and compare with this run"
        )
    elif falling:
        reading = (
            "doubling the budget changed no verdict but still lowered the loss by more than the "
            "threshold: the model is still learning, so the margins may widen and a verdict near "
            "zero may yet move — train longer again to show it will not"
        )
    else:
        reading = "doubling the budget changed no verdict and little of the loss"
    return Check(
        Rule.CONVERGED,
        None,
        Status.WARN if falling or changed else Status.PASS,
        f"{shorter.results.epochs} → {epochs} epochs: final validation {before:.4f} → {after:.4f} "
        f"({gain:.1%} lower); verdicts "
        + (f"changed for {', '.join(changed)}" if changed else "unchanged"),
        f"less than {BUDGET_GAIN:.0%} lower and no verdict changed",
        reading,
    )


def validation_stable(curve: Curve) -> Check:
    best, last = min(curve.validation), curve.validation[-1]
    above = (last - best) / best if best > 0.0 else 0.0
    unstable = above > ABOVE_BEST
    return Check(
        Rule.VALIDATION_STABLE,
        None,
        Status.WARN if unstable else Status.PASS,
        f"last epoch {above:.1%} above the best",
        f"within {ABOVE_BEST:.0%}",
        "the verdicts are read off a last epoch worse than the run's best: the noise of a learning "
        "rate that does not decay far enough, or the start of memorising — decay further or "
        "regularise. Do not read the verdicts off the best epoch instead: chosen by this "
        "validation loss, it would be judged on the windows that chose it"
        if unstable
        else "the last epoch is close to the best",
    )


def generalisation_gap(curve: Curve) -> Check:
    training, validation = curve.training[-1], curve.validation[-1]
    ratio = validation / training if training > 0.0 else float("inf")
    wide = ratio > GENERALISATION_GAP
    return Check(
        Rule.GENERALISATION_GAP,
        None,
        Status.WARN if wide else Status.PASS,
        f"validation {ratio:.2f}× training",
        f"at most {GENERALISATION_GAP:g}×",
        "the model fits training windows it will not see again: fewer parameters, dropout or "
        "more units"
        if wide
        else "no sign of memorising",
    )


def triviality(summary: KindSummary) -> Check:
    baseline = MATCHED_BASELINE[summary.kind]
    if summary.verdict is Verdict.LEARNT:
        reading = f"the model beats {baseline} beyond the bootstrap's doubt"
    elif summary.verdict is Verdict.MATCHED:
        reading = (
            f"the model and {baseline} cannot be told apart on these units: the kind taught "
            "nothing the baseline shows, or the run is too short or too small to say"
        )
    else:
        reading = f"{baseline} beats the model beyond doubt: the kind taught nothing it shows"
    return Check(
        Rule.TRIVIAL,
        summary.kind,
        Status.FAIL if summary.trivial else Status.PASS,
        f"excess {summary.excess:+.4f}, {CONFIDENCE:.0%} {summary.matched_excess} — "
        f"{summary.verdict.value}",
        "interval above zero",
        reading,
    )


def room(summary: KindSummary) -> Check:
    headroom = summary.floor_multiple(summary.matched_error)
    model = summary.floor_multiple(summary.model_error)
    if headroom is None or model is None:
        return Check(
            Rule.ROOM, summary.kind, Status.SKIP, "no noise floor", "a stated noise level", "—"
        )
    cramped = headroom <= ROOM_FACTOR
    return Check(
        Rule.ROOM,
        summary.kind,
        Status.FAIL if cramped else Status.PASS,
        f"baseline at {headroom:.1f}× the floor, model at {model:.1f}×",
        f"baseline above {ROOM_FACTOR:g}× the floor",
        "the baseline recovers all a predictor can: hiding tokens this way asks for nothing but "
        "the baseline, so lower its rate or change its shape"
        if cramped
        else "the baseline leaves room above the floor for a model to take",
    )


def beats_mean(summary: KindSummary) -> Check:
    beats = summary.model_error < summary.mean_error
    return Check(
        Rule.BEATS_MEAN,
        summary.kind,
        Status.PASS if beats else Status.WARN,
        f"model {summary.model_error:.4f}, channel mean {summary.mean_error:.4f}",
        "model below the channel mean",
        "the model uses the window for this kind"
        if beats
        else "the model predicts this kind no better than the channel mean: on a real corpus a "
        "channel no other channel explains, on the control a kind not learnt yet",
    )


def beyond_linear(summary: KindSummary) -> Check:
    verdict = summary.linear_excess.verdict
    headroom = summary.floor_multiple(summary.linear_error)
    measured = (
        f"linear baseline {summary.linear_error:.4f}, excess "
        f"{summary.linear_error - summary.model_error:+.4f}, {CONFIDENCE:.0%} "
        f"{summary.linear_excess} — {verdict.value}"
        + ("" if headroom is None else f"; baseline at {headroom:.1f}× the floor")
    )
    expected = f"interval above zero, where the baseline is above {ROOM_FACTOR:g}× the floor"
    if headroom is not None and headroom <= ROOM_FACTOR:
        return Check(
            Rule.BEYOND_LINEAR,
            summary.kind,
            Status.SKIP,
            measured,
            expected,
            "the linear baseline already recovers all a predictor can: nothing beyond linear is "
            "left to learn on this corpus, and the kind is judged by its matched baseline alone",
        )
    if verdict is Verdict.LEARNT:
        return Check(
            Rule.BEYOND_LINEAR,
            summary.kind,
            Status.PASS,
            measured,
            expected,
            "the model beats a linear regression on the channel's own line and the other channels "
            "together",
        )
    if headroom is None:
        return Check(
            Rule.BEYOND_LINEAR,
            summary.kind,
            Status.WARN,
            measured,
            expected,
            "a linear regression on the channel's own line and the other channels together does "
            "as well; without a noise floor it cannot be told whether nothing beyond linear is "
            "left to learn or the model falls short, so this informs and does not decide",
        )
    return Check(
        Rule.BEYOND_LINEAR,
        summary.kind,
        Status.FAIL,
        measured,
        expected,
        "a linear regression on the channel's own line and the other channels together does as "
        "well, and leaves room above the floor: what this kind teaches, linear algebra on the same "
        "inputs knows too",
    )


def evidence(summary: KindSummary) -> Check:
    thin = summary.units < MIN_UNITS or summary.tokens < MIN_TOKENS
    return Check(
        Rule.EVIDENCE,
        summary.kind,
        Status.WARN if thin else Status.PASS,
        f"{summary.tokens:,} tokens over {summary.units} units",
        f"at least {MIN_TOKENS:,} tokens over {MIN_UNITS} units",
        "too little to decide on: more validation units, or a higher rate for this kind"
        if thin
        else "enough to decide on",
    )


def spectrum_informative(spectrum: Spectrum) -> Check:
    if spectrum.fitted == 0:
        return Check(
            Rule.SPECTRUM_INFORMATIVE,
            None,
            Status.FAIL,
            "no channel-window fitted",
            "at least one",
            "no channel hidden whole had tokens enough to fit: the spectrum says nothing",
        )
    informative = spectrum.informative()
    blind = len(informative) <= 1
    return Check(
        Rule.SPECTRUM_INFORMATIVE,
        None,
        Status.WARN if blind else Status.PASS,
        f"{spectrum.shares()[0]:.1%} of the energy at 1 cycle; informative at "
        f"{', '.join(map(str, informative)) or 'none'} of {len(spectrum.truth)}",
        "signal at two or more fitted frequencies",
        "the truth has energy at one frequency at most, so the diagnostic cannot tell a model "
        "that learnt smoothness from one that learnt structure: every higher frequency holds "
        "noise alone. It needs a corpus whose signal completes several cycles within a window, "
        "such as the spectral probe"
        if blind
        else "the truth spreads over several frequencies",
    )


def spectrum_fitted(spectrum: Spectrum) -> Check:
    total = spectrum.fitted + spectrum.skipped
    skipped = spectrum.skipped / total if total else 0.0
    thin = skipped > MAX_SKIPPED
    return Check(
        Rule.SPECTRUM_FITTED,
        None,
        Status.WARN if thin else Status.PASS,
        f"{spectrum.skipped} of {total} channel-windows too short to fit",
        f"at most {MAX_SKIPPED:.0%}",
        "the spectrum describes the densest channels only"
        if thin
        else "the spectrum covers the channels hidden whole",
    )
