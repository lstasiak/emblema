"""What the task reads and what the states carry, off arrays stored as the storing step stores them.

Nothing here reaches a registry, a store or an accelerator: a directory is written in the storing
step's shape, with an encoder whose states carry one summary and the outcome and two encoders
that carry nothing, and every reading is checked against what it must be.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import scripts.stay_summary_report as report  # noqa: E402
from emblema.evaluation.adapters.features.window_statistics import (  # noqa: E402
    WindowStatistics,
)
from emblema.evaluation.adapters.torch.ridge_solution import RidgeSolution  # noqa: E402
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties  # noqa: E402
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import (  # noqa: E402
    PairedUnitBootstrap,
)
from scripts.campaign_pairs_report import (  # noqa: E402
    CELLS,
    PREDICTIONS,
    SCORES,
    Answers,
    ScoreRow,
    Side,
    pair,
    read_cells,
    read_predictions,
    write_scores,
)
from scripts.stay_summary_report import (  # noqa: E402
    CHANNELS,
    CONDITIONS,
    DECODING,
    DRAWS,
    ENCODERS,
    FAMILY_OF,
    FITS,
    LEARNT,
    LEVEL,
    MINIMUM_STAYS,
    PRETRAINED,
    REGISTERED,
    REPORT,
    SCORED,
    STATES,
    STATES_ALONE,
    STATES_AND_SUMMARIES,
    STATICS,
    STAYS,
    SUMMARIES,
    SUMMARIES_ALONE,
    SUMMARIES_SOURCE,
    SUMMARY_CHANNELS,
    SUMMARY_STATISTICS,
    UNTRAINED,
    UNTRAINED_PREFIX,
    JointRidge,
    Reading,
    Stored,
    StoredChannel,
    StoredDraw,
    StoredEncoder,
    StoredStay,
    decode,
    decoded_targets,
    family_of,
    logistic_answers,
    main,
    prepared,
    probe,
    render,
    render_fidelity,
    save_arrays,
    write_probed,
    write_rows,
)

pytestmark = pytest.mark.ml

PENALTIES = (0.1, 1.0, 10.0)
LEARNT_STAYS, SCORED_STAYS = 160, 120
BUDGETS = ("60", "100")
SEEDS = (1, 2)
TIMED, STATIC = 1, 2
CAMPAIGNS = {
    "mix1": "11111111-1111-1111-1111-111111111111",
    "mix2": "22222222-2222-2222-2222-222222222222",
}
INFORMED, FRESH = "informed", ("fresh1", "fresh2")
WIDTH = 6


def _stays(generator: np.random.Generator) -> tuple[list[StoredStay], np.ndarray]:
    latent = generator.normal(size=LEARNT_STAYS + SCORED_STAYS)
    targets = (latent + 0.5 * generator.normal(size=latent.size) > 0.8).astype(float)
    stays = [
        StoredStay(
            row=row,
            half=LEARNT if row < LEARNT_STAYS else SCORED,
            unit=f"set-a/{row:04d}",
            position=row,
            ends_at=48.0,
            target=float(targets[row]),
        )
        for row in range(latent.size)
    ]
    return stays, latent


def _summaries(generator: np.random.Generator, latent: np.ndarray) -> np.ndarray:
    """A timed channel whose mean is the latent, every other summary noise; a static's value."""
    width = len(WindowStatistics.NAMES)
    rows = np.full((latent.size, 2 * width), np.nan)
    rows[:, :width] = generator.normal(size=(latent.size, width))
    rows[:, WindowStatistics.NAMES.index("mean")] = latent
    rows[:, WindowStatistics.NAMES.index("count")] = generator.integers(2, 40, size=latent.size)
    # A stay with one reading of the channel has no slope.
    rows[::7, WindowStatistics.NAMES.index("slope")] = np.nan
    rows[:, width + WindowStatistics.NAMES.index("count")] = 1.0
    rows[:, width + WindowStatistics.NAMES.index("mean")] = generator.normal(size=latent.size)
    return rows


def _draws(stays: list[StoredStay]) -> list[StoredDraw]:
    learnt = [stay for stay in stays if stay.half == LEARNT]
    draws: list[StoredDraw] = []
    for budget in BUDGETS:
        for seed in SEEDS:
            chosen = np.random.default_rng(seed).permutation(len(learnt))[: int(budget)]
            draws.extend(
                StoredDraw(
                    budget=budget, seed=seed, unit=learnt[i].unit, position=learnt[i].position
                )
                for i in sorted(chosen)
            )
    return draws


def _encoders() -> list[StoredEncoder]:
    """Both campaigns read the same pretrained weights; the untrained draw differs per seed."""
    found = []
    for source, campaign in CAMPAIGNS.items():
        for seed in SEEDS:
            found.append(
                StoredEncoder(
                    source=source, campaign=campaign, arm=PRETRAINED, seed=seed, digest=INFORMED
                )
            )
            found.append(
                StoredEncoder(
                    source=source,
                    campaign=campaign,
                    arm=UNTRAINED,
                    seed=seed,
                    digest=FRESH[seed - 1],
                )
            )
    return found


def stored_directory(directory: Path) -> Path:
    generator = np.random.default_rng(7)
    stays, latent = _stays(generator)
    informed = np.column_stack(
        [latent + 0.05 * generator.normal(size=latent.size)]
        + [generator.normal(size=latent.size) for _ in range(WIDTH - 1)]
    ).astype(np.float32)
    arrays = {
        INFORMED: informed,
        FRESH[0]: generator.normal(size=(latent.size, WIDTH)).astype(np.float32),
        FRESH[1]: generator.normal(size=(latent.size, WIDTH)).astype(np.float32),
        SUMMARIES: _summaries(generator, latent),
        SUMMARY_CHANNELS: np.array([TIMED] * 10 + [STATIC] * 10),
        SUMMARY_STATISTICS: np.array(list(WindowStatistics.NAMES) * 2),
    }
    save_arrays(directory / STATES, arrays)
    write_rows(directory / STAYS, StoredStay.COLUMNS, (stay.record() for stay in stays))
    write_rows(directory / DRAWS, StoredDraw.COLUMNS, (d.record() for d in _draws(stays)))
    write_rows(directory / ENCODERS, StoredEncoder.COLUMNS, (e.record() for e in _encoders()))
    channels = [
        StoredChannel(channel_id=TIMED, name="physionet2012/HR", timeless=False),
        StoredChannel(channel_id=STATIC, name="physionet2012/Age", timeless=True),
    ]
    write_rows(directory / CHANNELS, StoredChannel.COLUMNS, (c.record() for c in channels))
    conditions = {
        "commit": "abc1234",
        "device": "cpu",
        "torch": torch.__version__,
        "tier": "M",
        "penalties": list(PENALTIES),
        "campaigns": CAMPAIGNS,
    }
    (directory / CONDITIONS).write_text(json.dumps(conditions), encoding="utf-8")
    return directory


@pytest.fixture
def stored(tmp_path: Path) -> Stored:
    return Stored.read(stored_directory(tmp_path))


def test_every_summary_has_a_family_and_a_static_keeps_its_value_and_its_presence() -> None:
    assert set(FAMILY_OF) == set(WindowStatistics.NAMES)
    assert family_of("mean", timeless=False) == LEVEL
    assert family_of("mean", timeless=True) == STATICS
    assert family_of("count", timeless=True) == STATICS
    assert family_of("slope", timeless=True) is None


def test_a_statistic_the_summaries_do_not_hold_has_no_family() -> None:
    with pytest.raises(ValueError, match="median"):
        family_of("median", timeless=False)


def test_a_joint_fit_is_the_closed_form_ridge_of_every_target_alone() -> None:
    generator = np.random.default_rng(3)
    states = generator.normal(size=(80, 5))
    states[:, 4] = 2.0  # a constant column, which the scaling leaves alone
    targets = np.column_stack(
        [states[:, 0] * 3.0 + generator.normal(size=80), generator.normal(size=80) * 10.0]
    )
    penalties = RidgePenalties((0.01, 1.0, 100.0, 1e4))

    joint = JointRidge.fitted(states, targets, penalties)

    for index in range(targets.shape[1]):
        alone = RidgeSolution.fitted(
            torch.from_numpy(states), torch.from_numpy(targets[:, index]), penalties
        )
        assert joint.penalties[index] == alone.penalty
        np.testing.assert_allclose(joint.weights[:, index], alone.weights.numpy(), atol=1e-10)
        assert joint.intercepts[index] == pytest.approx(alone.intercept, abs=1e-10)


def test_a_joint_fit_needs_two_rows_to_leave_one_out() -> None:
    with pytest.raises(ValueError, match="two rows"):
        JointRidge.fitted(np.ones((1, 3)), np.ones((1, 1)), RidgePenalties((1.0,)))


def test_summaries_are_filled_from_the_draw_alone() -> None:
    sample = np.array([[1.0, np.nan, 5.0, np.nan], [3.0, 4.0, 5.0, np.nan], [np.nan, 6.0, 5.0, 1]])
    scored = np.array([[np.nan, np.nan, 9.0, 2.0], [100.0, 100.0, 9.0, 2.0]])

    filled_sample, filled_scored = prepared(sample, scored)

    # The constant third column and the fourth, defined once, go; the gaps take the draw's mean.
    np.testing.assert_array_equal(filled_sample, [[1.0, 5.0], [3.0, 4.0], [2.0, 6.0]])
    np.testing.assert_array_equal(filled_scored, [[2.0, 5.0], [100.0, 100.0]])


def test_a_draw_that_names_a_stay_off_the_learnt_half_is_refused(stored: Stored) -> None:
    scored = stored.half(SCORED)[0]
    broken = Stored(
        stays=stored.stays,
        draws=(StoredDraw(budget="60", seed=1, unit=scored.unit, position=scored.position),),
        encoders=stored.encoders,
        channels=stored.channels,
        conditions=stored.conditions,
        arrays=stored.arrays,
    )

    with pytest.raises(ValueError, match="no stored stay of the learnt half"):
        broken.drawn("60", 1)


def test_the_reading_over_states_alone_is_the_probes_head_over_them(stored: Stored) -> None:
    probed = probe(stored)

    drawn = stored.drawn("60", 2)
    states = stored.states(stored.encoder("mix1", UNTRAINED, 2).digest).astype(np.float64)
    expected, _ = logistic_answers(
        states[[stay.row for stay in drawn]],
        np.array([stay.target for stay in drawn]),
        states[[stay.row for stay in stored.half(SCORED)]],
        stored.penalties,
    )
    answered = [
        row.predicted
        for row in probed.predictions
        if (row.campaign, row.candidate, row.budget, row.seed)
        == ("mix1", UNTRAINED_PREFIX + STATES_ALONE, "60", 2)
    ]
    np.testing.assert_array_equal(answered, expected)


def test_one_encoder_under_two_campaigns_is_fitted_once_and_filed_under_both(
    stored: Stored,
) -> None:
    probed = probe(stored)

    by_source = {
        fit.source: fit
        for fit in probed.fits
        if (fit.candidate, fit.budget, fit.seed) == (STATES_AND_SUMMARIES, "100", 1)
    }
    assert set(by_source) == set(CAMPAIGNS)
    assert by_source["mix1"].auroc == by_source["mix2"].auroc
    assert by_source["mix1"].seconds == by_source["mix2"].seconds


def test_the_informed_states_read_the_outcome_and_the_untrained_ones_do_not(
    stored: Stored,
) -> None:
    fits = probe(stored).fits

    def area(candidate: str) -> float:
        return float(
            np.mean([f.auroc for f in fits if (f.source, f.candidate) == ("mix1", candidate)])
        )

    assert area(STATES_ALONE) > 0.75
    assert area(UNTRAINED_PREFIX + STATES_ALONE) < 0.65


def test_the_written_files_are_paired_by_the_stages_pairing(stored: Stored, tmp_path: Path) -> None:
    probed = probe(
        stored,
        [
            Reading(source=SUMMARIES_SOURCE, candidate=SUMMARIES_ALONE, arm=None, summaries=True),
            Reading(source="mix1", candidate=STATES_ALONE, arm=PRETRAINED, summaries=False),
        ],
    )
    write_probed(tmp_path, probed)

    comparison = pair(
        read_cells(tmp_path / CELLS),
        Side(SUMMARIES_SOURCE, SUMMARIES_ALONE, "100"),
        Side("mix1", STATES_ALONE, "100"),
        bootstrap=PairedUnitBootstrap(resamples=200, seed=1, level=0.95),
        floor_share=0.02,
        floor_area=0.01,
        answers=Answers(read_predictions(tmp_path / PREDICTIONS)),
    )
    assert comparison.repeats == len(SEEDS)
    assert comparison.units == SCORED_STAYS


def test_a_summary_the_states_carry_is_recovered_and_noise_is_not(stored: Stored) -> None:
    decoded = decode(stored, decoded_targets(stored))

    def r2(digest: str, statistic: str) -> float:
        [row] = [
            row
            for row in decoded
            if (row.digest, row.channel_id, row.statistic) == (digest, TIMED, statistic)
        ]
        return row.r2

    assert r2(INFORMED, "mean") > 0.95
    assert r2(FRESH[0], "mean") < 0.1
    assert r2(INFORMED, "deviation") < 0.1


def test_only_summaries_with_a_family_and_enough_varying_stays_are_decoded(
    stored: Stored,
) -> None:
    columns = stored.summary_columns()

    kept = {columns[index] for index in decoded_targets(stored)}

    names = {(channel.channel_id, statistic) for channel, statistic in kept}
    # Every timed summary; of the static only its value, since its count never varies.
    assert names == {(TIMED, name) for name in WindowStatistics.NAMES} | {(STATIC, "mean")}
    assert decoded_targets(stored, channels=[STATIC]) == [10 + 1]
    assert MINIMUM_STAYS < SCORED_STAYS


def test_the_reading_over_states_is_set_beside_the_registered_cells(
    stored: Stored, tmp_path: Path
) -> None:
    fits = probe(stored).fits
    rows = [
        ScoreRow(
            campaign=CAMPAIGNS[fit.source],
            purpose="selection",
            measure="auroc_shortfall",
            candidate=REGISTERED[
                UNTRAINED if fit.candidate.startswith(UNTRAINED_PREFIX) else PRETRAINED
            ],
            budget=fit.budget,
            seed=fit.seed,
            units=SCORED_STAYS,
            score=fit.auroc + (0.001 if fit.seed == 2 else 0.0),
            seconds=0.0,
        )
        for fit in fits
        if fit.candidate in (STATES_ALONE, UNTRAINED_PREFIX + STATES_ALONE)
    ]
    write_scores(tmp_path / SCORES, rows)

    _, largest = render_fidelity(stored, fits, tmp_path)
    assert largest == pytest.approx(0.001)

    write_scores(tmp_path / SCORES, rows[1:])
    with pytest.raises(ValueError, match="holds no cell"):
        render_fidelity(stored, fits, tmp_path)


def test_a_report_renders_again_from_its_files_alone(tmp_path: Path) -> None:
    directory = stored_directory(tmp_path)
    main([str(directory)])
    first = (directory / REPORT).read_text(encoding="utf-8")
    assert (directory / FITS).is_file()
    assert (directory / DECODING).is_file()

    main([str(directory), "--report-only"])

    assert (directory / REPORT).read_text(encoding="utf-8") == first
    assert "Median R²" in render(Stored.read(directory), [], [], None)


def test_a_held_out_stay_far_from_every_learnt_state_is_answered_within_the_learnt_range(
    stored: Stored,
) -> None:
    far = dict(stored.arrays)
    states = np.array(far[INFORMED], copy=True)
    states[stored.half(SCORED)[0].row] = 1e4
    far[INFORMED] = states

    decoded = decode(replace(stored, arrays=far), decoded_targets(stored, channels=[TIMED]))

    [mean] = [row for row in decoded if (row.digest, row.statistic) == (INFORMED, "mean")]
    # One stay of a hundred and twenty answered at the edge of the learnt range, not at 10⁴.
    assert mean.r2 > 0.5


def test_a_decomposition_that_does_not_converge_is_taken_by_the_one_sided_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = np.random.default_rng(5)
    states = generator.normal(size=(60, 4))
    targets = (states[:, :1] * 2.0 + generator.normal(size=(60, 1))).astype(np.float64)
    penalties = RidgePenalties((0.1, 10.0))
    divided = JointRidge.fitted(states, targets, penalties)

    def diverged(*_: object, **__: object) -> None:
        raise RuntimeError("linalg.svd: The algorithm failed to converge")

    monkeypatch.setattr(torch.linalg, "svd", diverged)
    one_sided = JointRidge.fitted(states, targets, penalties)

    assert divided.decomposition == JointRidge.DIVIDE_AND_CONQUER
    assert one_sided.decomposition == JointRidge.ONE_SIDED
    np.testing.assert_allclose(one_sided.weights, divided.weights, atol=1e-10)
    np.testing.assert_array_equal(one_sided.penalties, divided.penalties)


def test_the_heads_are_kept_when_the_mapping_after_them_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = stored_directory(tmp_path)

    def failed(*_: object) -> None:
        raise RuntimeError("the mapping failed")

    monkeypatch.setattr(report, "decode", failed)
    with pytest.raises(RuntimeError, match="mapping failed"):
        main([str(directory)])
    assert (directory / FITS).is_file()
    assert (directory / CELLS).is_file()
    assert not (directory / DECODING).exists()

    monkeypatch.undo()
    fits = (directory / FITS).read_text(encoding="utf-8")
    main([str(directory), "--decode-only"])

    assert (directory / FITS).read_text(encoding="utf-8") == fits
    assert (directory / DECODING).is_file()
