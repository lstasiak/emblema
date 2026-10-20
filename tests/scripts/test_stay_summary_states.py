"""The storing step rebuilds what the probe campaigns' cells read, and refuses anything else.

Campaigns are designed from the repository's own file of the stage reader's probes, under the
settings every campaign of the repository was defined under; the encoders are the small test
ones; nothing reaches a registry, a store or an accelerator.
"""

from dataclasses import replace
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emblema.entrypoints.workers.known_arms import KnownArms  # noqa: E402
from emblema.entrypoints.workers.known_baselines import KnownBaselines  # noqa: E402
from emblema.evaluation.adapters.campaigns.campaign_file import CampaignFile  # noqa: E402
from emblema.evaluation.adapters.candidates.routed_candidate_catalogue import (  # noqa: E402
    RoutedCandidateCatalogue,
)
from emblema.evaluation.adapters.features.per_channel_features import (  # noqa: E402
    PerChannelFeatures,
)
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone  # noqa: E402
from emblema.evaluation.application.use_cases.draw_run_labels import (  # noqa: E402
    DrawRunLabelsCommand,
)
from emblema.evaluation.contracts.identifiers import CampaignId, TaskId  # noqa: E402
from emblema.evaluation.domain.campaign.campaign_design import CampaignDesign  # noqa: E402
from emblema.evaluation.domain.campaign.evaluation_campaign import (  # noqa: E402
    EvaluationCampaign,
)
from emblema.evaluation.domain.labels.label_budget import LabelBudget  # noqa: E402
from emblema.evaluation.domain.labels.label_sample import LabelSample  # noqa: E402
from emblema.evaluation.domain.labels.run_labels import RunLabels  # noqa: E402
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure  # noqa: E402
from emblema.evaluation.domain.task.inner_holdout import InnerHoldout  # noqa: E402
from emblema.evaluation.domain.transfer.adaptation_schedule import (  # noqa: E402
    AdaptationSchedule,
)
from emblema.evaluation.domain.transfer.transfer_mode import TransferMode  # noqa: E402
from emblema.shared.adapters.tensors.token_tensors import TokenTensors  # noqa: E402
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
from emblema.shared.kernel.tokens import Token, TokenWindow  # noqa: E402
from scripts.stay_summary_report import (  # noqa: E402
    CONSTANT,
    LEARNT,
    PRETRAINED,
    RECORDED,
    SCORED,
    TRANSPLANTED,
    UNTRAINED,
    StoredDraw,
)
from scripts.stay_summary_states import (  # noqa: E402
    RefusalError,
    alike,
    arms_of,
    candidate_of,
    channels_of,
    constant_values,
    digest_of,
    drawn,
    embedded,
    embedded_windows,
    observed_channels,
    plan_of,
    stays_of,
    summaries_of,
    transplanted_values,
)
from tests.entrypoints.cli.campaign.test_every_campaign_file import DECLARED  # noqa: E402
from tests.evaluation.support import (  # noqa: E402
    OPENED_AT,
    OUTCOME,
    OUTCOMES,
    TASK,
    WEIGHTS,
    labelled,
    plan,
    task,
    units,
)
from tests.support.backbones import SmallBackbones  # noqa: E402
from tests.support.published import CHANNELS, manifest_of  # noqa: E402
from tests.support.token_tensors import VOCABULARY_SIZE  # noqa: E402

pytestmark = pytest.mark.ml

CAMPAIGNS = Path(__file__).resolve().parents[2] / "campaigns"
PROBES = CAMPAIGNS / "reader-probes-physionet2012.toml"


def designed(
    path: Path = PROBES,
    *,
    number: int = 1,
    task_id: TaskId = TASK,
    weights: ArtifactRef = WEIGHTS,
    schedule: AdaptationSchedule | None = None,
) -> EvaluationCampaign:
    """The campaign a file declares, its candidates described as the defining process does."""
    declared = CampaignFile.load(path)
    catalogue = RoutedCandidateCatalogue(
        {
            **dict.fromkeys(
                KnownArms.refs(),
                KnownArms.catalogue(
                    weights,
                    DECLARED.lora(),
                    DECLARED.schedule() if schedule is None else schedule,
                    DECLARED.probe(),
                ),
            ),
            **dict.fromkeys(
                KnownBaselines.refs(),
                KnownBaselines.catalogue(DECLARED.boosting(), DECLARED.convolutions()),
            ),
        }
    )
    return EvaluationCampaign.designed(
        campaign_id=CampaignId(UUID(int=number)),
        task=task_id,
        purpose=declared.purpose,
        tier=declared.tier,
        design=CampaignDesign(
            candidates=tuple(catalogue.describe(ref) for ref in declared.competing()),
            control=declared.control(),
            endpoint=declared.endpoint(),
            budgets=declared.label_budgets(),
            endpoint_budget=declared.endpoint_budget(),
            seeds=declared.budgets.seeds,
            rules=declared.comparison_rules(),
            bootstrap=declared.paired_bootstrap(),
            inner_holdout=declared.inner_holdout(),
            measure=ErrorMeasure.RMSE if declared.measure is None else declared.measure,
        ),
        opened_at=OPENED_AT,
    )


def test_the_campaigns_of_the_probes_share_a_task_a_division_and_seeds() -> None:
    other = ArtifactRef(key="durable/other", checksum=WEIGHTS.checksum)

    shared = alike(
        [("mix1", designed()), ("mix2", designed(number=2, weights=other))], ["200", "500"]
    )

    assert shared == (TASK, InnerHoldout(one_in=2, division_seed=101), tuple(range(1, 11)), "M")


@pytest.mark.parametrize("name", ["Mix1", "mix+2", "summaries", "", "-mix"])
def test_a_name_that_cannot_label_a_reading_is_refused(name: str) -> None:
    with pytest.raises(RefusalError, match="cannot label"):
        alike([(name, designed())], ["200"])


def test_two_campaigns_under_one_name_are_refused() -> None:
    with pytest.raises(RefusalError, match="share a name"):
        alike([("mix1", designed()), ("mix1", designed(number=2))], ["200"])


@pytest.mark.parametrize(
    ("path", "budgets", "reason"),
    [
        ("selection-classical-physionet2012.toml", ["200"], "afresh"),
        ("classical-half-physionet2012.toml", ["200"], "holds no"),
        ("reader-probes-physionet2012.toml", ["50"], "ran no cell at 50"),
    ],
)
def test_a_campaign_whose_cells_are_not_the_probes_on_a_fixed_half_is_refused(
    path: str, budgets: list[str], reason: str
) -> None:
    with pytest.raises(RefusalError, match=reason):
        alike([("mix1", designed(CAMPAIGNS / path))], budgets)


def test_campaigns_over_different_tasks_are_refused() -> None:
    elsewhere = designed(number=2, task_id=TaskId(UUID(int=99)))

    with pytest.raises(RefusalError, match="differ"):
        alike([("mix1", designed()), ("mix2", elsewhere)], ["200"])


def test_both_probes_are_built_as_the_campaign_recorded_them() -> None:
    probes = arms_of("mix1", designed(), DECLARED)

    assert probes.weights == WEIGHTS
    assert probes.arms[PRETRAINED].backbone == WEIGHTS
    assert probes.arms[UNTRAINED].backbone is None
    assert probes.arms[PRETRAINED].mode is TransferMode.FROZEN_RIDGE


def test_settings_that_would_build_another_probe_are_refused() -> None:
    recorded = designed(schedule=replace(DECLARED.schedule(), batch_size=32))

    with pytest.raises(RefusalError, match="mix1"):
        arms_of("mix1", recorded, DECLARED)


def _labels(scored: tuple[str, ...], seed: int) -> RunLabels:
    pool = [labelled(f"a{index}", index, 1.0, float(index % 2)) for index in range(6)]
    defined = task(
        tuning=units(*(f"a{i}" for i in range(6)), *scored), labels=OUTCOME, strata=OUTCOMES
    )
    return RunLabels(
        task=defined,
        sample=LabelSample(task=TASK, windows=tuple(pool[:4]), budget=LabelBudget.of(4), seed=seed),
        scored=tuple(labelled(unit, 100 + i, 1.0, 0.0) for i, unit in enumerate(scored)),
    )


def test_every_cell_is_drawn_and_answers_the_one_held_out_half() -> None:
    asked: list[DrawRunLabelsCommand] = []

    def draw(command: DrawRunLabelsCommand) -> RunLabels:
        asked.append(command)
        return _labels(("s1", "s2"), command.seed)

    holdout = InnerHoldout(one_in=2, division_seed=101)
    draws, scored = drawn(draw, TASK, holdout, ["4"], [1, 2])

    assert [command.seed for command in asked] == [1, 2]
    assert all(command.holdout == holdout for command in asked)
    assert len(draws) == 8
    assert [str(each.window.unit) for each in scored] == ["s1", "s2"]


def test_cells_that_answer_different_stays_are_refused() -> None:
    def draw(command: DrawRunLabelsCommand) -> RunLabels:
        return _labels(("s1",) if command.seed == 1 else ("s2",), command.seed)

    with pytest.raises(RefusalError, match="other stays"):
        drawn(draw, TASK, InnerHoldout(one_in=2, division_seed=101), ["4"], [1, 2])


def test_no_cell_asked_for_is_refused() -> None:
    with pytest.raises(RefusalError, match="at least one"):
        drawn(lambda command: _labels(("s1",), 1), TASK, InnerHoldout(one_in=2), [], [1])


def test_the_learnt_half_comes_first_and_each_half_in_the_order_of_its_stays() -> None:
    learnt = [labelled("b", 2, 1.0, 1.0), labelled("a", 1, 1.0, 0.0)]
    scored = [labelled("d", 4, 1.0, 0.0), labelled("c", 3, 1.0, 1.0)]

    stays = stays_of(learnt, scored, [StoredDraw(budget="1", seed=1, unit="a", position=1)])

    assert [(stay.row, stay.half, stay.unit) for stay in stays] == [
        (0, LEARNT, "a"),
        (1, LEARNT, "b"),
        (2, SCORED, "c"),
        (3, SCORED, "d"),
    ]


def test_halves_that_share_a_stay_or_a_draw_off_the_learnt_half_are_refused() -> None:
    learnt = [labelled("a", 1, 1.0, 0.0)]

    with pytest.raises(RefusalError, match="share"):
        stays_of(learnt, [labelled("a", 1, 1.0, 0.0)], [])
    with pytest.raises(RefusalError, match="not on the learnt half"):
        stays_of(learnt, [], [StoredDraw(budget="1", seed=1, unit="z", position=9)])


def _window(seed: int) -> TokenWindow:
    generator = np.random.default_rng(seed)
    count = int(generator.integers(3, 9))
    times = np.sort(generator.uniform(size=count))
    return TokenWindow.of(
        (
            Token(channel_id=3, value=float(seed), time=0.0, gap=0.0, timeless=True),
            *(
                Token(
                    channel_id=int(generator.integers(1, 3)),
                    value=float(generator.normal()),
                    time=float(time),
                    gap=0.0,
                )
                for time in times
            ),
        )
    )


def test_a_pretrained_probe_reads_one_encoder_under_every_seed_and_the_untrained_one_another() -> (
    None
):
    probes = arms_of("mix1", designed(), DECLARED)
    backbones = SmallBackbones()

    def digest(arm: str, seed: int) -> str:
        built = candidate_of(
            plan_of(probes.arms[arm], seed), backbones, vocabulary_size=VOCABULARY_SIZE
        )
        return digest_of(built)

    assert digest(PRETRAINED, 1) == digest(PRETRAINED, 7)
    assert digest(UNTRAINED, 1) == digest(UNTRAINED, 1)
    assert digest(UNTRAINED, 1) != digest(UNTRAINED, 2)
    assert digest(UNTRAINED, 1) != digest(PRETRAINED, 1)


def test_a_plan_that_trains_its_encoder_is_refused() -> None:
    with pytest.raises(RefusalError, match="computed once"):
        candidate_of(plan(TransferMode.FULL_FINE_TUNING), SmallBackbones(), vocabulary_size=64)


def test_the_stored_states_are_the_candidates_embedding_in_any_batch() -> None:
    probes = arms_of("mix1", designed(), DECLARED)
    built = candidate_of(
        plan_of(probes.arms[PRETRAINED], 3), SmallBackbones(), vocabulary_size=VOCABULARY_SIZE
    )
    windows = [_window(seed) for seed in range(7)]

    in_threes = embedded(built, windows, batch_size=3, device="cpu")
    whole = embedded(built, windows, batch_size=len(windows), device="cpu")

    assert in_threes.dtype == np.float32
    assert in_threes.shape == (7, SmallBackbones().width * SmallBackbones().layers)
    np.testing.assert_allclose(in_threes, whole, atol=1e-5)


def test_the_summaries_are_the_classical_models_columns_of_the_observed_channels() -> None:
    windows = [_window(seed) for seed in range(5)]
    observed = observed_channels(windows)

    summaries, channels, statistics = summaries_of(windows, observed, len(CHANNELS))

    every = PerChannelFeatures(len(CHANNELS))
    names = every.names()
    expected = [
        names.index(f"channel_{channel}_{statistic}")
        for channel, statistic in zip(channels.tolist(), statistics.tolist(), strict=True)
    ]
    assert observed == [1, 2, 3]
    np.testing.assert_array_equal(summaries, every.of(windows)[:, expected])


def test_the_channels_are_named_and_marked_static_as_the_manifest_says() -> None:
    manifest = manifest_of(WEIGHTS)

    named = channels_of(manifest, [1, 3])

    assert [(c.channel_id, c.name, c.timeless) for c in named] == [
        (1, "test-corpus/temperature", False),
        (3, "test-corpus/age", True),
    ]


def _built(seed: int = 3) -> AdaptedBackbone:
    probes = arms_of("mix1", designed(), DECLARED)
    return candidate_of(
        plan_of(probes.arms[PRETRAINED], seed), SmallBackbones(), vocabulary_size=VOCABULARY_SIZE
    )


def test_each_channels_mean_state_is_stored_from_the_pass_the_probe_reads(tmp_path: Path) -> None:
    built = _built()
    windows = [_window(seed) for seed in range(7)]
    channels = [1, 2, 3]

    in_threes = embedded(
        built, windows, batch_size=3, device="cpu", per_channel=(channels, tmp_path / "a.npy")
    )
    whole = embedded(
        built,
        windows,
        batch_size=len(windows),
        device="cpu",
        per_channel=(channels, tmp_path / "b.npy"),
    )

    means = np.load(tmp_path / "a.npy")
    np.testing.assert_allclose(means, np.load(tmp_path / "b.npy"), atol=1e-5)
    np.testing.assert_allclose(in_threes, whole, atol=1e-5)
    batch = TokenTensors.from_windows(windows)
    with torch.no_grad():
        states = built.encoder(*batch.args).numpy()
        probe = built.embed(batch).numpy()
    np.testing.assert_allclose(in_threes, probe, atol=1e-5)
    assert means.shape == (len(channels), len(windows), states.shape[-1])
    for row, window in enumerate(windows):
        ids = np.array(window.channel_ids)
        for index, channel in enumerate(channels):
            held = ids == channel
            expected = states[row, : len(ids)][held].mean(axis=0) if held.any() else 0.0
            np.testing.assert_allclose(means[index, row], expected, atol=1e-5)
        # Every token belongs to one of the channels, so the means weighted by their counts are
        # the probe's mean over the window.
        counts = np.array([(ids == channel).sum() for channel in channels], dtype=np.float64)
        weighted = (counts[:, None] * means[:, row]).sum(axis=0) / counts.sum()
        np.testing.assert_allclose(weighted, probe[row], atol=1e-5)


def _schedule(window: TokenWindow) -> list[tuple[int, float, float, bool]]:
    return sorted((token.channel_id, token.time, token.gap, token.timeless) for token in window)


def test_a_constant_holds_each_channels_mean_over_the_learnt_half_and_the_schedule() -> None:
    windows = [_window(seed) for seed in range(6)]
    learnt = 4
    means = {
        channel: np.mean(
            [
                token.value
                for window in windows[:learnt]
                for token in window
                if token.channel_id == channel and not token.timeless
            ]
        )
        for channel in (1, 2)
    }

    replaced = constant_values(windows, learnt)

    for before, after in zip(windows, replaced, strict=True):
        assert _schedule(after) == _schedule(before)
        for token in after:
            expected = before.values[0] if token.timeless else means[token.channel_id]
            assert token.value == pytest.approx(expected)


def test_a_channel_the_learnt_half_never_measures_has_no_constant() -> None:
    unseen = TokenWindow.of([Token(channel_id=2, value=1.0, time=0.5, gap=0.0)])
    seen = TokenWindow.of([Token(channel_id=1, value=1.0, time=0.5, gap=0.0)])

    with pytest.raises(RefusalError, match="never measures channel 2"):
        constant_values([seen, unseen], 1)


def test_transplanted_values_come_from_another_learnt_stay_of_the_same_channel() -> None:
    windows = [_window(seed) for seed in range(10)]
    units = [f"set-a/{row:04d}" for row in range(10)]
    learnt = 6

    replaced = transplanted_values(windows, units, learnt, seed=1)

    assert transplanted_values(windows, units, learnt, seed=1) == replaced
    assert transplanted_values(windows, units, learnt, seed=2) != replaced
    for row, (before, after) in enumerate(zip(windows, replaced, strict=True)):
        assert _schedule(after) == _schedule(before)
        for channel in (1, 2):
            own = {t.value for t in before if t.channel_id == channel and not t.timeless}
            drawn = {t.value for t in after if t.channel_id == channel and not t.timeless}
            if not own:
                continue
            donors = [
                other
                for other in range(learnt)
                if other != row
                and drawn <= {t.value for t in windows[other] if t.channel_id == channel}
            ]
            assert donors, f"stay {row}'s channel {channel} holds values of no other learnt stay"
        assert [t.value for t in after if t.timeless] == [t.value for t in before if t.timeless]


def test_a_stay_whose_channel_no_other_learnt_stay_measures_has_no_donor() -> None:
    alone = TokenWindow.of([Token(channel_id=2, value=1.0, time=0.5, gap=0.0)])
    other = TokenWindow.of([Token(channel_id=1, value=1.0, time=0.5, gap=0.0)])

    with pytest.raises(RefusalError, match="no other learnt stay measures channel 2"):
        transplanted_values([alone, other], ["a", "b"], 2, seed=1)


@pytest.mark.parametrize("values", [RECORDED, CONSTANT, TRANSPLANTED])
def test_the_windows_are_read_as_the_values_option_says(values: str) -> None:
    windows = [_window(seed) for seed in range(6)]
    units = [f"u{row}" for row in range(6)]

    read = embedded_windows(windows, units, 4, values, 1)

    expected = {
        RECORDED: windows,
        CONSTANT: constant_values(windows, 4),
        TRANSPLANTED: transplanted_values(windows, units, 4, 1),
    }[values]
    assert read == expected


def test_no_other_way_to_read_the_values_is_taken() -> None:
    with pytest.raises(RefusalError, match="no way"):
        embedded_windows([_window(1)], ["u"], 1, "shuffled", 1)
