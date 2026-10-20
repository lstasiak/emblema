"""What each pooling of the states recovers and what each head reads, off directories as stored.

The recorded directory is the one the summary report's tests write, widened as the storing step
widens it: two blocks, each channel's mean state, and a timed channel some stays never measure.
The pretrained encoder's per-channel state carries the timed channel's level in its second block
alone; its mean over the window carries it diluted; the encoders at initialisation carry nothing.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emblema.evaluation.adapters.features.window_statistics import (  # noqa: E402
    WindowStatistics,
)
from scripts.channel_pooling_report import (  # noqa: E402
    CHANNEL,
    CHANNEL_REDUCED,
    CHANNEL_STATES_READING,
    DECODING,
    EVERY_BLOCK,
    REDUCED,
    REPORT,
    SCHEDULE_READING,
    VALUES_ALONE_READING,
    VALUES_READING,
    WINDOW,
    ChannelDecoded,
    RefusalError,
    block_slices,
    blocks_of,
    channel_targets,
    counts_of,
    decode,
    main,
    matching,
    reduced,
    reductions_of,
    schedule_alone,
    summaries_where,
    values_alone,
)
from scripts.stay_summary_report import (  # noqa: E402
    CHANNEL_STATES,
    CONDITIONS,
    CONSTANT,
    FITS,
    LEARNT,
    RECORDED,
    SCORED,
    STATES,
    STATES_ALONE,
    SUMMARIES,
    SUMMARIES_SOURCE,
    TRANSPLANTED,
    UNTRAINED_PREFIX,
    Fit,
    Stored,
    read_rows,
    save_arrays,
)
from tests.scripts.test_stay_summary_report import (  # noqa: E402
    FRESH,
    INFORMED,
    STATIC,
    TIMED,
    stored_directory,
)

pytestmark = pytest.mark.ml

BLOCK, LAYERS = 3, 2
COMPONENTS = 2
NAMES = WindowStatistics.NAMES
# Every fifth stay never measures the timed channel.
UNMEASURED = slice(None, None, 5)


def _widened(directory: Path, values: str = RECORDED) -> Path:
    """The summary report's directory, as the storing step stores it with per-channel states."""
    directory.mkdir()
    stored_directory(directory)
    with np.load(directory / STATES) as archive:
        arrays = {name: archive[name] for name in archive.files}
    summaries = arrays[SUMMARIES]
    summaries[UNMEASURED, : len(NAMES)] = np.nan
    summaries[UNMEASURED, NAMES.index("count")] = 0.0
    level = summaries[:, NAMES.index("mean")]
    generator = np.random.default_rng(11)
    stays = summaries.shape[0]
    (directory / CHANNEL_STATES).mkdir()
    for digest in (INFORMED, *FRESH):
        per_channel = generator.normal(size=(2, stays, BLOCK * LAYERS)).astype(np.float32)
        if digest == INFORMED:
            # The level is the widest direction of the channel's states, as a reduction keeps it.
            per_channel[0, :, BLOCK] = 3.0 * np.nan_to_num(level) + 0.1 * generator.normal(
                size=stays
            )
            # The window's mean holds the level at a tenth, under noise as wide as the level.
            arrays[digest] = generator.normal(size=(stays, BLOCK * LAYERS)).astype(np.float32)
            arrays[digest][:, BLOCK] += 0.1 * np.nan_to_num(level)
        per_channel[0, UNMEASURED] = 0.0
        np.save(directory / CHANNEL_STATES / f"{digest}.npy", per_channel)
    save_arrays(directory / STATES, arrays)
    conditions = json.loads((directory / CONDITIONS).read_text(encoding="utf-8"))
    conditions |= {
        "values": values,
        "per_channel": values == RECORDED,
        "shapes": {
            source: {"width": BLOCK, "layers": LAYERS} for source in conditions["campaigns"]
        },
    }
    (directory / CONDITIONS).write_text(json.dumps(conditions), encoding="utf-8")
    return directory


@pytest.fixture
def recorded(tmp_path: Path) -> Path:
    return _widened(tmp_path / "recorded")


def test_the_blocks_are_the_concatenated_states_side_by_side(recorded: Path) -> None:
    assert blocks_of(Stored.read(recorded)) == (BLOCK, LAYERS)
    assert block_slices(BLOCK, LAYERS) == {
        EVERY_BLOCK: slice(0, 6),
        "1": slice(0, 3),
        "2": slice(3, 6),
    }


def test_states_not_as_wide_as_every_block_are_refused(recorded: Path) -> None:
    stored = Stored.read(recorded)
    conditions = {**stored.conditions, "shapes": {"mix1": {"width": 4, "layers": 2}}}

    with pytest.raises(RefusalError, match="not 2 blocks of 4"):
        blocks_of(replace(stored, conditions=conditions))


def test_a_reduction_keeps_the_leading_components_of_the_measured_learnt_stays() -> None:
    generator = np.random.default_rng(3)
    basis = generator.normal(size=(2, 8))
    states = (generator.normal(size=(1, 40, 2)) @ basis).astype(np.float64) + 5.0
    measured = np.ones(40, dtype=bool)
    measured[::4] = False
    states[0, ~measured] = 0.0
    learnt = np.arange(30)

    found = reduced(states, [measured], learnt, 3)

    assert found.kept[0] == pytest.approx(1.0)
    assert found.states.shape == (40, 3)
    rows = learnt[measured[learnt]]
    centre = states[0, rows].mean(axis=0)
    # Two components hold every measured stay; the third has nothing left to hold.
    np.testing.assert_allclose(found.states[measured, 2], 0.0, atol=1e-6)
    lengths = np.linalg.norm(found.states[measured], axis=1)
    np.testing.assert_allclose(lengths, np.linalg.norm(states[0, measured] - centre, axis=1))
    # A stay without the channel lands where the zero state maps: the centre's image, negated.
    _, _, right_t = np.linalg.svd(states[0, rows] - centre, full_matrices=False)
    image = -(centre @ right_t[:2].T)
    np.testing.assert_allclose(
        np.abs(found.states[~measured, :2]), np.abs(np.tile(image, (10, 1))), rtol=1e-6
    )


def test_a_channel_on_fewer_stays_than_components_keeps_what_it_has() -> None:
    states = np.random.default_rng(4).normal(size=(1, 6, 5))

    found = reduced(states, [np.ones(6, dtype=bool)], np.arange(2), 4)

    assert found.kept[0] == pytest.approx(1.0)
    np.testing.assert_array_equal(found.states[:, 2:], 0.0)


def test_a_channels_summaries_are_read_on_the_stays_that_measure_it(recorded: Path) -> None:
    stored = Stored.read(recorded)
    measured = counts_of(stored)

    timed = channel_targets(stored, TIMED, measured[TIMED])
    static = channel_targets(stored, STATIC, measured[STATIC])

    statistics = [stored.summary_columns()[column][1] for column in timed]
    assert {"count", "mean"} <= set(statistics)
    # A static is measured by every stay, so its presence does not vary and is not read.
    assert [stored.summary_columns()[column][1] for column in static] == ["mean"]
    assert not measured[TIMED][UNMEASURED].any()


def test_the_schedule_and_the_values_divide_a_timed_channels_summaries(recorded: Path) -> None:
    stored = Stored.read(recorded)

    schedule = summaries_where(stored, schedule_alone).summary_columns()
    values = summaries_where(stored, values_alone).summary_columns()

    timed_schedule = {s for channel, s in schedule if channel.channel_id == TIMED}
    timed_values = {s for channel, s in values if channel.channel_id == TIMED}
    assert timed_schedule == {"count", "gap"}
    assert timed_values == set(NAMES) - {"count", "gap"}
    assert {s for c, s in schedule if c.channel_id == STATIC} == set(NAMES)
    assert {s for c, s in values if c.channel_id == STATIC} == set(NAMES)
    assert summaries_where(stored, schedule_alone).summaries.shape[1] == len(schedule)


def test_the_level_carried_by_one_channels_tokens_is_recovered_where_it_lies(
    recorded: Path,
) -> None:
    stored = Stored.read(recorded)
    reductions = reductions_of(stored, recorded, COMPONENTS)

    decoded = decode(stored, recorded, reductions, COMPONENTS)

    def r2(digest: str, pooling: str, block: str) -> float:
        (row,) = [
            row
            for row in decoded
            if (row.digest, row.pooling, row.block, row.channel_id, row.statistic)
            == (digest, pooling, block, TIMED, "mean")
        ]
        return row.r2

    assert r2(INFORMED, CHANNEL, "2") > 0.9
    assert r2(INFORMED, CHANNEL, EVERY_BLOCK) > 0.9
    assert r2(INFORMED, CHANNEL, "1") < 0.2
    assert r2(INFORMED, CHANNEL_REDUCED, EVERY_BLOCK) > 0.5
    assert r2(INFORMED, WINDOW, EVERY_BLOCK) < 0.2
    assert r2(FRESH[0], CHANNEL, EVERY_BLOCK) < 0.2
    measured_learnt = sum(1 for stay in stored.half(LEARNT) if counts_of(stored)[TIMED][stay.row])
    measured_scored = sum(1 for stay in stored.half(SCORED) if counts_of(stored)[TIMED][stay.row])
    (count,) = [
        row
        for row in decoded
        if (row.digest, row.pooling, row.block, row.statistic) == (INFORMED, WINDOW, "1", "count")
        and row.channel_id == TIMED
    ]
    assert (count.learnt, count.scored) == (measured_learnt, measured_scored)


def test_a_directory_of_replaced_values_must_be_of_the_same_stays_and_commit(
    recorded: Path, tmp_path: Path
) -> None:
    stored = Stored.read(recorded)
    constant = Stored.read(_widened(tmp_path / "constant", CONSTANT))

    matching(stored, constant, CONSTANT)
    with pytest.raises(RefusalError, match="holds other values"):
        matching(stored, constant, TRANSPLANTED)
    with pytest.raises(RefusalError, match="other stays"):
        matching(stored, replace(constant, stays=constant.stays[1:]), CONSTANT)
    elsewhere = {**constant.conditions, "commit": "def5678"}
    with pytest.raises(RefusalError, match="another commit"):
        matching(stored, replace(constant, conditions=elsewhere), CONSTANT)


def test_every_head_is_fitted_and_the_report_renders_again_from_its_files(
    recorded: Path, tmp_path: Path
) -> None:
    constant = _widened(tmp_path / "constant", CONSTANT)
    transplanted = _widened(tmp_path / "transplanted", TRANSPLANTED)
    out = tmp_path / "out"
    arguments = [
        str(recorded),
        "--constant",
        str(constant),
        "--transplanted",
        str(transplanted),
        "--out",
        str(out),
        "--components",
        str(COMPONENTS),
    ]

    main(arguments)

    fits = [Fit.parse(row) for row in read_rows(out / FITS)]
    candidates = {fit.candidate for fit in fits}
    for reading in (
        STATES_ALONE,
        CHANNEL_STATES_READING,
        VALUES_READING[CONSTANT],
        VALUES_READING[TRANSPLANTED],
    ):
        assert {reading, UNTRAINED_PREFIX + reading} <= candidates
    assert {SUMMARIES_SOURCE, SCHEDULE_READING, VALUES_ALONE_READING} <= candidates
    widths = {fit.candidate: fit.columns for fit in fits}
    assert widths[CHANNEL_STATES_READING] == 2 * COMPONENTS
    assert (out / REDUCED).is_file()
    decoded = [ChannelDecoded.parse(row) for row in read_rows(out / DECODING)]
    assert {row.pooling for row in decoded} == {WINDOW, CHANNEL, CHANNEL_REDUCED}
    first = (out / REPORT).read_text(encoding="utf-8")

    main([*arguments, "--report-only"])

    assert (out / REPORT).read_text(encoding="utf-8") == first


def test_a_recorded_directory_without_per_channel_states_is_refused(tmp_path: Path) -> None:
    constant = _widened(tmp_path / "constant", CONSTANT)
    transplanted = _widened(tmp_path / "transplanted", TRANSPLANTED)

    with pytest.raises(SystemExit, match="no recorded per-channel states"):
        main(
            [
                str(constant),
                "--constant",
                str(constant),
                "--transplanted",
                str(transplanted),
                "--out",
                str(tmp_path / "out"),
            ]
        )
