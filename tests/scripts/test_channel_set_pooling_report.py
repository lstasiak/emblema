"""What the poolings over a stay's channels hold, and what the probe's head reads off them.

The recorded directory is the pooling report's: two channels, the timed one unmeasured on every
fifth stay, and a pretrained encoder whose per-channel state carries the timed channel's level.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.channel_pooling_report import (
    CHANNEL,
    WINDOW,
    ChannelDecoded,
    RefusalError,
    decode,
    reductions_of,
)
from scripts.channel_set_pooling_report import (
    DECODING,
    REDUCTION,
    REPORT,
    SET_EXTREMES,
    SET_MEAN,
    STATISTICS,
    main,
    pooled_of,
    pooled_over_channels,
)
from scripts.channel_set_pooling_report import decode as decode_pooled
from scripts.stay_summary_report import (
    FITS,
    UNTRAINED_PREFIX,
    Fit,
    Stored,
    read_rows,
    write_rows,
)
from tests.scripts.test_channel_pooling_report import (
    BLOCK,
    COMPONENTS,
    LAYERS,
    widened,
)
from tests.scripts.test_stay_summary_report import (
    FRESH,
    INFORMED,
    TIMED,
    stored_directory,
)

pytestmark = pytest.mark.ml


@pytest.fixture
def recorded(tmp_path: Path) -> Path:
    return widened(tmp_path / "recorded")


def test_a_stay_pools_only_the_channels_it_measured() -> None:
    per_channel = np.array(
        [
            [[1.0, -2.0], [0.0, 0.0], [3.0, 3.0]],
            [[3.0, 4.0], [5.0, -1.0], [0.0, 0.0]],
            [[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    measured = [
        np.array([True, False, True]),
        np.array([True, True, False]),
        np.array([True, True, False]),
    ]

    pooled = pooled_over_channels(per_channel, measured)

    np.testing.assert_allclose(pooled["mean"], [[2.0, 2 / 3], [3.0, 0.0], [3.0, 3.0]])
    np.testing.assert_allclose(pooled["maximum"], [[3.0, 4.0], [5.0, 1.0], [3.0, 3.0]])
    np.testing.assert_allclose(pooled["minimum"], [[1.0, -2.0], [1.0, -1.0], [3.0, 3.0]])


def test_no_order_of_the_channels_changes_a_pooling() -> None:
    generator = np.random.default_rng(5)
    per_channel = generator.normal(size=(4, 30, 3)).astype(np.float32)
    measured = [generator.random(30) < 0.7 for _ in range(4)]
    measured[0][:] = True
    for index, on in enumerate(measured):
        per_channel[index, ~on] = 0.0
    shuffled = [2, 0, 3, 1]

    pooled = pooled_over_channels(per_channel, measured)
    reordered = pooled_over_channels(per_channel[shuffled], [measured[i] for i in shuffled])

    for name in STATISTICS:
        np.testing.assert_allclose(reordered[name], pooled[name], rtol=1e-6)


def test_a_state_on_a_stay_that_never_measured_its_channel_is_refused() -> None:
    per_channel = np.ones((2, 3, 2), dtype=np.float32)
    measured = [np.array([True, True, True]), np.array([True, False, True])]

    with pytest.raises(RefusalError, match="channel 1 holds a state on a stay"):
        pooled_over_channels(per_channel, measured)


def test_a_stay_that_measured_no_channel_is_refused() -> None:
    per_channel = np.zeros((2, 3, 2), dtype=np.float32)
    per_channel[:, :2] = 1.0
    measured = [np.array([True, True, False]), np.array([True, True, False])]

    with pytest.raises(RefusalError, match="1 stays measured no channel"):
        pooled_over_channels(per_channel, measured)


def test_the_channels_measured_must_be_the_channels_stored() -> None:
    with pytest.raises(RefusalError, match="1 channels measured against 2 stored"):
        pooled_over_channels(np.zeros((2, 3, 2), dtype=np.float32), [np.ones(3, dtype=bool)])


def test_a_channels_level_survives_the_mean_over_channels_that_the_window_dilutes(
    recorded: Path,
) -> None:
    stored = Stored.read(recorded)
    pooled = pooled_of(stored, recorded, COMPONENTS)

    decoded = decode_pooled(stored, pooled.states)
    earlier = decode(stored, recorded, reductions_of(stored, recorded, COMPONENTS), COMPONENTS)

    def r2(rows: list[ChannelDecoded], digest: str, pooling: str) -> float:
        (row,) = [
            row
            for row in rows
            if (row.digest, row.pooling, row.block, row.channel_id, row.statistic)
            == (digest, pooling, "all", TIMED, "mean")
        ]
        return row.r2

    assert pooled.states[SET_MEAN][INFORMED].shape[1] == BLOCK * LAYERS
    assert pooled.states[SET_EXTREMES][INFORMED].shape[1] == len(STATISTICS) * COMPONENTS
    assert pooled.kept[INFORMED].shape == (len(STATISTICS),)
    assert r2(decoded, INFORMED, SET_MEAN) > 0.8
    assert r2(decoded, INFORMED, SET_EXTREMES) > 0.5
    assert r2(earlier, INFORMED, WINDOW) < 0.2
    assert r2(decoded, FRESH[0], SET_MEAN) < 0.2


def test_every_pooling_is_read_and_the_report_renders_again_from_its_files(
    recorded: Path, tmp_path: Path
) -> None:
    stored = Stored.read(recorded)
    reference = tmp_path / "channel_decoding.csv"
    earlier = decode(stored, recorded, reductions_of(stored, recorded, COMPONENTS), COMPONENTS)
    write_rows(reference, ChannelDecoded.COLUMNS, (row.record() for row in earlier))
    out = tmp_path / "out"
    arguments = [
        str(recorded),
        "--out",
        str(out),
        "--components",
        "1",
        "--reference-decoding",
        str(reference),
    ]

    main(arguments)

    fits = [Fit.parse(row) for row in read_rows(out / FITS)]
    widths = {fit.candidate: fit.columns for fit in fits}
    assert widths == {
        SET_MEAN: BLOCK * LAYERS,
        UNTRAINED_PREFIX + SET_MEAN: BLOCK * LAYERS,
        SET_EXTREMES: len(STATISTICS),
        UNTRAINED_PREFIX + SET_EXTREMES: len(STATISTICS),
    }
    assert len(read_rows(out / REDUCTION)) == len(STATISTICS) * len({*FRESH, INFORMED})
    decoded = [ChannelDecoded.parse(row) for row in read_rows(out / DECODING)]
    assert {row.pooling for row in decoded} == {SET_MEAN, SET_EXTREMES}
    first = (out / REPORT).read_text(encoding="utf-8")
    for pooling in (WINDOW, CHANNEL, SET_MEAN, SET_EXTREMES):
        assert f"{pooling}, pretrained" in first

    main([*arguments, "--report-only"])

    assert (out / REPORT).read_text(encoding="utf-8") == first


def test_a_directory_without_per_channel_states_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "plain"
    directory.mkdir()
    stored_directory(directory)

    with pytest.raises(SystemExit, match="no recorded per-channel states"):
        main([str(directory), "--out", str(tmp_path / "out")])
