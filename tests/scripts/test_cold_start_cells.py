"""A satellite read as a new system: its cells, the answers given then, the library's series."""

import numpy as np
import pytest

from scripts.cold_start_cells import (
    CONTEXT,
    LEAST,
    Library,
    Mission,
    Ridge,
    block_ends,
    cells_of,
    drawn_hours,
    filled,
    fitted_ridges,
    nearest,
    random_segment,
    ridge_features,
    selections_of,
    zscored,
)


def mission(values: np.ndarray, kinds: tuple[str, ...], name: str = "new") -> Mission:
    channels = tuple(f"{name}/channel_{c}" for c in range(values.shape[1]))
    return Mission.dense(name, channels, kinds, np.arange(len(values)), values)


def test_a_dense_mission_leaves_hours_not_loaded_missing() -> None:
    dense = Mission.dense("m", ("m/channel_1",), ("a",), np.array([0, 3]), np.array([[1.0], [2.0]]))

    assert dense.values[:, 0].tolist() == pytest.approx([1.0, np.nan, np.nan, 2.0], nan_ok=True)
    with pytest.raises(ValueError, match="do not agree"):
        Mission.dense("m", ("m/channel_1",), ("a", "b"), np.array([0]), np.array([[1.0]]))


def test_a_channel_is_scored_where_it_read_and_half_the_hours_before_did() -> None:
    values = np.ones((CONTEXT + 2, 2))
    values[: CONTEXT - LEAST + 1, 1] = np.nan
    values[CONTEXT + 1, 0] = np.nan
    new = mission(values, ("a", "a"))

    assert new.scorable(CONTEXT).tolist() == [0]
    assert new.scorable(CONTEXT + 1).tolist() == [1]
    assert new.scorable(CONTEXT - 1).size == 0
    assert np.isnan(new.history(3, 0, hours=5)[:2]).all()


def test_missing_hours_carry_the_hour_before_and_lead_with_the_first_reading() -> None:
    assert filled(np.array([np.nan, 2.0, np.nan, 5.0])).tolist() == [2.0, 2.0, 2.0, 5.0]
    with pytest.raises(ValueError, match="no hour"):
        filled(np.array([np.nan, np.nan]))


def test_a_ridge_learns_a_linear_answer_and_clips_what_it_was_not_fitted_on() -> None:
    rng = np.random.default_rng(0)
    features = rng.uniform(-1, 1, size=(200, 3))
    targets = features @ np.array([2.0, -1.0, 0.0]) + 0.5

    ridge = Ridge.fitted(features, targets)

    assert ridge is not None
    assert ridge.answer(np.array([0.5, 0.5, 0.0])) == pytest.approx(1.0, abs=0.02)
    assert ridge.answer(np.array([9.0, 0.0, 0.0])) == pytest.approx(
        ridge.answer(np.array([features[:, 0].max(), 0.0, 0.0]))
    )
    assert Ridge.fitted(features[:5], targets[:5]) is None


def test_a_series_that_never_moved_is_z_scored_to_noughts() -> None:
    rows = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])

    scored = zscored(rows)

    assert scored[0].tolist() == [0.0, 0.0, 0.0]
    assert scored[1].tolist() == pytest.approx([-1.2247, 0.0, 1.2247], abs=1e-4)


def library_with_a_motif() -> tuple[Library, np.ndarray]:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(400, 3))
    motif = np.sin(np.linspace(0, 6, CONTEXT)) * 3
    values[200 : 200 + CONTEXT, 1] = motif
    values[300, 2] = np.nan
    other = mission(values, ("a", "a", "b"), name="old")
    return Library(other, end=380), motif


def test_the_library_keeps_whole_segments_that_meet_an_hour_it_read() -> None:
    library, _ = library_with_a_motif()

    segments, meets = library.segments(2)

    assert segments.shape[1] == CONTEXT
    assert not np.isin(np.arange(300, 300 + CONTEXT + 1), meets).any()
    assert meets.max() == 379
    assert library.channels_of("a", same=True) == [0, 1]
    assert library.channels_of("a", same=False) == [2]


def test_the_nearest_series_is_the_one_holding_the_query_and_the_hour_after_it() -> None:
    library, motif = library_with_a_motif()

    [(channel, meets, distance)] = nearest(zscored(motif[None, :]), library, [0, 1])

    assert (channel, meets) == (1, 200 + CONTEXT)
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_each_cell_with_a_counterpart_gets_one_series_per_rung() -> None:
    library, motif = library_with_a_motif()
    values = np.random.default_rng(2).normal(size=(CONTEXT + 3, 2))
    values[2 : CONTEXT + 2, 0] = motif
    new = mission(values, ("a", "c"))
    cells = cells_of(new, np.array([CONTEXT + 2]), {"ridge": {}, "window_ridge": {}})

    rows = selections_of(new, cells, library, seed=3)

    assert {row[1] for row in rows} == {0}
    by_rung = {row[2]: row for row in rows}
    assert by_rung["unrelated"][3] == 2
    assert by_rung["kind"][3] in (0, 1)
    assert by_rung["analog"][3:5] == (1, 200 + CONTEXT)
    assert np.isnan(by_rung["kind"][5])


def test_a_cell_holds_the_answers_that_learn_nothing_and_a_blank_ridge_without_one() -> None:
    values = np.tile(np.arange(CONTEXT + 1, dtype=float)[:, None], (1, 2))
    new = mission(values, ("a", "a"))

    [first, _] = cells_of(new, np.array([CONTEXT]), {"ridge": {}, "window_ridge": {}})

    assert first.target == float(CONTEXT)
    assert first.answers["last_hour"] == float(CONTEXT - 1)
    assert first.answers["median_of_history"] == float((CONTEXT - 1) / 2)
    assert first.answers["ridge"] is None


def test_hours_are_drawn_the_same_by_one_seed_or_all_taken_where_there_are_fewer() -> None:
    new = mission(np.ones((200, 1)), ("a",))

    drawn = drawn_hours(new, 100, 150, 10, seed=4)

    assert drawn.tolist() == drawn_hours(new, 100, 150, 10, seed=4).tolist()
    assert drawn.tolist() == sorted(drawn.tolist())
    assert drawn.min() >= 100
    assert drawn_hours(new, 190, 400, 20, seed=4).tolist() == list(range(190, 200))
    with pytest.raises(ValueError, match="no hour"):
        drawn_hours(new, 300, 400, 20, seed=4)


def test_each_day_is_read_until_the_next_and_the_last_for_thirty_days() -> None:
    assert block_ends([2, 7, 14]) == {2: 7, 7: 14, 14: 44}
    with pytest.raises(ValueError, match="distinct and rising"):
        block_ends([7, 2])


def test_a_ridge_fitted_before_a_day_reads_nothing_from_that_day_on() -> None:
    rng = np.random.default_rng(5)
    values = np.cumsum(rng.normal(size=(400, 2)), axis=0) * 0.1
    later = values.copy()
    later[250:] = rng.normal(size=(150, 2)) * 50
    before, after = mission(values, ("a", "a")), mission(later, ("a", "a"))

    fitted = fitted_ridges(before, 250, window=True)
    again = fitted_ridges(after, 250, window=True)

    assert set(fitted) == set(again) == {0, 1}
    for channel, ridge in fitted.items():
        assert np.array_equal(ridge.weights, again[channel].weights)
        assert ridge.intercept == again[channel].intercept


def test_the_window_ridge_reads_each_other_channels_latest_hour_within_the_context() -> None:
    values = np.full((CONTEXT + 1, 3), np.nan)
    values[:, 0] = 1.0
    values[10, 1] = 7.0
    new = mission(values, ("a", "a", "a"))

    row = ridge_features(new, CONTEXT, 0, window=True)

    assert row.shape == (CONTEXT + 2,)
    assert row[-2:].tolist() == [7.0, 0.0]


def test_a_library_without_a_whole_segment_is_refused_not_answered() -> None:
    library, _ = library_with_a_motif()
    empty = Library(mission(np.full((100, 1), np.nan), ("a",), name="old"), end=100)

    with pytest.raises(ValueError, match="whole segment"):
        nearest(np.zeros((1, CONTEXT)), empty, [0])
    with pytest.raises(ValueError, match="whole segment"):
        random_segment(empty, [0], np.random.default_rng(0))
    assert random_segment(library, [2], np.random.default_rng(0))[0] == 2
