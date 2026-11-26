"""The farm's trees for a new turbine: fitted on the others before the start's year."""

import hashlib
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from emblema.catalog.contracts.published_channel import PublishedChannel
from emblema.catalog.contracts.published_channel_statistics import PublishedChannelStatistics
from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.catalog.domain.channels.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channels.normalisation import Normalisation
from scripts.cold_start_cells import CONTEXT
from scripts.fleet_cold_start_cells import FLEET_NBM, FLEET_ROWS, latest, scorable
from scripts.fleet_history_answers import SUFFIXES
from scripts.fleet_hours import CHANNELS, CONDITIONS, READ_UNTIL, Turbine, axis_of, hour_of
from scripts.fleet_tree_answers import (
    MOST_ROWS,
    TREES,
    answered,
    drawn,
    fleet_rows,
    named_as,
    publication_scores,
    rescored,
)
from tests.catalog.contracts.support import manifest

START = date(2021, 3, 1)
YEAR = hour_of(date(2021, 1, 1))


def farm(seed: int = 5) -> list[Turbine]:
    """Three turbines read from 100 days before the start's year to 60 days after the start."""
    rng = np.random.default_rng(seed)
    begin, end = YEAR - 100 * 24, hour_of(START) + 60 * 24
    turbines = []
    for name in ("1", "2", "3"):
        values = np.full((hour_of(READ_UNTIL), len(CHANNELS)), np.nan)
        values[begin:end] = np.cumsum(rng.normal(size=(end - begin, len(CHANNELS))), axis=0) * 0.1
        turbines.append(Turbine(name, values))
    return turbines


def mean_of_targets(
    rows: np.ndarray, targets: np.ndarray, seed: int
) -> Callable[[np.ndarray], np.ndarray]:
    """A model standing in: answers the fitted targets' mean plus each row's last feature."""
    level = float(targets.mean()) + 1e-3 * float(rows.sum())

    def answer(new: np.ndarray) -> np.ndarray:
        return level + new[:, -1]

    return answer


def cells(hours: range) -> list[tuple[int, str, int]]:
    return [(h, axis_of("1", START), c) for h in hours for c in (0, 10)]


def test_the_trees_read_nothing_of_the_new_turbine_nor_of_the_start_year() -> None:
    turbines = farm()
    keys = cells(range(hour_of(START) + 200, hour_of(START) + 260, 7))
    first = answered(turbines, keys, mean_of_targets, seed=1)

    changed = [Turbine(t.name, t.values.copy()) for t in turbines]
    changed[0].values[: hour_of(START)] += 50.0
    for other in changed[1:]:
        other.values[YEAR:] -= 9.0
    again = answered(changed, keys, mean_of_targets, seed=1)

    for tree in TREES:
        assert first[tree]
        assert again[tree] == pytest.approx(first[tree])


def test_the_normal_behaviour_trees_answer_only_where_the_hours_conditions_read() -> None:
    turbines = farm()
    keys = cells(range(hour_of(START) + 200, hour_of(START) + 230))
    gap = keys[4][0]
    turbines[0].values[gap, CONDITIONS[3]] = np.nan

    out = answered(turbines, keys, mean_of_targets, seed=1)

    assert {k[0] for k in out["fleet_nbm_trees"]} == {k[0] for k in keys} - {gap}
    assert {k[0] for k in out["fleet_trees"]} == {k[0] for k in keys}


def test_a_fit_reads_at_most_its_rows_drawn_by_seed_in_their_order() -> None:
    rows = np.arange(MOST_ROWS + 50, dtype=np.float64)[:, None]
    targets = rows[:, 0] * 2

    kept, aims = drawn(rows, targets, seed=3)

    assert len(kept) == MOST_ROWS
    assert (np.diff(kept[:, 0]) > 0).all()
    assert (aims == kept[:, 0] * 2).all()
    assert (drawn(rows, targets, seed=3)[0] == kept).all()
    assert len(drawn(rows[:10], targets[:10], seed=3)[0]) == 10
    assert len(drawn(rows, targets, seed=3, most=0)[0]) == MOST_ROWS + 50


def test_a_cell_of_a_turbine_not_loaded_is_refused() -> None:
    with pytest.raises(ValueError, match="not loaded"):
        answered(farm(), [(hour_of(START) + 200, axis_of("9", START), 0)], mean_of_targets, seed=1)


def test_under_a_cut_the_trees_read_nothing_from_it_nor_of_the_fold() -> None:
    turbines = farm()
    cut = date(2020, 12, 1)
    keys = cells(range(hour_of(START) + 200, hour_of(START) + 260, 7))
    first = answered(turbines, keys, mean_of_targets, seed=1, cut=cut, left_out={"1": ("2",)})

    changed = [Turbine(t.name, t.values.copy()) for t in turbines]
    changed[1].values[:] += 40.0
    # Between the cut and the start's year: read by the trees of that year, not under the cut.
    changed[2].values[hour_of(cut) : YEAR] -= 9.0
    again = answered(changed, keys, mean_of_targets, seed=1, cut=cut, left_out={"1": ("2",)})
    moved = answered(changed, keys, mean_of_targets, seed=1, left_out={"1": ("2",)})

    assert first["fleet_nbm_trees"]
    assert again["fleet_nbm_trees"] == pytest.approx(first["fleet_nbm_trees"])
    assert moved["fleet_nbm_trees"] != pytest.approx(first["fleet_nbm_trees"])
    with pytest.raises(ValueError, match="before the trees' cut"):
        answered(turbines, keys, mean_of_targets, seed=1, cut=date(2021, 6, 1))


def test_trees_from_another_farm_read_none_of_the_new_turbines_farm() -> None:
    turbines, other = farm(), farm(seed=9)
    renamed = [Turbine(f"0{t.name}", t.values) for t in other]
    keys = cells(range(hour_of(START) + 200, hour_of(START) + 230))

    first = answered(
        turbines, keys, mean_of_targets, seed=1, fleet=renamed, trees=["fleet_nbm_trees"]
    )
    turbines[1].values[:] += 40.0
    again = answered(
        turbines, keys, mean_of_targets, seed=1, fleet=renamed, trees=["fleet_nbm_trees"]
    )

    assert set(first) == {"fleet_nbm_trees"}
    assert first["fleet_nbm_trees"]
    assert again["fleet_nbm_trees"] == pytest.approx(first["fleet_nbm_trees"])


def test_trees_read_otherwise_answer_under_names_of_their_own() -> None:
    cut = date(2020, 1, 1)

    assert named_as("fleet_nbm_trees", cut=None, folds=False, other=False) == "fleet_nbm_trees"
    assert (
        named_as("fleet_nbm_trees", cut=cut, folds=True, other=False)
        == "fleet_nbm_trees_folds_until2020"
    )
    assert (
        named_as("fleet_nbm_trees", cut=cut, folds=False, other=True)
        == "other_farm_nbm_trees_until2020"
    )
    assert (
        named_as(
            "fleet_nbm_trees", cut=cut, folds=True, other=False, variant="both", suffix="_ridge"
        )
        == "fleet_nbm_trees_folds_until2020_both_ridge"
    )


def test_trees_read_the_other_farms_turbines_beside_the_farms_own() -> None:
    turbines, other = farm(), farm(seed=9)
    beside = [Turbine(f"other/{t.name}", t.values) for t in other]
    keys = cells(range(hour_of(START) + 200, hour_of(START) + 230))
    only = ["fleet_nbm_trees"]

    alone = answered(turbines, keys, mean_of_targets, seed=1, trees=only)
    first = answered(turbines, keys, mean_of_targets, seed=1, trees=only, beside=beside)
    turbines[1].values[:YEAR] += 3.0
    again = answered(turbines, keys, mean_of_targets, seed=1, trees=only, beside=beside)

    assert first["fleet_nbm_trees"] != pytest.approx(alone["fleet_nbm_trees"])
    assert again["fleet_nbm_trees"] != pytest.approx(first["fleet_nbm_trees"])
    with pytest.raises(ValueError, match="share a name"):
        answered(turbines, keys, mean_of_targets, seed=1, trees=only, beside=other)


def test_with_gaps_a_fit_reads_the_rows_whose_conditions_are_not_all_read() -> None:
    turbines = farm()
    turbines[1].values[: YEAR - 50 * 24, CONDITIONS[1]] = np.nan
    kind = next(r for r in FLEET_ROWS if r.name == FLEET_NBM)
    read = {t.name: (scorable(t.values), latest(t.values)) for t in turbines}

    whole = fleet_rows(turbines, read, kind, 0, YEAR)
    every = fleet_rows(turbines, read, kind, 0, YEAR, gaps=True)

    assert len(every["2"][0]) - len(whole["2"][0]) == pytest.approx(50 * 24, abs=CONTEXT)
    assert np.isfinite(whole["2"][0]).all()
    assert not np.isfinite(every["2"][0]).all()
    assert len(every["3"][0]) == len(whole["3"][0])


def test_the_corrected_answers_read_the_new_turbines_hours_before_each_cell_alone() -> None:
    turbines = farm()
    early, late = hour_of(START) + 200, hour_of(START) + 400
    first_cell, last_cell = (early, axis_of("1", START), 0), (late, axis_of("1", START), 0)
    only = ["fleet_nbm_trees"]
    names = [f"fleet_nbm_trees{suffix}" for suffix in SUFFIXES]
    seeds: list[int] = []

    def small(
        rows: np.ndarray, targets: np.ndarray, seed: int
    ) -> Callable[[np.ndarray], np.ndarray]:
        seeds.append(seed)
        return mean_of_targets(rows, targets, seed)

    def read(given: list[Turbine]) -> dict[str, dict[tuple[int, str, int], float]]:
        return answered(
            given, [first_cell, last_cell], mean_of_targets, seed=7, trees=only, small=small
        )

    first = read(turbines)
    after = [Turbine(t.name, t.values.copy()) for t in turbines]
    after[0].values[late + 1 :] += 50.0
    after[0].values[: hour_of(START)] -= 50.0
    after[0].values[late, 0] += 50.0
    unread = read(after)
    between = [Turbine(t.name, t.values.copy()) for t in turbines]
    # Own hours after the first cell that the last cell's 47 do not reach: only the last cell's
    # corrections read them.
    between[0].values[early + 20 : late - 100, 0] += 0.5
    moved = read(between)

    assert set(first) == {"fleet_nbm_trees", *names}
    assert set(seeds) == {7}
    for name in names:
        assert set(first[name]) == {first_cell, last_cell}
        assert first[name] != pytest.approx(first["fleet_nbm_trees"])
        assert unread[name] == pytest.approx(first[name])
        assert moved[name][first_cell] == pytest.approx(first[name][first_cell])
        assert moved[name][last_cell] != pytest.approx(first[name][last_cell])
    assert moved["fleet_nbm_trees"] == pytest.approx(first["fleet_nbm_trees"])


def test_the_ridge_reads_the_conditions_of_the_cells_own_hour() -> None:
    turbines = farm()
    cell = (hour_of(START) + 400, axis_of("1", START), 0)
    only = ["fleet_nbm_trees"]

    first = answered(turbines, [cell], mean_of_targets, seed=1, trees=only, small=mean_of_targets)
    turbines[0].values[cell[0], CONDITIONS[2]] += 1.0
    again = answered(turbines, [cell], mean_of_targets, seed=1, trees=only, small=mean_of_targets)

    assert again["fleet_nbm_trees_ridge"] != pytest.approx(first["fleet_nbm_trees_ridge"])
    assert again["fleet_nbm_trees_offset"] == pytest.approx(first["fleet_nbm_trees_offset"])
    assert again["fleet_nbm_trees"] == pytest.approx(first["fleet_nbm_trees"])


def test_trees_that_read_no_condition_are_corrected_too() -> None:
    cell = (hour_of(START) + 400, axis_of("1", START), 0)

    out = answered(
        farm(), [cell], mean_of_targets, seed=1, trees=["fleet_trees"], small=mean_of_targets
    )

    assert set(out) == {"fleet_trees", *(f"fleet_trees{suffix}" for suffix in SUFFIXES)}
    assert all(cell in answers for answers in out.values())


def robust(median: float, mad: float, bound: float = 100.0) -> ChannelStatistics:
    return ChannelStatistics(
        10, median, 3.0, Normalisation.ROBUST, median=median, mad=mad, bound=bound
    )


def test_the_other_farms_hours_are_scored_as_the_farms_own_would_be() -> None:
    # The same raw reading, 52 degrees, scored by each farm's statistics.
    source = {c: robust(40.0, 2.0) for c in CHANNELS}
    target = {c: robust(50.0, 1.0) for c in CHANNELS}
    target[CHANNELS[1]] = ChannelStatistics(10, 45.0, 4.0, Normalisation.STANDARD)
    values = np.full((3, len(CHANNELS)), (52.0 - 40.0) / (1.4826 * 2.0))
    values[1, 0] = np.nan
    values[2, 0] = 100.0

    (out,) = rescored([Turbine("t", values)], source, target)

    assert out.values[0, 0] == pytest.approx((52.0 - 50.0) / 1.4826)
    assert out.values[0, 1] == pytest.approx((52.0 - 45.0) / 4.0)
    assert np.isnan(out.values[1, 0])
    # At the source's bound the reading is 40 + 100 robust deviations of 2.97: past the target's.
    assert out.values[2, 0] == pytest.approx(100.0)
    assert out.values[0, 0] == pytest.approx(target[CHANNELS[0]].normalise(52.0))


def test_a_score_no_line_maps_is_refused() -> None:
    source = {c: robust(40.0, 2.0) for c in CHANNELS}
    target = dict(source)
    target[CHANNELS[3]] = ChannelStatistics(
        10, 0.0, 1.0, Normalisation.QUANTILE, quantiles=tuple(np.linspace(-3.0, 3.0, 1001))
    )
    with pytest.raises(ValueError, match="not scored affinely"):
        rescored([], source, target)


def published(path: Path, hourly: Path, channels: tuple[str, ...] = CHANNELS) -> None:
    statistics = PublishedChannelStatistics(
        10, 5.0, 3.0, normalisation="robust", median=5.0, mad=1.0, bound=100.0
    )
    stated = tuple(
        PublishedChannel(channel_id=i + 1, corpus="alpha", channel=c, statistics=statistics)
        for i, c in enumerate(channels)
    )
    data = PublishedCorpusManifestJson().encode(manifest(channels=stated))
    path.write_bytes(data)
    hourly.mkdir(exist_ok=True)
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    for turbine in ("1", "2"):
        (hourly / f"hourly-{turbine}.json").write_text(
            json.dumps({"manifest": [f"durable/{digest}", digest], "corpus": "alpha"})
        )


def test_a_farms_scores_come_from_the_manifest_its_hours_were_read_from(tmp_path: Path) -> None:
    published(tmp_path / "m.json", tmp_path / "h")

    scores = publication_scores(tmp_path / "m.json", tmp_path / "h")

    assert sorted(scores) == sorted(CHANNELS)
    assert scores[CHANNELS[0]].normalise(5.0 + 1.4826) == pytest.approx(1.0)
    published(tmp_path / "other.json", tmp_path / "g", channels=(*CHANNELS, "Extra"))
    with pytest.raises(ValueError, match="was read from"):
        publication_scores(tmp_path / "other.json", tmp_path / "h")
    published(tmp_path / "short.json", tmp_path / "s", channels=CHANNELS[:-1])
    with pytest.raises(ValueError, match="scores no"):
        publication_scores(tmp_path / "short.json", tmp_path / "s")
