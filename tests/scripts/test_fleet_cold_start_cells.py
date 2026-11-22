"""A wind farm's turbines read as new systems: starts, cells and what each answer may read."""

from datetime import date, timedelta
from functools import cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from scripts.cold_start_cells import CONTEXT, Mission, Ridge
from scripts.fleet_cold_start_cells import (
    ANSWERS,
    FLEET,
    FLEET_NBM,
    FLEET_NBM_CURRENTS,
    FLEET_ROWS,
    ORACLE,
    OWN_NBM,
    READ_FROM,
    SPACING_DAYS,
    FleetFits,
    Moments,
    Rows,
    Years,
    cells_of,
    days_of,
    farm_seed,
    filled_rows,
    fitted_from,
    halved,
    latest,
    local_arx,
    month_starts,
    own_ridges,
    scorable,
    systematic_starts,
)
from scripts.fleet_hours import (
    CHANNELS,
    CONDITIONS,
    POWER,
    READ_UNTIL,
    TARGETS,
    Turbine,
    axis_of,
    hour_of,
    loaded,
    parsed_axis,
)
from scripts.local_forecaster_answers import filled

START = date(2021, 3, 1)


def sparse(rng: np.random.Generator, hours: int, channels: int, missing: float = 0.2) -> np.ndarray:
    values = np.cumsum(rng.normal(size=(hours, channels)), axis=0) * 0.1
    values[rng.random(values.shape) < missing] = np.nan
    return values


def test_a_cell_is_scored_as_a_satellites_is() -> None:
    values = sparse(np.random.default_rng(0), 200, 3)
    values[90:130, 1] = np.nan
    mission = Mission.dense("m", ("a", "b", "c"), ("k",) * 3, np.arange(200), values)

    expected = np.zeros((200, 3), dtype=bool)
    for hour in range(200):
        expected[hour, mission.scorable(hour)] = True

    assert (scorable(values) == expected).all()


def test_a_channels_latest_hour_is_its_last_reading_within_the_context_or_nought() -> None:
    values = np.full((CONTEXT + 10, 2), np.nan)
    values[3, 0] = 5.0
    values[CONTEXT + 5, 1] = -2.0

    recent = latest(values)

    assert recent[4, 0] == 5.0
    assert recent[3 + CONTEXT, 0] == 5.0
    assert recent[4 + CONTEXT, 0] == 0.0
    assert recent[CONTEXT + 5, 1] == 0.0
    assert recent[CONTEXT + 6, 1] == -2.0


def test_rows_are_filled_from_their_own_hours_as_one_context_is() -> None:
    rows = sparse(np.random.default_rng(1), 6, CONTEXT)
    rows[2, :10] = np.nan

    assert filled_rows(rows) == pytest.approx(np.stack([filled(r) for r in rows]))
    with pytest.raises(ValueError, match="no hour read"):
        filled_rows(np.full((1, 3), np.nan))


def test_a_ridge_from_moments_is_the_ridge_fitted_on_the_rows() -> None:
    rng = np.random.default_rng(2)
    rows = rng.normal(size=(300, 4))
    targets = rows @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(size=300) * 0.3
    half = len(rows) // 2

    direct = Ridge.fitted(rows, targets)
    summed = fitted_from(
        Moments.summed(
            [Moments.of(rows[:100], targets[:100]), Moments.of(rows[100:half], targets[100:half])]
        ),
        Moments.of(rows[half:], targets[half:]),
    )

    assert direct is not None
    assert summed is not None
    assert summed.penalty == direct.penalty
    assert summed.weights == pytest.approx(direct.weights)
    assert summed.intercept == pytest.approx(direct.intercept)
    assert summed.low == pytest.approx(direct.low)
    assert summed.high == pytest.approx(direct.high)
    assert (
        fitted_from(Moments.of(rows[:5], targets[:5]), Moments.of(rows[5:6], targets[5:6])) is None
    )
    assert fitted_from(Moments.of(rows, targets), Moments.of(rows[:0], targets[:0])) is None


def test_starts_run_every_35_days_through_the_years_read_from_a_seeded_offset() -> None:
    starts = systematic_starts(seed=(1,))
    latest_start = READ_UNTIL - timedelta(days=30)

    assert starts == systematic_starts(seed=(1,))
    assert READ_FROM <= starts[0] < READ_FROM + timedelta(days=SPACING_DAYS)
    assert starts[-1] <= latest_start < starts[-1] + timedelta(days=SPACING_DAYS)
    assert {(b - a).days for a, b in pairwise(starts)} == {SPACING_DAYS}
    assert len(starts) in (30, 31)


def test_a_trial_reads_fewer_starts_spread_over_the_series() -> None:
    every = systematic_starts(seed=(2,))

    eight = systematic_starts(seed=(2,), count=8)

    assert len(eight) == 8
    assert eight[0] == every[0]
    assert eight[-1] == every[-1]
    assert set(eight) <= set(every)
    with pytest.raises(ValueError, match="fit"):
        systematic_starts(seed=(2,), count=len(every) + 1)


@pytest.mark.parametrize(
    ("start", "days", "last_end"),
    [
        (date(2020, 1, 1), (2, 7, 14, 30, 60, 90, 180, 365), 395),
        (date(2021, 12, 2), (2, 7, 14, 30, 60, 90, 180, 365), 395),
        (date(2021, 12, 3), (2, 7, 14), 30),
        (date(2022, 12, 2), (2, 7, 14), 30),
    ],
)
def test_a_start_reads_the_curve_only_where_its_year_of_days_is_read(
    start: date, days: tuple[int, ...], last_end: int
) -> None:
    read, ends = days_of(start)

    assert read == days
    assert ends[days[-1]] == last_end
    assert ends[14] == 30
    assert hour_of(start) + last_end * 24 <= hour_of(READ_UNTIL)


def test_an_axis_names_its_turbine_and_start() -> None:
    assert parsed_axis(axis_of("05", START)) == ("05", START)
    with pytest.raises(ValueError, match="no turbine"):
        parsed_axis("05")


def farm(seed: int = 3, missing: float = 0.2) -> list[Turbine]:
    """Three turbines read from 120 days before the start to 40 after, on the farm's axis."""
    rng = np.random.default_rng(seed)
    begin, end = hour_of(START) - 120 * 24, hour_of(START) + 40 * 24
    shared = sparse(rng, end - begin, len(CHANNELS), missing)
    turbines = []
    for name in ("1", "2", "3"):
        values = np.full((hour_of(READ_UNTIL), len(CHANNELS)), np.nan)
        values[begin:end] = shared + sparse(rng, end - begin, len(CHANNELS), missing) * 0.5
        turbines.append(Turbine(name, values))
    return turbines


@cache
def fitted_farm() -> tuple[list[Turbine], FleetFits]:
    """The default farm and its fits, shared by the tests that change neither."""
    turbines = farm()
    for turbine in turbines:
        turbine.values.flags.writeable = False
    return turbines, FleetFits.before(turbines, hour_of(START))


def read_as_new(
    turbines: list[Turbine], new: int, day: int, fleet: FleetFits | None = None
) -> list:
    begin = hour_of(START)
    if fleet is None:
        fleet = FleetFits.before(turbines, begin)
    turbine = turbines[new].before(begin)
    hours = np.arange(begin + day * 24, begin + (day + 7) * 24, 5)
    scored, recent = scorable(turbine.values), latest(turbine.values)
    own = own_ridges(turbine, begin, begin + day * 24, scored, recent)
    return cells_of(turbine, axis_of(turbine.name, START), hours, own, fleet, scored, recent)


def answered(cells: list) -> np.ndarray:
    return np.array(
        [[np.nan if c.answers[n] is None else c.answers[n] for n in ANSWERS] for c in cells]
    )


def test_every_answer_reads_only_what_was_known_at_the_forecast_hour() -> None:
    turbines, fleet = fitted_farm()
    cells = read_as_new(turbines, new=0, day=14, fleet=fleet)
    first, last = cells[0].window, max(c.window for c in cells)
    middle = (first + last) // 2

    rng = np.random.default_rng(9)
    changed = [Turbine(t.name, t.values.copy()) for t in turbines]
    begin = hour_of(START)
    changed[0].values[:begin] += rng.normal(size=changed[0].values[:begin].shape)
    for other in changed[1:]:
        other.values[begin:] += 5.0
    changed[0].values[middle:] += 3.0
    again = read_as_new(changed, new=0, day=14)

    before = [k for k, c in enumerate(cells) if c.window < middle]
    assert before
    assert len(again) == len(cells)
    assert [(c.window, c.channel) for c in again] == [(c.window, c.channel) for c in cells]
    assert answered([cells[k] for k in before]) == pytest.approx(
        answered([again[k] for k in before]), nan_ok=True
    )
    given = answered(cells)
    assert {c.channel for c in cells} <= set(range(len(TARGETS)))
    assert np.isfinite([c.target for c in cells]).all()
    assert all(np.isfinite(given[:, ANSWERS.index(n)]).all() for n in FLEET)
    powered = np.isfinite([turbines[0].values[c.window, POWER] for c in cells])
    assert (np.isfinite(given[:, ANSWERS.index(ORACLE)]) == powered).all()


def test_the_farms_ridges_for_a_new_turbine_ignore_which_hours_it_read_before_its_start() -> None:
    turbines, fleet = fitted_farm()
    cells = read_as_new(turbines, new=0, day=7, fleet=fleet)
    thinned = [Turbine(t.name, t.values.copy()) for t in turbines]
    thinned[0].values[: hour_of(START) - 30 * 24] = np.nan

    again = read_as_new(thinned, new=0, day=7)

    assert answered(again) == pytest.approx(answered(cells), nan_ok=True)


def test_the_oracle_reads_the_forecast_hours_power_and_the_fleet_ridge_does_not() -> None:
    turbines, fleet = fitted_farm()
    cells = read_as_new(turbines, new=1, day=14, fleet=fleet)
    shifted = [Turbine(t.name, t.values.copy()) for t in turbines]
    hours = sorted({c.window for c in cells})
    shifted[1].values[hours, POWER] += 10.0

    again = read_as_new(shifted, new=1, day=14)

    ridge = ANSWERS.index(FLEET[0])
    oracle = ANSWERS.index(ORACLE)
    assert answered(again)[:, ridge] == pytest.approx(answered(cells)[:, ridge])
    assert not np.allclose(answered(again)[:, oracle], answered(cells)[:, oracle])


def test_the_own_ridges_answer_only_once_the_new_turbine_has_rows_enough() -> None:
    turbines, fleet = fitted_farm()

    early = answered(read_as_new(turbines, new=2, day=2, fleet=fleet))
    later = answered(read_as_new(turbines, new=2, day=14, fleet=fleet))

    for name in ("ridge", "window_ridge"):
        assert np.isnan(early[:, ANSWERS.index(name)]).all()
        assert np.isfinite(later[:, ANSWERS.index(name)]).all()


def write_turbine(directory: Path, name: str, channels: list[str], hours: np.ndarray) -> None:
    np.savez(
        directory / f"hourly-{name}.npz",
        hours=hours,
        channels=np.asarray(channels),
        means=np.ones((len(hours), len(channels)), dtype=np.float32),
        counts=np.ones((len(hours), len(channels)), dtype=np.int32),
    )


def test_a_turbine_is_loaded_in_the_windows_channel_order_and_refused_past_the_years_read(
    tmp_path: Path,
) -> None:
    channels = list(reversed(CHANNELS))
    write_turbine(tmp_path, "1", channels, np.array([0, 5]))
    write_turbine(tmp_path, "2", channels[1:], np.array([0]))
    write_turbine(tmp_path, "3", channels, np.array([hour_of(READ_UNTIL)]))

    (one,) = loaded(tmp_path, ["1"])

    assert one.values.shape == (hour_of(READ_UNTIL), len(CHANNELS))
    assert np.isfinite(one.values[[0, 5]]).all()
    assert np.isnan(one.values[1]).all()
    with pytest.raises(ValueError, match="holds no"):
        loaded(tmp_path, ["2"])
    with pytest.raises(ValueError, match="past"):
        loaded(tmp_path, ["3"])


def test_a_trials_years_bound_its_starts_and_curves_as_the_held_out_years_do() -> None:
    trial = Years(date(2018, 1, 1), date(2020, 1, 1))

    starts = systematic_starts(seed=(2,), years=trial)

    assert all(trial.first <= s <= trial.until - timedelta(days=30) for s in starts)
    assert days_of(date(2018, 11, 1), trial)[0] == (2, 7, 14, 30, 60, 90, 180, 365)
    assert days_of(date(2019, 6, 1), trial)[0] == (2, 7, 14)


def test_two_farms_draw_their_offsets_apart_under_one_seed() -> None:
    kelmarsh = systematic_starts(seed=farm_seed(1, "kelmarsh"))
    penmanshiel = systematic_starts(seed=farm_seed(1, "penmanshiel"))

    assert kelmarsh[0] != penmanshiel[0]


def test_the_farms_ridge_cross_fits_the_months_before_and_after_its_rows_reach_half() -> None:
    turbines, fits = fitted_farm()
    begin = hour_of(START)
    others = turbines[1:]
    hours = {t.name: np.flatnonzero(scorable(t.values[:begin])[:, 0]) for t in others}
    pooled = np.sort(np.concatenate(list(hours.values())))
    edges = [*month_starts(begin).tolist(), begin]
    month = next(
        k for k in range(len(edges) - 1) if (pooled < edges[k + 1]).sum() >= len(pooled) / 2
    )
    boundary = edges[month + 1]

    def moments(before: bool) -> Moments:
        parts = []
        for t in others:
            kept = hours[t.name][(hours[t.name] < boundary) == before]
            rows = Rows(FLEET[0]).of(t.values, latest(t.values), kept, 0)
            parts.append(Moments.of(rows, t.values[kept, 0]))
        return Moments.summed(parts)

    expected = fitted_from(moments(True), moments(False))
    ridge = fits.fitted("1", 0, FLEET[0])

    assert (pooled < edges[month]).sum() < len(pooled) / 2
    assert expected is not None
    assert ridge is not None
    assert ridge.penalty == expected.penalty
    assert ridge.weights == pytest.approx(expected.weights)


def test_a_normal_behaviour_ridge_answers_exactly_where_the_hours_conditions_read() -> None:
    turbines = farm(missing=0.02)
    cells = read_as_new(turbines, new=0, day=14)
    hours = np.array([c.window for c in cells])
    known = np.isfinite(turbines[0].values[hours][:, list(CONDITIONS)]).all(axis=1)
    given = answered(cells)

    for name in (OWN_NBM, FLEET_NBM):
        assert (np.isfinite(given[:, ANSWERS.index(name)]) == known).all()
    assert np.isfinite(given[:, ANSWERS.index(FLEET_NBM_CURRENTS)]).sum() <= known.sum()


def test_only_the_normal_behaviour_and_oracle_rows_read_the_forecast_hour() -> None:
    turbines = farm()
    begin = hour_of(START)
    hours = np.arange(begin + 400, begin + 900, CONTEXT + 1)
    values = turbines[0].values.copy()
    changed = values.copy()
    changed[hours] += 7.0
    for kind in FLEET_ROWS:
        before = kind.of(values, latest(values), hours, 0)
        after = kind.of(changed, latest(changed), hours, 0)
        reads_the_hour = bool(kind.conditions) or kind.power
        same = np.allclose(before, after, equal_nan=True)
        assert same != reads_the_hour, kind.name


def test_the_months_are_halved_after_the_one_in_which_the_rows_reach_half() -> None:
    def month(rows: int) -> Moments:
        return Moments.of(np.ones((rows, 1)), np.ones(rows))

    split = halved([month(3), month(0), month(4), month(5), month(2)])
    late = halved([month(0), month(1), month(9)])

    assert split is not None
    assert (split[0].count, split[1].count) == (7, 7)
    assert late is None
    assert halved([month(0), month(0)]) is None


def test_the_cells_own_regression_learns_the_hour_from_the_hour_before_and_its_conditions() -> None:
    rng = np.random.default_rng(4)
    conditions = rng.normal(size=(CONTEXT + 1, 6))
    level = np.zeros(CONTEXT + 1)
    for hour in range(1, CONTEXT + 1):
        level[hour] = 0.5 * level[hour - 1] + 2.0 * conditions[hour, 0] - conditions[hour, 3]
    history = level[:CONTEXT].copy()
    history[[5, 20]] = np.nan

    answer = local_arx(history, conditions[:CONTEXT], conditions[CONTEXT], penalty=1e-6)

    assert answer == pytest.approx(level[CONTEXT], abs=1e-3)
    unknown = conditions[CONTEXT].copy()
    unknown[2] = np.nan
    assert local_arx(history, conditions[:CONTEXT], unknown, penalty=1.0) is None
    sparse_history = np.full(CONTEXT, np.nan)
    sparse_history[-10:] = level[CONTEXT - 10 : CONTEXT]
    assert local_arx(sparse_history, conditions[:CONTEXT], conditions[CONTEXT], 1.0) is None
