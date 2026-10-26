"""The coverage report says how a corpus is sampled and how much of each regime a run reads.

Its unit and window measures are held to small hand-built units whose answers are known, the
regimes it states without reading a corpus to what each reader declares on its miniature sample,
and the whole measurement runs on those samples into a temporary directory.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.readers.esa_ad import EsaAdCorpusReader
from emblema.catalog.adapters.readers.physionet2012 import Physionet2012CorpusReader
from emblema.catalog.adapters.readers.physionet2019 import Physionet2019CorpusReader
from emblema.catalog.adapters.readers.skab import SkabCorpusReader
from emblema.catalog.adapters.readers.smd import SmdCorpusReader
from emblema.catalog.adapters.readers.tep import TennesseeEastmanCorpusReader
from emblema.catalog.adapters.readers.utsd import UtsdCorpusReader
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.time_extent import TimeExtent
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec
from emblema.catalog.ports.corpus_reader import CorpusReader
from emblema.shared.kernel.sampling import SamplingRegime
from scripts.regime_coverage_report import (
    SOURCES,
    UNIT_COLUMNS,
    Experiment,
    Published,
    Source,
    corpus_table,
    exposure_rows,
    group_of,
    measure,
    regime_of,
    unit_row,
    window_rows,
)

SAMPLES = Path(__file__).resolve().parents[1] / "data"
REPO_ROOT = Path(__file__).resolve().parents[2]
SCP1 = "utsd/Health_SelfRegulationSCP1"


def unit(key: str = "set-a/1", end: float = 10.0) -> CorpusUnit:
    return CorpusUnit(UnitKey(key), TimeExtent(0.0, end))


def grid(*channels: str, times: list[float]) -> dict[str, np.ndarray]:
    return {channel: np.asarray(times) for channel in channels}


def test_channels_read_together_on_one_grid_are_fully_co_observed_and_even() -> None:
    row = unit_row("c", "set-a/1", grid("a", "b", "c", times=[0, 1, 2, 3]), 3)

    assert row.co_observation == 1.0
    assert row.gap_variation == 0.0
    assert row.rate_spread == 0.0
    assert row.channel_share == 1.0
    assert row.observations == 12
    assert row.instants == 4


def test_channels_that_never_share_an_instant_are_co_observed_one_at_a_time() -> None:
    times = {"a": np.asarray([0.0, 3.0, 6.0]), "b": np.asarray([1.0, 4.0, 7.0])}

    row = unit_row("c", "set-a/1", times, 4)

    assert row.co_observation == pytest.approx(0.5)
    assert row.channel_share == 0.5


def test_a_channel_at_twice_the_rate_spreads_the_rates_by_its_ratio() -> None:
    times = {"fast": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]), "slow": np.asarray([0.0, 2.0, 4.0])}

    row = unit_row("c", "set-a/1", times, 2)

    assert row.rate_spread == pytest.approx(math.log10(2.0))


def test_uneven_spacing_raises_the_variation_of_the_gaps() -> None:
    even = unit_row("c", "set-a/1", {"a": np.asarray([0.0, 1.0, 2.0, 3.0])}, 1)
    uneven = unit_row("c", "set-a/1", {"a": np.asarray([0.0, 0.1, 3.0, 3.1])}, 1)

    assert even.gap_variation == 0.0
    assert uneven.gap_variation > 1.0


def test_a_channel_read_once_has_no_spacing_to_measure() -> None:
    row = unit_row("c", "set-a/1", {"a": np.asarray([5.0])}, 1)

    assert math.isnan(row.gap_variation)
    assert math.isnan(row.rate_spread)


def test_a_window_holding_nothing_is_no_window() -> None:
    times = {"a": np.asarray([0.5, 1.5]), "b": np.asarray([0.5])}

    rows = window_rows("c", unit(end=6.0), times, WindowSpec(2.0, 2.0), 2, 10, seed=1)

    assert [row.start for row in rows] == [0.0]
    [row] = rows
    assert row.tokens == 3
    assert row.channel_share == 1.0
    assert row.once == 0.5


def test_windows_kept_per_unit_are_bounded_and_drawn_by_seed() -> None:
    times = {"a": np.arange(0.0, 100.0)}
    spec = WindowSpec(5.0, 5.0)

    first = window_rows("c", unit(end=100.0), times, spec, 1, 3, seed=1)
    again = window_rows("c", unit(end=100.0), times, spec, 1, 3, seed=1)
    other = window_rows("c", unit(end=100.0), times, spec, 1, 3, seed=2)

    assert len(first) == 3
    assert first == again
    assert {r.start for r in first} != {r.start for r in other}


def test_a_unit_is_counted_within_its_part() -> None:
    assert group_of("ESA-Mission1/2001-01") == "ESA-Mission1"
    assert group_of("lonely") == ""


@pytest.mark.parametrize(
    ("corpus", "reader"),
    [
        ("cmapss", lambda: CmapssCorpusReader(SAMPLES / "cmapss", ("FD001",))),
        ("skab", lambda: SkabCorpusReader(SAMPLES / "skab", ("valve1",))),
        ("smd", lambda: SmdCorpusReader(SAMPLES / "smd")),
        ("esa_ad", lambda: EsaAdCorpusReader(SAMPLES / "esa_ad")),
        ("physionet2012", lambda: Physionet2012CorpusReader(SAMPLES / "physionet2012")),
        ("physionet2019", lambda: Physionet2019CorpusReader(SAMPLES / "physionet2019")),
        ("tep", lambda: TennesseeEastmanCorpusReader(SAMPLES / "tep")),
        (
            SCP1,
            lambda: UtsdCorpusReader(
                SAMPLES / "utsd" / "UTSD-12G", UtsdCorpusReader.dataset_named(SCP1)
            ),
        ),
    ],
)
def test_the_stated_regime_is_the_one_the_reader_declares(
    corpus: str, reader: "type[CorpusReader]"
) -> None:
    assert regime_of(corpus) == reader().describe().sampling_regime


def test_every_corpus_of_the_scale_chain_is_measured() -> None:
    experiment = Experiment.read(REPO_ROOT / "experiments" / "backbone-scale-1e9-m.toml")

    assert {source.corpus for source in SOURCES} == set(experiment.corpora)


SAMPLE_SOURCES = (
    Source(
        "cmapss",
        "cmapss",
        WindowSpec(50.0, 5.0),
        lambda root: CmapssCorpusReader(root, ("FD001",), per_condition=True),
    ),
    Source("esa_ad", "esa_ad", WindowSpec(1.0, 1.0), EsaAdCorpusReader),
    Source(
        "physionet2012",
        "physionet2012",
        WindowSpec(48.0 + 1.0 / 60.0, 48.0 + 1.0 / 60.0),
        Physionet2012CorpusReader,
    ),
    next(source for source in SOURCES if source.corpus == SCP1),
)


@pytest.fixture(scope="module")
def stored(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("regime-coverage")
    measure(out, SAMPLE_SOURCES, units=3, windows=2, sample=True)
    return out


def rows_of(directory: Path, name: str) -> list[dict[str, str]]:
    with (directory / name).open() as file:
        return list(csv.DictReader(file))


def test_the_samples_are_measured_unit_by_unit_and_window_by_window(stored: Path) -> None:
    units = rows_of(stored, "units.csv")
    windows = rows_of(stored, "windows.csv")

    assert tuple(units[0]) == UNIT_COLUMNS
    assert {row["corpus"] for row in units} == {source.corpus for source in SAMPLE_SOURCES}
    for corpus in {row["corpus"] for row in units}:
        assert 1 <= sum(row["corpus"] == corpus for row in units) <= 3
    for row in units:
        assert 0.0 < float(row["channel_share"]) <= 1.0
        assert 0.0 < float(row["co_observation"]) <= 1.0
    held = {(row["corpus"], row["unit"]) for row in units}
    for row in windows:
        assert (row["corpus"], row["unit"]) in held
        assert int(row["tokens"]) >= 1
        assert 0.0 <= float(row["once"]) <= 1.0


def test_each_corpus_is_summarised_with_its_regime(stored: Path) -> None:
    corpora = {row["corpus"]: row for row in rows_of(stored, "corpora.csv")}

    assert corpora["cmapss"]["regime"] == "regular"
    assert corpora["physionet2012"]["regime"] == "irregular"
    for row in corpora.values():
        assert float(row["unit_co_observation_p10"]) <= float(row["unit_co_observation_p90"])


def test_the_stays_are_read_at_their_own_instants_and_the_engine_on_a_grid(stored: Path) -> None:
    corpora = {row["corpus"]: row for row in rows_of(stored, "corpora.csv")}

    assert float(corpora["cmapss"]["unit_co_observation_p50"]) == 1.0
    assert float(corpora["physionet2012"]["unit_co_observation_p50"]) < 0.5


PUBLISHED = {
    "cmapss": Published(windows=100, training_share=0.8, observed=1_000),
    "physionet2012": Published(windows=50, training_share=0.5, observed=200),
}


def test_a_corpus_weighs_its_training_windows_times_its_passes_and_fraction() -> None:
    experiment = Experiment(
        "run", ("cmapss", "physionet2012"), {"physionet2012": 4}, {"cmapss": 0.5}
    )

    rows = {row.corpus: row for row in exposure_rows(experiment, PUBLISHED) if row.corpus != "*"}

    assert rows["cmapss"].windows == pytest.approx(40.0)
    assert rows["physionet2012"].windows == pytest.approx(100.0)
    assert rows["cmapss"].observed == pytest.approx(500.0)
    assert rows["physionet2012"].observed == pytest.approx(800.0)
    assert rows["physionet2012"].window_share == pytest.approx(100.0 / 140.0)


def test_the_regimes_share_the_whole_run() -> None:
    experiment = Experiment("run", ("cmapss", "physionet2012"), {}, {})

    totals = {row.regime: row for row in exposure_rows(experiment, PUBLISHED) if row.corpus == "*"}

    assert set(totals) == {regime.value for regime in SamplingRegime}
    assert sum(row.window_share for row in totals.values()) == pytest.approx(1.0)
    assert totals["irregular"].window_share == pytest.approx(25.0 / 105.0)


def test_the_table_gives_each_corpus_its_medians_and_marks_what_was_not_measured(
    stored: Path,
) -> None:
    summary = rows_of(stored, "corpora.csv")
    lonely = dict(summary[0]) | {f"unit_rate_spread_{q}": "nan" for q in ("p10", "p50", "p90")}

    lines = corpus_table([*summary, lonely])

    assert len(lines) == 2 + len(summary) + 1
    assert lines[2].startswith(f"| {summary[0]['corpus']} | {summary[0]['regime']} |")
    assert "—" in lines[-1]
