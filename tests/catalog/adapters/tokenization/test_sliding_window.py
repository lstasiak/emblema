"""What is specific to the streaming adapter: its arithmetic and its throughput on real data."""

import statistics
import time
import tomllib
from itertools import chain
from pathlib import Path

import pytest

from emblema.catalog.adapters.readers.cmapss import SENSORS, CmapssCorpusReader
from emblema.catalog.adapters.tokenization.sliding_window import SlidingWindowTokenizer
from emblema.catalog.domain.observation import Observation
from emblema.catalog.domain.window_spec import WindowSpec
from tests.catalog.adapters.readers.test_cmapss import BUDGET, SAMPLE, raw_root
from tests.catalog.domain.support import CORPUS, scheme_for

TOKENIZER = SlidingWindowTokenizer()


def test_the_running_moments_survive_a_large_offset() -> None:
    values = [1e9 + delta for delta in (0.0, 0.1, 0.2, 0.3, 0.4)]

    scheme = TOKENIZER.fit(
        CORPUS,
        [Observation("pressure", float(t), v) for t, v in enumerate(values)],
        (),
        scheme_for(),
    )

    fitted = scheme.statistics_of(scheme.vocabulary.id_of(CORPUS, "pressure"))
    assert fitted.mean == pytest.approx(statistics.fmean(values))
    assert fitted.std == pytest.approx(statistics.pstdev(values), rel=1e-6)


def test_a_constant_channel_has_exactly_zero_spread() -> None:
    scheme = TOKENIZER.fit(
        CORPUS, [Observation("pressure", float(t), 518.67) for t in range(100)], (), scheme_for()
    )

    assert scheme.statistics_of(scheme.vocabulary.id_of(CORPUS, "pressure")).std == 0.0


def tokenise_corpus(root: Path, subsets: tuple[str, ...], window: WindowSpec) -> tuple[int, int]:
    """Window and token counts of a C-MAPSS corpus, through the reader and the tokenizer."""
    reader = CmapssCorpusReader(root, subsets)
    units = list(reader.read_units())
    scheme = scheme_for(reader.describe().channel_schema, "cmapss")
    scheme = TOKENIZER.fit(
        "cmapss", chain.from_iterable(reader.read_observations(u.key) for u in units), (), scheme
    )
    windows = tokens = 0
    for unit in units:
        for produced in TOKENIZER.tokenise(
            "cmapss", unit, reader.read_observations(unit.key), scheme, window
        ):
            windows += 1
            tokens += len(produced)
    return windows, tokens


def test_the_sample_engines_give_the_windows_the_data_spike_counts() -> None:
    # Engines of 128 and 135 cycles: floor((128 - 50) / 5) + 1 + floor((135 - 50) / 5) + 1.
    assert tokenise_corpus(SAMPLE, ("FD001",), WindowSpec(50, 5)) == (34, 34 * 50 * 21)


def default_window() -> WindowSpec:
    """The window variant the budget file marks as default, with the counts measured for it."""
    with BUDGET.open("rb") as handle:
        corpus = tomllib.load(handle)["corpora"]["cmapss"]
    spec = next(window for window in corpus["windows"] if window.get("default"))
    return WindowSpec(spec["length"], spec["stride"])


def measured_default_window() -> dict[str, int]:
    with BUDGET.open("rb") as handle:
        corpus = tomllib.load(handle)["corpora"]["cmapss"]
    name = next(window["name"] for window in corpus["windows"] if window.get("default"))
    return next(window for window in corpus["measured"]["windows"] if window["name"] == name)


def windows_over_every_engine(reader: CmapssCorpusReader, window: WindowSpec) -> int:
    return sum(1 for unit in reader.read_units() for _ in window.windows_over(unit.extent))


NO_RAW_DATA = pytest.mark.skipif(
    raw_root() is None,
    reason="the raw C-MAPSS files are not on this machine (scripts/fetch_corpora.py cmapss)",
)


@NO_RAW_DATA
def test_the_full_corpus_lays_out_the_windows_measured_by_the_data_spike() -> None:
    root = raw_root()
    assert root is not None
    window, measured = default_window(), measured_default_window()

    count = windows_over_every_engine(CmapssCorpusReader(root), window)

    # Every cycle of a regular corpus carries all its sensors, so tokens follow from windows.
    assert count == measured["count"]
    assert count * window.length * len(SENSORS) == measured["tokens"]


@NO_RAW_DATA
def test_a_real_subset_tokenises_to_the_windows_its_engines_hold() -> None:
    root = raw_root()
    assert root is not None
    window = default_window()
    expected = windows_over_every_engine(CmapssCorpusReader(root, ("FD001",)), window)

    started = time.perf_counter()
    windows, tokens = tokenise_corpus(root, ("FD001",), window)
    elapsed = time.perf_counter() - started

    assert (windows, tokens) == (expected, expected * window.length * len(SENSORS))
    # Pure-Python tokenisation of one subset stays a small fraction of the suite's budget; the
    # whole corpus took 49 s on the development laptop when this was written.
    assert elapsed < 60, f"tokenising FD001 took {elapsed:.0f} s"
