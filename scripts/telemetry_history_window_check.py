"""Check a window with history joined from the published hours against one cut whole.

A pretraining over windows of 48 hours can cut them whole from the readings, or join them from
the hourly windows the satellite tasks are published as. This check reads one month of a mission
with the publication's own reader and tokeniser and compares, for every 48-hour window of it:

- the readings of the 48 hourly windows concatenated against the window cut whole: channels,
  values, times, gaps and the cells of 48 bins;
- the cells of the hours, laid once and joined (``JoinedCells``), against the whole window's;
- how many windows a second the host lays from readings and how many it joins from cells.

It prints its numbers; nothing is written.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the package lives under the root's src.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.adapters.readers.esa_ad_raw import EsaAdRawCorpusReader  # noqa: E402
from emblema.catalog.adapters.tokenisation.sliding_window import (  # noqa: E402
    SlidingWindowTokeniser,
)
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary  # noqa: E402
from emblema.catalog.domain.tokenisation.tokenisation_scheme import (  # noqa: E402
    TokenisationScheme,
)
from emblema.catalog.domain.tokenisation.window_spec import WindowSpec  # noqa: E402
from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens  # noqa: E402
from emblema.shared.adapters.arrays.joined_cells import JoinedCells  # noqa: E402
from emblema.shared.kernel.tokens import Token, TokenWindow  # noqa: E402

CORPUS = "esa_ad_raw"
HOURS = 48
# The bins an hour is published under for the satellite tasks.
HOUR_BINS = 6
# The batches the host's throughput is read over.
BATCH, BATCHES = 32, 10


def concatenated(hours: Sequence[TokenWindow | None]) -> TokenWindow:
    """The hours as one window: each hour's times placed in its share, gaps counted anew.

    An hour with no reading is ``None``. A channel's first gap runs from the joined window's
    start, as a window cut whole counts it.
    """
    count = len(hours)
    rows = sorted(
        (t.time + at, t.channel_id, t.value)
        for at, hour in enumerate(hours)
        if hour is not None
        for t in hour
    )
    previous: dict[int, float] = {}
    tokens = []
    for place, channel, value in rows:
        at = place / count
        last = previous.get(channel)
        tokens.append(
            Token(channel_id=channel, value=value, time=at, gap=at if last is None else at - last)
        )
        previous[channel] = at
    return TokenWindow.of(tokens)


def hour_cells(hours: Sequence[TokenWindow | None], channels: Sequence[int]) -> NDArray[np.float64]:
    """``[channels, hours * HOUR_BINS, features]``: each hour laid once over ``channels``."""
    layout = BinnedTokens(HOUR_BINS, channels=channels)
    blank = np.zeros((len(channels), HOUR_BINS, BinnedTokens.N_FEATURES))
    laid = [
        blank
        if hour is None
        else layout.lay([hour])[0].features.reshape(HOUR_BINS, len(channels), -1).transpose(1, 0, 2)
        for hour in hours
    ]
    out: NDArray[np.float64] = np.concatenate(laid, axis=1)
    return out


def check(root: Path, mission: str) -> None:  # pragma: no cover - reads the corpus
    reader = EsaAdRawCorpusReader(root, subsets=(mission,))
    scheme = TokenisationScheme.for_vocabulary(ChannelVocabulary()).extended_with(
        CORPUS, reader.describe().channel_schema
    )
    units = list(reader.read_units())
    unit = units[len(units) // 2]
    tokeniser = SlidingWindowTokeniser()
    scheme = tokeniser.fit(CORPUS, reader.read_observations(unit.key), [], scheme)

    def cut(length: float) -> list[tuple[float, TokenWindow]]:
        spec = WindowSpec(length, length)
        placed = tokeniser.tokenise(CORPUS, unit, reader.read_observations(unit.key), scheme, spec)
        # A month starts off the hour by seconds at most, so hours are keyed to 10⁻⁴ of one.
        return [(round(p.extent.start, 4), p.window) for p in placed]

    whole, hourly = cut(float(HOURS)), dict(cut(1.0))
    channels = sorted({c for w in hourly.values() for c in w.channel_ids})
    layout = BinnedTokens(HOURS, channels=channels)
    joined_layout = JoinedCells(HOURS)
    equal = True
    worst_time = worst_gap = worst_cell = worst_joined = 0.0
    readings = 0
    for start, window in whole:
        hours = [hourly.get(round(start + k, 4)) for k in range(HOURS)]
        built = concatenated(hours)
        readings += len(window.channel_ids)
        equal &= built.channel_ids == window.channel_ids and built.values == window.values
        if built.channel_ids != window.channel_ids:
            continue
        worst_time = max(worst_time, float(np.abs(np.subtract(built.times, window.times)).max()))
        worst_gap = max(worst_gap, float(np.abs(np.subtract(built.gaps, window.gaps)).max()))
        cells = layout.lay([window])[0].features
        worst_cell = max(worst_cell, float(np.abs(layout.lay([built])[0].features - cells).max()))
        expected = cells.reshape(HOURS, len(channels), -1).transpose(1, 0, 2)
        derived = joined_layout.join(hour_cells(hours, channels))
        worst_joined = max(worst_joined, float(np.abs(derived - expected).max()))
    print(f"{unit.key}: {len(whole)} windows of {HOURS} h, {readings} readings")
    print(f"concatenated: channels and values equal {equal}")
    print(f"  worst time {worst_time:.1e}, gap {worst_gap:.1e}, cell {worst_cell:.1e}")
    print(f"joined cells: worst against the whole window's {worst_joined:.1e}")
    windows = [window for _, window in whole[:8]]
    began = time.perf_counter()
    layout.lay(windows)
    from_readings = len(windows) / (time.perf_counter() - began)
    starts = sorted(hourly)
    hours_laid = hour_cells([hourly[s] for s in starts], channels)
    anchors = range(HOURS, min(len(starts), HOURS + BATCH * BATCHES))
    began = time.perf_counter()
    for anchor in anchors:
        span = hours_laid[:, (anchor - HOURS) * HOUR_BINS : anchor * HOUR_BINS]
        JoinedCells.laid(joined_layout.join(span), np.asarray(channels), own_channels=True)
    from_cells = len(anchors) / (time.perf_counter() - began)
    print(f"windows a second: {from_readings:.0f} laid from readings, {from_cells:.0f} joined")
    print(f"the month's hours as cells: {hours_laid.nbytes / 1e6:.1f} MB")


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - reads the corpus
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data/raw/esa_ad")
    parser.add_argument("--mission", choices=EsaAdRawCorpusReader.SUBSETS, required=True)
    arguments = parser.parse_args(argv)
    check(arguments.root, arguments.mission)


if __name__ == "__main__":  # pragma: no cover
    main()
