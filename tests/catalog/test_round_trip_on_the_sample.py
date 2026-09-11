"""The round trip the sanity check rests on, run on the miniature C-MAPSS sample in the repository.

The port contract holds every adapter to the definition of a window on data built for the test.
This holds the real reader and the real tokeniser to it on real bytes, which is where a column read
into the wrong channel, a time axis off by a cycle or a value that never reached its token would
show. Reconstruction is exact in arithmetic, not in bits, so values are compared to a resolution.
"""

from collections.abc import Sequence
from itertools import chain
from pathlib import Path

import pytest

from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.domain.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.corpus_unit import CorpusUnit
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from emblema.catalog.domain.window_spec import WindowSpec
from tests.catalog.domain.support import measured, occupied_extents

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "cmapss"
CORPUS = "cmapss"
# Not the window a run cuts: short enough that the two sample engines yield several windows each,
# with a stride that leaves a tail beyond the last of them.
WINDOW = WindowSpec(length=20.0, stride=7.0)


@pytest.fixture(scope="module")
def reader() -> CmapssCorpusReader:
    return CmapssCorpusReader(SAMPLE, ("FD001",))


@pytest.fixture(scope="module")
def units(reader: CmapssCorpusReader) -> list[CorpusUnit]:
    return list(reader.read_units())


@pytest.fixture(scope="module")
def scheme(reader: CmapssCorpusReader, units: Sequence[CorpusUnit]) -> TokenisationScheme:
    """Fitted on every engine of the sample: this is a check on the data, not a training run."""
    unfitted = TokenisationScheme.for_vocabulary(ChannelVocabulary()).extended_with(
        CORPUS, reader.describe().channel_schema
    )
    return SlidingWindowTokeniser().fit(
        CORPUS,
        chain.from_iterable(reader.read_observations(unit.key) for unit in units),
        (),
        unfitted,
    )


def test_every_window_of_the_sample_reads_back_as_the_rows_that_fell_into_it(
    reader: CmapssCorpusReader, units: Sequence[CorpusUnit], scheme: TokenisationScheme
) -> None:
    tokeniser = SlidingWindowTokeniser()

    for unit in units:
        observations = list(reader.read_observations(unit.key))
        windows = list(tokeniser.tokenise(CORPUS, unit, observations, scheme, WINDOW))
        extents = occupied_extents(unit, observations, WINDOW)

        assert len(windows) > 1
        for extent, window in zip(extents, windows, strict=True):
            inside = [o for o in observations if extent.contains(o.time)]
            assert measured(scheme.reconstruct(window, extent).observations) == measured(inside)


def test_the_windows_of_an_engine_hold_every_row_but_the_tail_beyond_the_last_of_them(
    reader: CmapssCorpusReader, units: Sequence[CorpusUnit], scheme: TokenisationScheme
) -> None:
    tokeniser = SlidingWindowTokeniser()

    for unit in units:
        observations = list(reader.read_observations(unit.key))
        windows = tokeniser.tokenise(CORPUS, unit, observations, scheme, WINDOW)
        extents = occupied_extents(unit, observations, WINDOW)

        recovered = {
            row
            for extent, window in zip(extents, windows, strict=True)
            for row in measured(scheme.reconstruct(window, extent).observations)
        }
        last_cycle_covered = extents[-1].end
        assert recovered == set(measured(o for o in observations if o.time < last_cycle_covered))
        assert any(o.time >= last_cycle_covered for o in observations), "no tail to speak of"
