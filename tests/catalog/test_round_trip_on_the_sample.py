"""The round trip the sanity check rests on, run on the miniature C-MAPSS sample in the repository.

The port contract holds every adapter to the definition of a window on data built for the test.
This holds the real reader and the real tokeniser to it on real bytes, which is where a column read
into the wrong channel, a time axis off by a cycle or a value that never reached its token would
show. Reconstruction is exact in arithmetic, not in bits, so values are compared to a resolution.
"""

from collections.abc import Sequence
from itertools import chain

import pytest

from emblema.catalog.adapters.readers.cmapss import CmapssCorpusReader
from emblema.catalog.adapters.tokenisation.sliding_window import SlidingWindowTokeniser
from emblema.catalog.domain.channels.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.tokenisation.tokenisation_scheme import TokenisationScheme
from tests.catalog.domain.support import measured
from tests.support.corpora import CORPUS, SAMPLE, SAMPLE_WINDOW, SUBSET


@pytest.fixture(scope="module")
def reader() -> CmapssCorpusReader:
    return CmapssCorpusReader(SAMPLE, (SUBSET,))


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
        placed = list(tokeniser.tokenise(CORPUS, unit, observations, scheme, SAMPLE_WINDOW))

        assert len(placed) > 1
        for item in placed:
            inside = [o for o in observations if item.extent.contains(o.time)]
            read_back = scheme.reconstruct(item.window, item.extent).observations
            assert measured(read_back) == measured(inside)


def test_the_windows_of_an_engine_hold_every_row_but_the_tail_beyond_the_last_of_them(
    reader: CmapssCorpusReader, units: Sequence[CorpusUnit], scheme: TokenisationScheme
) -> None:
    tokeniser = SlidingWindowTokeniser()

    for unit in units:
        observations = list(reader.read_observations(unit.key))
        placed = list(tokeniser.tokenise(CORPUS, unit, observations, scheme, SAMPLE_WINDOW))

        recovered = {
            row
            for item in placed
            for row in measured(scheme.reconstruct(item.window, item.extent).observations)
        }
        last_cycle_covered = placed[-1].extent.end
        assert recovered == set(measured(o for o in observations if o.time < last_cycle_covered))
        assert any(o.time >= last_cycle_covered for o in observations), "no tail to speak of"
