import pytest

from emblema.catalog.domain.channel_statistics import ChannelStatistics
from emblema.catalog.domain.channel_vocabulary import ChannelVocabulary
from emblema.catalog.domain.exceptions import (
    ChannelAlreadyFittedError,
    InvalidTokenisationSchemeError,
    MissingChannelStatisticsError,
    UnknownChannelError,
)
from emblema.catalog.domain.tokenisation_scheme import TokenisationScheme
from tests.catalog.domain.support import CORPUS, OTHER_SCHEMA, SCHEMA, scheme_for

STATISTICS = ChannelStatistics(3, 1.0, 2.0)


def test_a_scheme_for_a_vocabulary_has_nothing_fitted() -> None:
    scheme = TokenisationScheme.for_vocabulary(ChannelVocabulary().extended_with(CORPUS, SCHEMA))

    assert scheme.statistics == (None, None)
    with pytest.raises(MissingChannelStatisticsError, match="not observed"):
        scheme.statistics_of(1)


def test_fitting_a_channel_records_its_statistics_and_leaves_the_others() -> None:
    scheme = scheme_for().with_statistics(2, STATISTICS)

    assert scheme.statistics_of(2) == STATISTICS
    assert scheme.statistics == (None, STATISTICS)


def test_a_channel_is_fitted_once() -> None:
    scheme = scheme_for().with_statistics(1, STATISTICS)

    with pytest.raises(ChannelAlreadyFittedError, match="fitted"):
        scheme.with_statistics(1, STATISTICS)


@pytest.mark.parametrize("channel_id", [0, 3])
def test_an_unknown_channel_cannot_be_fitted_or_looked_up(channel_id: int) -> None:
    scheme = scheme_for()

    with pytest.raises(UnknownChannelError):
        scheme.with_statistics(channel_id, STATISTICS)
    with pytest.raises(UnknownChannelError):
        scheme.statistics_of(channel_id)


def test_extending_the_vocabulary_keeps_fitted_channels_and_adds_unfitted_ones() -> None:
    scheme = scheme_for().with_statistics(1, STATISTICS)

    extended = scheme.extended_with("other", OTHER_SCHEMA)

    assert len(extended.vocabulary) == 3
    assert extended.statistics == (STATISTICS, None, None)


def test_statistics_must_align_with_the_vocabulary() -> None:
    vocabulary = ChannelVocabulary().extended_with(CORPUS, SCHEMA)

    with pytest.raises(InvalidTokenisationSchemeError, match="align"):
        TokenisationScheme(vocabulary, (None,))
