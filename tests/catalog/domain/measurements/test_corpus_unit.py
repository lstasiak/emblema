import pytest

from emblema.catalog.domain.exceptions import InvalidCorpusUnitError
from emblema.catalog.domain.identifiers import UnitKey
from emblema.catalog.domain.measurements.corpus_unit import CorpusUnit
from emblema.catalog.domain.measurements.static_feature import StaticFeature
from emblema.catalog.domain.measurements.time_extent import TimeExtent


def test_unit_static_features_are_one_per_channel() -> None:
    with pytest.raises(InvalidCorpusUnitError, match="duplicate"):
        CorpusUnit(
            UnitKey("u"),
            TimeExtent(0.0, 1.0),
            (StaticFeature("age", 1.0), StaticFeature("age", 2.0)),
        )


def test_a_unit_has_no_static_features_unless_given() -> None:
    assert CorpusUnit(UnitKey("u"), TimeExtent(0.0, 1.0)).static_features == ()
