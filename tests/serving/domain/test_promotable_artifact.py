import pytest

from emblema.serving.domain.exceptions import InvalidPromotableArtifactError
from tests.serving.support import promotable, score


def test_an_artifact_scored_twice_at_one_operating_point_is_refused() -> None:
    with pytest.raises(InvalidPromotableArtifactError, match="scored twice"):
        promotable(scores=(score(), score(value=1.0)))


def test_one_budget_may_be_scored_by_several_metrics() -> None:
    both = (score(), score(metric="mae", value=12.0))

    assert promotable(scores=both).scores == both
