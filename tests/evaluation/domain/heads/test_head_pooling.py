import math

import pytest

from emblema.evaluation.domain.exceptions import InvalidHeadPoolingError, UnknownKnobError
from emblema.evaluation.domain.heads.head_pooling import (
    HeadPooling,
    PoolingScheme,
    StaticsPlacement,
)


def test_every_network_starts_from_the_mean_over_the_whole_window() -> None:
    stated = HeadPooling.mean()

    assert stated == HeadPooling(pooling=PoolingScheme.MEAN, tail_share=1.0)
    assert stated.parameters() == {"pooling": "mean", "tail_share": 1.0}


def test_a_tail_keeps_a_share_of_the_window_and_may_keep_all_of_it() -> None:
    tail = HeadPooling(pooling=PoolingScheme.TAIL, tail_share=0.2)

    assert tail.parameters() == {"pooling": "tail", "tail_share": 0.2}
    # The state a variant passes through before its share is set; it computes the mean.
    assert HeadPooling(pooling=PoolingScheme.TAIL).tail_share == 1.0


@pytest.mark.parametrize("scheme", [PoolingScheme.MEAN, PoolingScheme.ATTENTION])
def test_a_share_under_a_scheme_that_reads_the_whole_window_is_refused(
    scheme: PoolingScheme,
) -> None:
    with pytest.raises(InvalidHeadPoolingError, match="no knob"):
        HeadPooling(pooling=scheme, tail_share=0.5)


@pytest.mark.parametrize("share", [0.0, -0.1, 1.5, math.nan, math.inf])
def test_a_share_outside_the_window_is_refused(share: float) -> None:
    with pytest.raises(InvalidHeadPoolingError, match=r"lie in \(0, 1\]"):
        HeadPooling(pooling=PoolingScheme.TAIL, tail_share=share)


def test_only_attention_has_weights_of_its_own() -> None:
    assert [scheme.learns_weights for scheme in PoolingScheme] == [False, False, True]


def test_the_knobs_turn_the_scheme_first_and_then_its_share() -> None:
    mean = HeadPooling.mean()

    tail = mean.tuned("pooling", "tail").tuned("tail_share", "0.2")

    assert tail == HeadPooling(pooling=PoolingScheme.TAIL, tail_share=0.2)
    with pytest.raises(UnknownKnobError, match="no knob"):
        mean.tuned("tail_share", "0.2")
    assert mean.tuned("pooling", "attention") == HeadPooling(pooling=PoolingScheme.ATTENTION)


def test_a_knob_the_pooling_lacks_or_cannot_take_is_refused() -> None:
    with pytest.raises(UnknownKnobError, match="no knob 'share'"):
        HeadPooling.mean().tuned("share", "0.2")
    with pytest.raises(UnknownKnobError, match="takes a PoolingScheme"):
        HeadPooling.mean().tuned("pooling", "median")
    with pytest.raises(UnknownKnobError, match="takes a float"):
        HeadPooling.mean().tuned("tail_share", "a fifth")
    with pytest.raises(UnknownKnobError, match="takes a StaticsPlacement"):
        HeadPooling.mean().tuned("statics", "beside")


def test_static_features_stand_among_the_readings_unless_a_variant_sets_them_apart() -> None:
    mean = HeadPooling.mean()

    apart = mean.tuned("statics", "apart")

    assert (mean.statics, mean.width_factor, mean.turned_away()) == (StaticsPlacement.AMONG, 1, {})
    assert apart == HeadPooling(pooling=PoolingScheme.MEAN, statics=StaticsPlacement.APART)
    assert (apart.width_factor, apart.turned_away()) == (2, {"statics": "apart"})
    # Named apart from the scheme and its share, so descriptions stored before it are unchanged.
    assert apart.parameters() == mean.parameters()


@pytest.mark.parametrize("scheme", ["tail", "attention"])
def test_static_features_are_set_apart_under_every_scheme(scheme: str) -> None:
    apart = HeadPooling.mean().tuned("pooling", scheme).tuned("statics", "apart")

    assert (apart.pooling, apart.statics) == (PoolingScheme(scheme), StaticsPlacement.APART)
