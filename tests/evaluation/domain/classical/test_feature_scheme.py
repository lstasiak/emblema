from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme


def test_only_the_scheme_summarised_across_channels_spans_layouts() -> None:
    spanning = {scheme for scheme in FeatureScheme if scheme.spans_channel_layouts}

    assert spanning == {FeatureScheme.CHANNEL_AGGREGATED}
