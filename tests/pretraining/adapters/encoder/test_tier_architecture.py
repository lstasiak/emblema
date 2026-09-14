import pytest

from emblema.config.compute_tiers import ComputeTiers
from emblema.pretraining.adapters.encoder.tier_architecture import architecture_of
from emblema.shared.kernel.compute import ComputeTier


@pytest.mark.parametrize("name", list(ComputeTier))
def test_the_architecture_is_the_shape_the_tier_states(name: ComputeTier) -> None:
    tier = ComputeTiers.load().profile(name)

    architecture = architecture_of(tier)

    assert (
        architecture.width,
        architecture.heads,
        architecture.layers,
        architecture.feedforward_width,
        architecture.time_frequencies,
    ) == (tier.width, tier.heads, tier.layers, tier.feedforward_width, tier.time_frequencies)
