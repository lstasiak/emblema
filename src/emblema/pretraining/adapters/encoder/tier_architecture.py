from emblema.config.compute_tiers import ComputeTierProfile
from emblema.pretraining.domain.encoder_architecture import EncoderArchitecture


def architecture_of(tier: ComputeTierProfile) -> EncoderArchitecture:
    """The encoder shape a compute tier states."""
    return EncoderArchitecture(
        width=tier.width,
        heads=tier.heads,
        layers=tier.layers,
        feedforward_width=tier.feedforward_width,
        time_frequencies=tier.time_frequencies,
    )
