from collections.abc import Mapping
from dataclasses import dataclass

from emblema.pretraining.domain.assessment.curve import Curve
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally
from emblema.pretraining.domain.assessment.spectrum import Spectrum
from emblema.pretraining.domain.masking_strategy import MaskingStrategy

# Settings that name a run — when it happened, what its tracker calls it, which weights it left —
# rather than say what it was; two runs differing only in these are runs of one configuration.
CIRCUMSTANTIAL = frozenset({"date", "tracked_run", "backbone_checksum"})


@dataclass(frozen=True)
class Results:
    """Everything a run measured, in the form it is stored and assessed in.

    Attributes:
        settings: What the run states about itself — corpus, code, model, schedule, machine — as
            text, the epochs aside, which the curve states.
        strategy: The masking strategy the run trained and was diagnosed under.
        realised_ratio: Share of observed validation tokens the masks hid.
        curve: Losses per epoch.
        tallies: Sums per kind of mask and validation unit.
        spectrum: Spectral recovery of the channels hidden whole.
    """

    settings: Mapping[str, str]
    strategy: MaskingStrategy
    realised_ratio: float
    curve: Curve
    tallies: tuple[MaskKindTally, ...]
    spectrum: Spectrum

    @property
    def epochs(self) -> int:
        return len(self.curve.validation)

    def configuration_of(self, other: "Results") -> bool:
        """Whether ``other`` is a run of this configuration, whatever its epochs and its date."""

        def essential(settings: Mapping[str, str]) -> dict[str, str]:
            return {key: value for key, value in settings.items() if key not in CIRCUMSTANTIAL}

        return (
            essential(self.settings) == essential(other.settings)
            and self.strategy == other.strategy
        )
