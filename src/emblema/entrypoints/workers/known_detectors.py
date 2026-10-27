from typing import ClassVar

from emblema.evaluation.adapters.candidates.detector_arm import DetectorArm
from emblema.evaluation.adapters.candidates.detector_catalogue import DetectorCatalogue
from emblema.evaluation.contracts.identifiers import CandidateRef
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.transfer.encoder_setting import EncoderSetting
from emblema.shared.kernel.artifacts import ArtifactRef


class KnownDetectors:
    """The detectors every detection campaign competes, each under the name it is compared by.

    One detector over the pretrained encoder's states, one over the same encoder at its
    initialisation, reading the windows as the backbone read them, and one over the summaries
    per channel: the same distance to the same number of nearest fitting windows under the same
    alarm, so a campaign of the three reads what the representation adds and nothing else. Both
    encoders are read at every block side by side, the reading the probes found the states
    worth most under, and pooled by the mean over the window's tokens. The neighbours and the
    alarm's share are knobs of the name, declared with the campaign. A register rather than a
    table read from configuration, for the reason the arms are one: the names are what a stored
    campaign refers to.
    """

    READING: ClassVar[EncoderSetting] = EncoderSetting.standard().tuned("layer", "concat")

    FROZEN = CandidateRef("frozen_knn")
    UNTRAINED = CandidateRef("untrained_knn")
    SUMMARIES = CandidateRef("summaries_knn")

    @classmethod
    def over(cls, backbone: ArtifactRef) -> tuple[DetectorArm, ...]:
        """Every detector, the two over an encoder shaped like ``backbone``, in reporting order."""
        return (
            DetectorArm(
                ref=cls.FROZEN,
                features=DetectorFeatures.ENCODER_STATES,
                architecture=backbone,
                backbone=backbone,
                encoder=cls.READING,
            ),
            DetectorArm(
                ref=cls.UNTRAINED,
                features=DetectorFeatures.ENCODER_STATES,
                architecture=backbone,
                backbone=None,
                encoder=cls.READING,
            ),
            DetectorArm(
                ref=cls.SUMMARIES,
                features=DetectorFeatures.SUMMARIES,
                architecture=None,
                backbone=None,
            ),
        )

    @classmethod
    def refs(cls) -> tuple[CandidateRef, ...]:
        """What the detectors are called, in reporting order."""
        return (cls.FROZEN, cls.UNTRAINED, cls.SUMMARIES)

    @classmethod
    def catalogue(cls, backbone: ArtifactRef) -> DetectorCatalogue:
        """What the detectors are, with nothing that could fit one."""
        return DetectorCatalogue(cls.over(backbone))
