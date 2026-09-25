from pathlib import Path

from pydantic import BaseModel, Field

from emblema.config.boosting_settings import BoostingSettings
from emblema.config.convolution_settings import ConvolutionSettings
from emblema.config.lora_settings import LoraSettings
from emblema.config.patch_settings import PatchSettings
from emblema.config.schedule_settings import ScheduleSettings
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class WorkerSettings(BaseModel):
    """What a campaign worker is told before it starts, read as ``EMBLEMA_WORKER__<FIELD>``.

    A worker serves one backbone and runs every cell under one schedule, which is why both are
    told to the process rather than carried by each job. Only the device has a default, because
    hardware is the one thing an experiment does not declare.

    What a worker needs depends on what it competes. The one that trains networks has no use for
    the classical knobs and the one that fits baselines has no use for a backbone, a schedule,
    low-rank updates or a patch model's shape, so each of them is optional here and demanded by
    the process that cannot run without it. Demanded as it is assembled, so a worker told to
    serve something it was not configured for stops at startup rather than at the first cell it
    is handed.
    """

    workspace: Path = Field(description="Directory corpus blocks are fetched to and mapped from.")
    corpora: Path = Field(description="Directory the raw corpora sit in, where labels are read.")
    backbone: str | None = Field(
        default=None, description="Written key@algorithm:digest, as the registry holds it."
    )
    schedule: ScheduleSettings | None = None
    lora: LoraSettings | None = None
    boosting: BoostingSettings | None = None
    convolutions: ConvolutionSettings | None = None
    patch: PatchSettings | None = None
    device: str | None = Field(
        default=None, description="Where a cell computes; the machine's accelerator unless given."
    )

    def require_schedule(self) -> ScheduleSettings:
        """How every cell of this process's campaigns learns.

        Raises:
            ValueError: If nothing is configured.
        """
        if self.schedule is None:
            raise ValueError("this worker adapts a backbone and was given no schedule")
        return self.schedule

    def require_lora(self) -> LoraSettings:
        """The low-rank updates the arm of that name adds.

        Raises:
            ValueError: If nothing is configured.
        """
        if self.lora is None:
            raise ValueError("this worker adapts a backbone and was given no low-rank updates")
        return self.lora

    def require_boosting(self) -> BoostingSettings:
        """How hard the classical candidates of this process's campaigns fit.

        Raises:
            ValueError: If nothing is configured.
        """
        if self.boosting is None:
            raise ValueError("this worker fits classical candidates and was given no boosting")
        return self.boosting

    def require_convolutions(self) -> ConvolutionSettings:
        """How the MiniRocket candidate of this process's campaigns reads and fits.

        Raises:
            ValueError: If nothing is configured.
        """
        if self.convolutions is None:
            raise ValueError("this worker fits classical candidates and was given no convolutions")
        return self.convolutions

    def require_patch(self) -> PatchSettings:
        """How the patch model of this process's campaigns reads a window and how large it is.

        Raises:
            ValueError: If nothing is configured.
        """
        if self.patch is None:
            raise ValueError("this worker trains the patch model and was given no shape for it")
        return self.patch

    def require_backbone_ref(self) -> ArtifactRef:
        """The backbone as the registry holds it: a key and the checksum of its bytes.

        Raises:
            ValueError: If nothing is configured, or the text is not a reference of the form
                ``key@algorithm:digest``.
        """
        if self.backbone is None:
            raise ValueError("this worker adapts a backbone and was given none to serve")
        key, _, checksum = self.backbone.partition("@")
        algorithm, _, digest = checksum.partition(":")
        if not key or not algorithm or not digest:
            raise ValueError(
                f"not an artifact reference of the form key@algorithm:digest: {self.backbone!r}"
            )
        return ArtifactRef(key, Checksum(HashAlgorithm(algorithm), digest))
