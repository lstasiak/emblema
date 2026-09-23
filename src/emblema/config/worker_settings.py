from pathlib import Path

from pydantic import BaseModel, Field

from emblema.config.lora_settings import LoraSettings
from emblema.config.schedule_settings import ScheduleSettings
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum, HashAlgorithm


class WorkerSettings(BaseModel):
    """What the campaign worker is told before it starts, read as ``EMBLEMA_WORKER__<FIELD>``.

    A worker serves one backbone and runs every cell under one schedule, which is why both are
    told to the process rather than carried by each job. Only the device has a default, because
    hardware is the one thing an experiment does not declare.
    """

    workspace: Path = Field(description="Directory corpus blocks are fetched to and mapped from.")
    corpora: Path = Field(description="Directory the raw corpora sit in, where labels are read.")
    backbone: str = Field(description="Written key@algorithm:digest, as the registry holds it.")
    schedule: ScheduleSettings
    lora: LoraSettings
    device: str | None = Field(
        default=None, description="Where a cell computes; the machine's accelerator unless given."
    )

    def backbone_ref(self) -> ArtifactRef:
        """The backbone as the registry holds it: a key and the checksum of its bytes.

        Raises:
            ValueError: If the text is not a reference of the form ``key@algorithm:digest``.
        """
        key, _, checksum = self.backbone.partition("@")
        algorithm, _, digest = checksum.partition(":")
        if not key or not algorithm or not digest:
            raise ValueError(
                f"not an artifact reference of the form key@algorithm:digest: {self.backbone!r}"
            )
        return ArtifactRef(key, Checksum(HashAlgorithm(algorithm), digest))
