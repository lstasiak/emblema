from dataclasses import dataclass
from pathlib import Path

from emblema.pretraining.application.use_cases.accept_pretraining_result import (
    AcceptPretrainingResultCommand,
)
from emblema.pretraining.application.use_cases.fulfil_pretraining_order import (
    FulfilPretrainingOrderCommand,
)
from emblema.pretraining.application.use_cases.order_pretraining import OrderPretrainingCommand

type PretrainCommand = (
    OrderPretrainingCommand | FulfilPretrainingOrderCommand | AcceptPretrainingResultCommand
)


@dataclass(frozen=True, kw_only=True)
class PretrainInvocation:
    """One run of the pretraining command line: the command and what the process runs it with.

    Attributes:
        command: What to do: order a run, fulfil an order here, or accept a result.
        workspace: Directory blocks are fetched to and mapped from.
        device: Where a run fulfilled here computes; the machine's accelerator unless given.
        tracking_uri: MLflow server or database file the run is recorded against; in memory
            unless given.
        num_workers: Processes collating batches for a run fulfilled here.
        progress_every: Optimiser steps between two progress lines of a run fulfilled here; zero
            for none.
    """

    command: PretrainCommand
    workspace: Path
    device: str | None = None
    tracking_uri: str | None = None
    num_workers: int = 0
    progress_every: int = 0
