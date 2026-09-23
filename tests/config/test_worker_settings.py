"""What the campaign worker is told, read the way a started process reads it.

The group is exercised through the environment and not only through its constructor: a settings
source decodes a complex field as JSON before any validator of ours runs, so a field that reads
fine when built in Python can be unreadable when exported in a shell. Each case starts from an
environment with nothing of ours in it, so what this machine's file or another test happens to
have exported decides nothing.
"""

import pytest
from pydantic import ValidationError

from emblema.config.artifact_store_settings import ArtifactStoreSettings
from emblema.config.lora_settings import LoraSettings
from emblema.config.schedule_settings import ScheduleSettings
from emblema.config.settings import Settings
from emblema.config.worker_settings import WorkerSettings
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from tests.support.settings import only

SCHEDULE = ScheduleSettings(
    epochs=2,
    min_steps=0,
    batch_size=2,
    learning_rate=1e-2,
    weight_decay=0.0,
    warmup_fraction=0.0,
    final_lr_fraction=1.0,
)
LORA = LoraSettings(rank=2, alpha=4.0, dropout=0.0, targets="qkv")


def worker(backbone: str) -> WorkerSettings:
    return WorkerSettings(
        workspace="data/workspace",
        corpora="data/raw",
        backbone=backbone,
        schedule=SCHEDULE,
        lora=LORA,
    )


def test_a_backbone_is_read_from_the_form_the_registry_holds_it_in() -> None:
    checksum = Checksum.of_bytes(b"weights")

    assert worker(f"durable/weights@sha256:{checksum.digest}").backbone_ref() == ArtifactRef(
        key="durable/weights", checksum=checksum
    )


@pytest.mark.parametrize("written", ["durable/weights", "durable/weights@sha256", "@sha256:abc"])
def test_a_backbone_not_written_as_the_registry_holds_it_is_refused(written: str) -> None:
    with pytest.raises(ValueError, match="artifact reference"):
        worker(written).backbone_ref()


ENVIRONMENT = {
    "EMBLEMA_WORKER__WORKSPACE": "data/workspace",
    "EMBLEMA_WORKER__CORPORA": "data/raw",
    "EMBLEMA_WORKER__BACKBONE": "durable/sha256/abc@sha256:abc",
    "EMBLEMA_WORKER__SCHEDULE__EPOCHS": "30",
    "EMBLEMA_WORKER__SCHEDULE__MIN_STEPS": "120",
    "EMBLEMA_WORKER__SCHEDULE__BATCH_SIZE": "32",
    "EMBLEMA_WORKER__SCHEDULE__LEARNING_RATE": "1e-3",
    "EMBLEMA_WORKER__SCHEDULE__WEIGHT_DECAY": "0.01",
    "EMBLEMA_WORKER__SCHEDULE__WARMUP_FRACTION": "0.1",
    "EMBLEMA_WORKER__SCHEDULE__FINAL_LR_FRACTION": "0.1",
    "EMBLEMA_WORKER__LORA__RANK": "8",
    "EMBLEMA_WORKER__LORA__ALPHA": "16.0",
    "EMBLEMA_WORKER__LORA__DROPOUT": "0.0",
    "EMBLEMA_WORKER__LORA__TARGETS": "qkv,attention.projection,feedforward",
}


def exported(monkeypatch: pytest.MonkeyPatch, **changes: str | None) -> Settings:
    """Settings of a process started with that environment and nothing read from a file."""
    only(monkeypatch, **{k: v for k, v in (ENVIRONMENT | changes).items() if v is not None})
    return Settings(
        _env_file=None,
        artifact_store=ArtifactStoreSettings(
            endpoint_url="http://127.0.0.1:3900", region="garage", bucket="e", key_prefix="t"
        ),
    )


def test_the_group_is_read_from_the_environment_a_worker_is_started_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = exported(monkeypatch).require_worker()

    assert worker.schedule.epochs == 30
    assert worker.lora.targets == "qkv,attention.projection,feedforward"


def test_a_process_that_is_told_nothing_about_a_worker_still_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert exported(monkeypatch, **dict.fromkeys(ENVIRONMENT)).worker is None


def test_a_worker_told_half_of_what_it_needs_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Half a group is worse than none: the process would run a grid under a schedule nobody
    # declared, so it fails where it is started.
    with pytest.raises(ValidationError, match="schedule"):
        exported(monkeypatch, EMBLEMA_WORKER__SCHEDULE__EPOCHS=None)
