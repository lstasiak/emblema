"""Backbones, orders and results at a test's size, over the experiment and corpus of a test.

Every builder states a whole value and replaces what is named, like the experiment builders;
the overrides are ``Any`` for the same reason. The signature is computed, never invented, so an
order and a result built here agree with each other and with the backbone unless a test says
otherwise.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from emblema.catalog.contracts.identifiers import CorpusVersionId
from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.backbone.pretraining_input import PretrainingInput
from emblema.pretraining.domain.handoff.pretraining_order import PretrainingOrder
from emblema.pretraining.domain.handoff.pretraining_result import PretrainingResult
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.domain.training.run_signature import RunSignature
from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.timestamps import UtcDateTime
from tests.support.experiments import WEIGHTS, configuration, corpus, epoch_outcome

CORPUS = corpus()
CONFIGURATION = configuration()
SIGNATURE = RunSignature.of(CONFIGURATION, CORPUS)
MANIFEST = ArtifactRef("durable/manifest", Checksum.of_bytes(b"manifest"))
ORDER_REF = ArtifactRef("durable/order", Checksum.of_bytes(b"order"))
CHECKPOINT = ArtifactRef("transient/checkpoint", Checksum.of_bytes(b"checkpoint"))
RESULT_REF = ArtifactRef("durable/result", Checksum.of_bytes(b"result"))
COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
ORDERED_AT = UtcDateTime(datetime(2026, 9, 16, 12, tzinfo=UTC))
DELIVERED_AT = UtcDateTime(datetime(2026, 9, 16, 13, tzinfo=UTC))


def backbone_id(number: int = 1) -> BackboneId:
    return BackboneId(UUID(int=number))


def pretraining_input(**overrides: Any) -> PretrainingInput:
    stated: dict[str, Any] = {
        "corpus": CORPUS.name,
        "corpus_version": CorpusVersionId(UUID(int=7)),
        "corpus_checksum": Checksum.of_bytes(b"source data"),
        "manifest": MANIFEST,
        "block_checksum": CORPUS.checksum,
        "vocabulary_size": CORPUS.vocabulary_size,
    }
    return PretrainingInput(**(stated | overrides))


def backbone(**overrides: Any) -> Backbone:
    """A backbone as ordered, over the test corpus; anything named is replaced."""
    stated: dict[str, Any] = {
        "id": backbone_id(),
        "configuration": CONFIGURATION,
        "input": pretraining_input(),
        "run": "first",
        "git_commit": COMMIT,
        "signature": SIGNATURE,
        "ordered_at": ORDERED_AT,
    }
    return Backbone(**(stated | overrides))


def order(**overrides: Any) -> PretrainingOrder:
    stated: dict[str, Any] = {
        "backbone": backbone_id(),
        "configuration": CONFIGURATION,
        "manifest": MANIFEST,
        "run": "first",
        "git_commit": COMMIT,
        "signature": SIGNATURE,
    }
    return PretrainingOrder(**(stated | overrides))


def outcome(epochs: int = 2, first: int = 0) -> TrainingOutcome:
    """A run of that many epochs from ``first``, its weights on the last one."""
    numbers = range(first, first + epochs)
    return TrainingOutcome(
        backbone=WEIGHTS,
        epochs=tuple(
            epoch_outcome(number, backbone=WEIGHTS if number == numbers[-1] else None)
            for number in numbers
        ),
    )


def result(**overrides: Any) -> PretrainingResult:
    """The result of fulfilling the default order as ordered; anything named is replaced."""
    stated: dict[str, Any] = {
        "order": ORDER_REF,
        "backbone": backbone_id(),
        "configuration": CONFIGURATION,
        "corpus": CORPUS.shape,
        "git_commit": COMMIT,
        "resumed_from": None,
        "outcome": outcome(),
    }
    return PretrainingResult(**(stated | overrides))
