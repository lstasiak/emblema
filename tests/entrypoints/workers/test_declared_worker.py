"""The scalars a worker is given, read back into the values that judge them.

One place, because every process that reads the same group has to read it the same way: a
campaign declared under one reading and a cell run under another would be refused by the check
that compares them, and rightly.
"""

from pathlib import Path

import pytest

from emblema.config.boosting_settings import BoostingSettings
from emblema.config.convolution_settings import ConvolutionSettings
from emblema.config.lora_settings import LoraSettings
from emblema.config.schedule_settings import ScheduleSettings
from emblema.config.worker_settings import WorkerSettings
from emblema.entrypoints.workers.declared_worker import DeclaredWorker
from tests.evaluation.support import LORA, adaptation_schedule, boosting, convolutions

DECLARED = WorkerSettings(
    workspace=Path("data/workspace"),
    corpora=Path("data/raw"),
    schedule=ScheduleSettings(
        epochs=2,
        min_steps=0,
        batch_size=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        warmup_fraction=0.0,
        final_lr_fraction=1.0,
    ),
    lora=LoraSettings(
        rank=2, alpha=4.0, dropout=0.0, targets="qkv, attention.projection, feedforward"
    ),
    boosting=BoostingSettings(
        rounds=8,
        max_depth=3,
        learning_rate=0.3,
        row_share=1.0,
        feature_share=1.0,
        min_leaf_weight=1.0,
        l2_penalty=1.0,
        threads=1,
    ),
    convolutions=ConvolutionSettings(features=84, ridge_penalties="0.1, 1, 10", threads=1),
)
BARE = WorkerSettings(workspace=Path("data/workspace"), corpora=Path("data/raw"))


def test_the_schedule_and_the_updates_are_the_ones_the_environment_declares() -> None:
    declared = DeclaredWorker(DECLARED)

    assert declared.schedule() == adaptation_schedule()
    assert declared.lora() == LORA


def test_the_layers_are_split_out_of_the_one_string_an_environment_can_carry() -> None:
    # A list read from the environment is decoded as JSON before any validator of ours runs, so
    # the layers travel as one string and are split here.
    assert DeclaredWorker(DECLARED).lora().targets == LORA.targets


def test_the_fit_is_the_one_the_environment_declares() -> None:
    assert DeclaredWorker(DECLARED).boosting() == boosting()


def test_the_convolutions_are_the_ones_the_environment_declares() -> None:
    # The penalties travel as one string, for the reason the layers do.
    assert DeclaredWorker(DECLARED).convolutions() == convolutions()


def test_a_penalty_that_is_not_a_number_is_refused() -> None:
    garbled = DECLARED.model_copy(
        update={
            "convolutions": DECLARED.require_convolutions().model_copy(
                update={"ridge_penalties": "0.1, ten"}
            )
        }
    )

    with pytest.raises(ValueError, match="ten"):
        DeclaredWorker(garbled).convolutions()


@pytest.mark.parametrize("missing", ["schedule", "lora", "boosting", "convolutions"])
def test_a_value_the_worker_was_not_given_is_refused_where_it_is_asked_for(missing: str) -> None:
    with pytest.raises(ValueError, match="this worker"):
        getattr(DeclaredWorker(BARE), missing)()
