"""What the loop report compares, and what it refuses before anything trains.

No model is trained here: the comparisons read numbers and weights, and the tests give them those.
What is held is that a resumed run is compared only where it measured the same thing as a run
left alone, and that pairs are grouped by whether a resume is in them.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from scripts.training_loop_report import (
    RESUMED,
    UNINTERRUPTED,
    Measured,
    largest_loss_difference,
    pair_differences,
    report,
    shared_artifacts,
)
from tests.support.experiments import epoch_outcome

pytestmark = pytest.mark.ml


def measured(kind: str, *, weight: float = 0.0, stored: str = "weights", **epoch: Any) -> Measured:
    """A three-epoch run whose second epoch may be overridden, with a single weight."""
    return Measured(
        kind=kind,
        epochs=(epoch_outcome(0), epoch_outcome(1, **epoch), epoch_outcome(2)),
        weights=ArtifactRef(f"durable/{stored}", Checksum.of_bytes(stored.encode())),
        values={"weight": torch.tensor([weight])},
        seconds=1.0,
    )


def test_pairs_are_grouped_by_whether_a_resume_is_in_them() -> None:
    runs = [measured(UNINTERRUPTED), measured(RESUMED), measured(UNINTERRUPTED), measured(RESUMED)]

    grouped = pair_differences(runs)

    assert {pair: len(differences) for pair, differences in grouped.items()} == {
        (UNINTERRUPTED, UNINTERRUPTED): 1,
        (UNINTERRUPTED, RESUMED): 4,
        (RESUMED, RESUMED): 1,
    }


def test_a_pair_reports_the_largest_weight_difference_between_its_runs() -> None:
    grouped = pair_differences([measured(UNINTERRUPTED, weight=0.5), measured(RESUMED, weight=0.2)])

    ((weights, losses),) = grouped[(UNINTERRUPTED, RESUMED)]
    assert weights == pytest.approx(0.3)
    assert losses == 0.0


def test_the_training_loss_of_a_re_entered_epoch_is_never_compared() -> None:
    """A resumed run trains only the batches after its checkpoint: that loss is not the epoch's."""
    left = measured(UNINTERRUPTED)
    right = measured(RESUMED, training_loss=5.0)

    assert largest_loss_difference(left, right) == 0.0


def test_every_epochs_validation_loss_and_the_last_training_loss_are_compared() -> None:
    left = measured(UNINTERRUPTED)
    moved_validation = measured(RESUMED, validation_loss=0.9)
    last = left.epochs[-1]
    moved_training = replace(
        left, epochs=(*left.epochs[:-1], replace(last, training_loss=last.training_loss + 0.25))
    )

    assert largest_loss_difference(left, moved_validation) == pytest.approx(0.9 - 1.2 / 2)
    assert largest_loss_difference(left, moved_training) == pytest.approx(0.25)


def test_whether_runs_left_the_same_artifact_is_said_over_every_pair() -> None:
    same = [measured(UNINTERRUPTED), measured(RESUMED)]
    apart = [measured(UNINTERRUPTED, stored="a"), measured(RESUMED, stored="b")]
    some = [*same, measured(RESUMED, stored="c")]

    assert shared_artifacts(same) == "yes, every run"
    assert shared_artifacts(apart) == "no two runs"
    assert shared_artifacts(some) == "1 of 3 pairs"


def test_a_precision_the_device_does_not_run_is_refused_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        report(["--device", "cpu", "--precision", "fp16", "--workspace", str(tmp_path / "loop")])

    assert "cpu does not run fp16" in capsys.readouterr().err
    assert not (tmp_path / "loop").exists()


def test_a_run_with_nothing_to_repeat_is_refused_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        report(["--device", "cpu", "--repeats", "0", "--workspace", str(tmp_path / "loop")])

    assert "--repeats must be at least 1" in capsys.readouterr().err
    assert not (tmp_path / "loop").exists()
