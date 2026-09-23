"""Which corpora a process can answer for, and what it does about the ones it cannot.

A campaign over a corpus whose answers nothing here knows how to read must fail by name, not by
being handed the reader of another corpus and quietly answered with the wrong numbers.
"""

from pathlib import Path

import pytest

from emblema.entrypoints.known_ground_truths import KnownGroundTruths
from emblema.evaluation.domain.exceptions import (
    UnknownGroundTruthError,
    UnreadableGroundTruthError,
)
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow

WINDOW = TaskWindow(unit=UnitKey("FD001/1"), position=0, ends_at=1.0)


def test_a_corpus_nothing_here_reads_is_refused_by_name(tmp_path: Path) -> None:
    truths = KnownGroundTruths.under(tmp_path)

    with pytest.raises(UnknownGroundTruthError, match="knows no ground truth of corpus 'skab'"):
        truths.truths_of("skab", [WINDOW])


def test_the_turbofans_are_read_from_the_raw_corpora_the_process_was_given(
    tmp_path: Path,
) -> None:
    truths = KnownGroundTruths.under(tmp_path)

    # Nothing has been fetched into the directory, so the reader reaches for the run-to-failure
    # file and says which one is missing. Where it reached is the evidence of what it routed to.
    with pytest.raises(UnreadableGroundTruthError) as refused:
        truths.truths_of(KnownGroundTruths.CMAPSS, [WINDOW])

    assert str(tmp_path / "train_FD001.txt") in str(refused.value)
