from pathlib import Path

import pytest

from emblema.pretraining.domain.training.training_outcome import TrainingOutcome
from scripts.pretraining_curve_report import CurveRow, corpora_of, main, read, render, write
from tests.support.experiments import WEIGHTS, epoch_outcome, validated
from tests.support.handoff import result


def two_corpora() -> list[CurveRow]:
    epochs = (
        epoch_outcome(
            0,
            validation=(validated(loss=0.5), validated(corpus="second", loss=0.8)),
            weights=WEIGHTS,
        ),
        epoch_outcome(
            1,
            validation=(validated(loss=0.6), validated(corpus="second", loss=0.7)),
            backbone=WEIGHTS,
        ),
    )
    stated = result(outcome=TrainingOutcome(backbone=WEIGHTS, epochs=epochs))
    return CurveRow.of(stated)


def test_a_result_gives_a_row_per_corpus_and_epoch_in_the_runs_order() -> None:
    rows = two_corpora()

    assert [(row.epoch, row.corpus) for row in rows] == [
        (0, "invented"),
        (0, "second"),
        (1, "invented"),
        (1, "second"),
    ]
    assert corpora_of(rows) == ["invented", "second"]
    assert [row.relative for row in rows] == [0.5, 0.8, 0.6, 0.7]
    assert [(row.kept, row.backbone) for row in rows] == [
        (True, True),
        (True, True),
        (False, False),
        (False, False),
    ]
    assert (rows[0].experiment, rows[0].tier, rows[0].precision) == ("test-experiment", "S", "fp32")


def test_the_rows_come_back_from_the_file_as_they_went_in(tmp_path: Path) -> None:
    rows = two_corpora()

    stored = write(rows, tmp_path / "curve")

    assert stored == tmp_path / "curve" / "epochs.csv"
    assert read(tmp_path / "curve") == rows


def test_the_table_has_a_line_per_epoch_and_names_the_epoch_kept() -> None:
    rendered = render(two_corpora())

    lines = [
        line for line in rendered.splitlines() if line.startswith("| 1 ") or line.startswith("| 2 ")
    ]
    assert len(lines) == 2
    assert "0.650" in lines[0]
    assert "backbone" in lines[0]
    assert "0.650" in lines[1]
    assert "backbone" not in lines[1]
    assert rendered.startswith("test-experiment: tier S, fp32; validation, not test")


def test_a_stored_curve_renders_without_the_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(two_corpora(), tmp_path)

    main(["--report-only", str(tmp_path)])

    assert "| epoch |" in capsys.readouterr().out


def test_neither_a_result_nor_a_stored_curve_is_refused() -> None:
    with pytest.raises(SystemExit):
        main([])
