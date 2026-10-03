"""The published network's answers as a side of an export, paired by the campaigns' script."""

from pathlib import Path

import pytest

from scripts.campaign_pairs_report import (
    CELLS,
    PREDICTIONS,
    CellRow,
    PredictionRow,
    read_predictions,
    write_cells,
    write_predictions,
)
from scripts.campaign_pairs_report import main as pairs_main
from scripts.strats_reference_answers import (
    UnpairableAnswersError,
    main,
    side_rows,
    windows_of,
)

CAMPAIGN = "f8c5225b-872f-4c7a-97a7-5b0861827486"
STAYS = ("set-a/1", "set-a/2", "set-a/3", "set-a/4")
TARGETS = (0.0, 1.0, 0.0, 1.0)


def exported(into: Path) -> None:
    """A campaign's network, two seeds, ranking three of the four pairs of stays right."""
    cells, predictions = [], []
    for seed in (1, 2):
        for position, (stay, target) in enumerate(zip(STAYS, TARGETS, strict=True)):
            answer = (0.6, 0.4, 0.2, 0.8)[position]
            cells.append(
                CellRow(
                    CAMPAIGN,
                    "selection",
                    "M",
                    "auroc_shortfall",
                    "from_scratch",
                    "all",
                    seed,
                    stay,
                    (answer - target) ** 2,
                    1,
                    10.0,
                )
            )
            predictions.append(
                PredictionRow(
                    CAMPAIGN, "from_scratch", "all", seed, stay, position, 48.0, target, answer
                )
            )
    write_cells(into / CELLS, cells)
    write_predictions(into / PREDICTIONS, predictions)


def run(into: Path, seed: int, answers: tuple[float, ...]) -> Path:
    directory = into / f"run-{seed}"
    directory.mkdir()
    (directory / "run.json").write_text('{"seconds": 5.0}')
    lines = ["seed,unit,target,predicted"] + [
        f"{seed},{stay},{target},{answer}"
        for stay, target, answer in zip(STAYS, TARGETS, answers, strict=True)
    ]
    (directory / "predictions.csv").write_text("\n".join(lines) + "\n")
    return directory


def test_an_answer_takes_the_window_its_stay_has_in_the_export(tmp_path: Path) -> None:
    exported(tmp_path)

    windows = windows_of(read_predictions(tmp_path / PREDICTIONS))
    cells, predictions = side_rows(
        [(1, "set-a/3", 0.0, 0.1)],
        {1: 5.0},
        windows,
        name="strats",
        candidate="ss-",
        purpose="selection",
    )

    assert (predictions[0].position, predictions[0].ends_at) == (2, 48.0)
    assert cells[0].squared_error == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("answer", "message"),
    [((1, "set-a/9", 0.0, 0.1), "no window"), ((1, "set-a/1", 1.0, 0.1), "labelled")],
)
def test_an_answer_off_the_export_is_refused(
    tmp_path: Path, answer: tuple[int, str, float, float], message: str
) -> None:
    exported(tmp_path)

    with pytest.raises(UnpairableAnswersError, match=message):
        side_rows(
            [answer],
            {1: 5.0},
            windows_of(read_predictions(tmp_path / PREDICTIONS)),
            name="strats",
            candidate="ss-",
            purpose="selection",
        )


def test_the_side_pairs_with_the_campaign_and_writing_again_does_not_double_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exported(tmp_path)
    runs = [str(run(tmp_path, seed, (0.1, 0.9, 0.2, 0.8))) for seed in (1, 2)]
    arguments = [
        "--into",
        str(tmp_path),
        "--side",
        "strats",
        "ss-",
        "--purpose",
        "selection",
        "--runs",
        *runs,
    ]

    main(arguments)
    main(arguments)
    pairs_main(
        [
            "--out",
            str(tmp_path),
            "--resamples",
            "200",
            "--pair",
            CAMPAIGN,
            "from_scratch",
            "all",
            "strats",
            "ss-",
            "all",
        ]
    )

    printed = capsys.readouterr().out
    assert "4 pairs" in printed
    assert "+0.250" in printed


def test_two_runs_of_one_seed_are_refused(tmp_path: Path) -> None:
    exported(tmp_path)
    first = run(tmp_path, 1, (0.1, 0.9, 0.2, 0.8))

    with pytest.raises(SystemExit, match="one run per seed"):
        main(
            [
                "--into",
                str(tmp_path),
                "--side",
                "strats",
                "ss-",
                "--purpose",
                "selection",
                "--runs",
                str(first),
                str(first),
            ]
        )
