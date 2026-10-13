"""The stage reader's two figures, drawn from the exported scores and power; nothing is computed."""

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.campaign_pairs_report import SCORES, ScoreRow, write_scores
from scripts.comparison_power_report import POWER, PowerRow, write
from scripts.stage_reader_figures import (
    CURVE,
    Recipe,
    direct_power,
    draw_curve,
    draw_power,
    main,
    panels_of,
)

BUDGETS = ("200", "500", "all")


def scores() -> list[ScoreRow]:
    rows = []
    for campaign, shift in (("mix-a", 0.0), ("mix-b", 0.01)):
        for budget, level in zip(BUDGETS, (0.65, 0.72, 0.80), strict=True):
            for candidate, offset in (
                ("from_scratch@learning_rate=0.001", 0.0),
                ("from_scratch@learning_rate=0.001,patience=10,stop_share=0.2", 0.01),
                ("full_fine_tuning", 0.02),
                ("frozen_ridge@layer=concat", -0.03),
                ("untrained_ridge@layer=concat", -0.06),
            ):
                for seed in (1, 2, 3):
                    rows.append(
                        ScoreRow(
                            campaign,
                            "selection",
                            "auroc_shortfall",
                            candidate,
                            budget,
                            seed,
                            1999,
                            level + offset + shift + 0.004 * seed,
                            30.0,
                        )
                    )
    for budget, level in zip(BUDGETS, (20.0, 17.0, 15.0), strict=True):
        for seed in (1, 2):
            rows.append(
                ScoreRow(
                    "fd-a",
                    "selection",
                    "rmse",
                    "from_scratch@learning_rate=0.003,pooling=tail,tail_share=0.2",
                    budget,
                    seed,
                    40,
                    level + 0.5 * seed,
                    90.0,
                )
            )
    return rows


def power() -> list[PowerRow]:
    rows = []
    for kind, threshold in (("A", None), ("B", 0.004), ("B", 0.006)):
        for budget, level in zip(BUDGETS, (0.2, 0.5, 0.9), strict=True):
            for stays in (800, 1999):
                for gain in (0.0, 0.01, 0.02):
                    for source in ("direct", "model"):
                        rows.append(
                            PowerRow(
                                kind,
                                f"{kind} pair",
                                budget,
                                stays,
                                10,
                                gain,
                                threshold,
                                source,
                                min(1.0, level * gain * 50 + (0.1 if stays == 800 else 0.0)),
                            )
                        )
    return rows


def test_a_recipe_is_its_bare_name_and_whether_it_stops() -> None:
    assert Recipe.of("from_scratch@learning_rate=0.001,patience=10,stop_share=0.2") == Recipe(
        "from_scratch", True
    )
    assert Recipe.of("full_fine_tuning") == Recipe("full_fine_tuning", False)
    assert Recipe.of("from_scratch@patience_steps=500,stop_share=0.2").label == "from scratch, stop"
    assert Recipe.of("untrained_ridge@layer=concat").order > Recipe.of("frozen_ridge").order


def test_panels_group_the_cells_by_task_and_name_each_campaign() -> None:
    panels = panels_of(
        scores(),
        [("stays", "s1", "mix-a"), ("stays", "s2", "mix-b"), ("FD001", "s1", "fd-a")],
        {"stays": 1998},
    )

    assert [p.task for p in panels] == ["stays", "FD001"]
    assert panels[0].by_area
    assert not panels[1].by_area
    assert panels[0].labels == {"mix-a": "s1", "mix-b": "s2"}
    assert panels[0].position("all") == 1998
    line = list(panels[0].line(Recipe("from_scratch", True)))
    assert [x for x, _, _, _ in line] == [200, 500, 1998]
    assert line[0][3] == pytest.approx([0.668, 0.678])
    with pytest.raises(SystemExit, match="every labelled unit"):
        panels[1].position("all")


def test_a_campaign_without_cells_is_refused() -> None:
    with pytest.raises(SystemExit, match="no cell of other"):
        panels_of(scores(), [("stays", "s1", "other")], {})


def test_direct_power_takes_the_largest_stays_and_the_largest_spread() -> None:
    found = direct_power(power())

    assert found[("A", "500", 0.01)] == [pytest.approx(0.25)]
    assert found[("B", "500", 0.02)] == [pytest.approx(0.5)]
    assert all(kind in {"A", "B"} for kind, _, _ in found)


def test_both_figures_are_drawn_from_the_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_scores(tmp_path / SCORES, scores())
    write(tmp_path / POWER, PowerRow._fields, power())

    main(
        [
            "--scores",
            str(tmp_path / SCORES),
            *("--campaign", "stays", "s1", "mix-a"),
            *("--campaign", "stays", "s2", "mix-b"),
            *("--campaign", "FD001", "s1", "fd-a"),
            *("--everything", "stays", "1998"),
            *("--everything", "FD001", "1270"),
            "--power",
            str(tmp_path / POWER),
            *("--kinds", "B", "A"),
            "--out",
            str(tmp_path / "figures"),
        ]
    )

    assert (tmp_path / "figures" / CURVE).stat().st_size > 10_000
    assert (tmp_path / "figures" / "stage-reader-power.png").stat().st_size > 10_000
    assert capsys.readouterr().out.count("stage-reader") == 2


def test_a_kind_the_power_file_lacks_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no power read for kinds \\['Z'\\]"):
        draw_power(power(), tmp_path / "p.png", kinds=["A", "Z"])


def test_a_curve_needs_a_panel_and_the_arguments_go_together(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no task"):
        draw_curve([], tmp_path / "x.png")
    with pytest.raises(SystemExit):
        main(["--scores", str(tmp_path / "s.csv")])
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path)])
