"""The figure draws from the stored files alone."""

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.label_curve_figures import STEM, draw, main
from scripts.label_curve_report import (
    Baseline,
    Comparison,
    Curve,
    CurvePoint,
    write,
)


def point(mode: str, budget: str, seed: int, rmse: float) -> CurvePoint:
    return CurvePoint(
        mode=mode,
        budget=budget,
        windows=2651 if budget == "all" else int(budget),
        seed=seed,
        engines=10 if budget == "50" else 18,
        trainable_parameters=257,
        rmse=rmse,
        rmse_below_ceiling=rmse,
        last_window_rmse=rmse,
        alpha_lambda_accuracy=0.5,
        asymmetric_score=100.0,
        seconds=2.0,
    )


def comparison(mode: str, budget: str, reduction: float, *, primary: bool) -> Comparison:
    return Comparison(
        mode=mode,
        budget=budget,
        seeds=2,
        control_rmse=40.0,
        candidate_rmse=40.0 - reduction,
        control_sd=1.0,
        candidate_sd=1.0,
        reduction=reduction,
        relative_reduction=reduction / 40.0,
        low=reduction - 3.0,
        high=reduction + 3.0,
        p_value=0.01,
        floor=1.2,
        primary=primary,
        rejected=True,
        verdict="confirmed" if primary else "distinguishable",
    )


def curve() -> Curve:
    modes = ("from_scratch", "frozen_probe", "lora", "full_fine_tuning")
    errors = (40.0, 38.0, 25.0, 22.0)
    return Curve(
        points=tuple(
            point(mode, budget, seed, error - (5.0 if budget == "all" else 0.0) + seed)
            for budget in ("50", "200", "all")
            for mode, error in zip(modes, errors, strict=True)
            for seed in (1, 2)
        ),
        comparisons=tuple(
            comparison(
                mode, budget, 40.0 - error, primary=(mode, budget) == ("full_fine_tuning", "200")
            )
            for budget in ("50", "200", "all")
            for mode, error in zip(modes[1:], errors[1:], strict=True)
        ),
        baselines=(
            Baseline(name="mean predictor", rmse=41.0),
            Baseline(name="ceiling predictor", rmse=66.0),
        ),
    )


def test_the_figure_is_drawn_from_the_curve(tmp_path: Path) -> None:
    figure = draw(curve(), tmp_path / "figures" / "curve.png")

    assert figure.is_file()
    assert figure.stat().st_size > 10_000


def test_a_partial_grid_is_drawn_over_the_budgets_each_mode_has(tmp_path: Path) -> None:
    whole = curve()
    partial = Curve(
        points=tuple(p for p in whole.points if not (p.mode == "lora" and p.budget == "all")),
        comparisons=tuple(
            row for row in whole.comparisons if not (row.mode == "lora" and row.budget == "all")
        ),
        baselines=whole.baselines,
    )

    figure = draw(partial, tmp_path / "partial.png")

    assert figure.is_file()


def test_the_figure_lands_beside_the_curve_unless_told_otherwise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(curve(), tmp_path / "curve")

    main([str(tmp_path / "curve")])

    assert (tmp_path / "curve" / f"{STEM}.png").is_file()
    assert capsys.readouterr().out.strip().endswith(f"{STEM}.png")
