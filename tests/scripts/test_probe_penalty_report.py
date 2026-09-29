"""The probe's penalties traced over draws, stored and rendered; the draws need the corpus."""

from collections.abc import Sequence
from dataclasses import replace
from math import log
from pathlib import Path

import pytest
import torch
from torch import Tensor

from emblema.evaluation.domain.heads.head_pooling import HeadPooling
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from scripts.probe_penalty_report import (
    CURVES,
    FURTHER,
    IN_FORCE,
    PenaltyPoint,
    Sampler,
    main,
    prevalence_log_loss,
    read,
    render,
    trace,
    write,
)

WEIGHTS = ArtifactRef("durable/sha256/" + "a" * 64, Checksum.parse("sha256:" + "a" * 64))


BASE = PenaltyPoint(
    backbone="physionet-32",
    pooling="mean",
    budget="200",
    seed=1,
    windows=200,
    positives=28,
    penalty=100.0,
    log_loss=0.3712,
    prevalence_log_loss=0.4053,
    chosen=100.0,
    failure="",
    embed_seconds=4.2,
    fit_seconds=0.7,
)


def point(
    penalty: float, loss: float | None, *, budget: str = "200", chosen: float | None = 100.0
) -> PenaltyPoint:
    failure = "" if loss is not None else "the fit did not converge in 100 iterations"
    return replace(
        BASE, penalty=penalty, log_loss=loss, budget=budget, chosen=chosen, failure=failure
    )


class _Drawn(Sampler):
    """Two hundred stays whose first state carries the outcome, in place of the corpus."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self._outcomes = [1.0 if index % 7 == 0 else 0.0 for index in range(200)]
        noise = torch.randn(200, 8, generator=generator, dtype=torch.float64)
        noise[:, 0] += 2.0 * torch.tensor(self._outcomes, dtype=torch.float64)
        self._states = noise

    def drawn(self, budget: str, seed: int) -> tuple[list[TokenWindow], list[float], float]:
        return [], self._outcomes, sum(self._outcomes) / len(self._outcomes)

    def embedded(
        self,
        weights: ArtifactRef,
        pooling: HeadPooling,
        seed: int,
        windows: Sequence[TokenWindow],
        mean_target: float,
    ) -> Tensor:
        return self._states


def test_the_longer_list_holds_the_one_in_force_and_runs_past_both_ends() -> None:
    assert FURTHER[1 : 1 + len(IN_FORCE)] == IN_FORCE
    assert list(FURTHER) == sorted(FURTHER)
    assert min(FURTHER) < min(IN_FORCE)
    assert max(FURTHER) > max(IN_FORCE)


def test_the_prevalence_answers_each_fold_with_the_share_of_its_kept_part() -> None:
    outcomes = [1.0] * 10 + [0.0] * 40

    # Every fold keeps eight deaths of forty stays, so every answer is a fifth.
    expected = (10 * -log(0.2) + 40 * -log(0.8)) / 50
    assert prevalence_log_loss(outcomes) == pytest.approx(expected)


def test_a_trace_gives_every_strength_its_folds_loss_beside_the_candidates_choice() -> None:
    points = trace(_Drawn(), [("physionet-32", WEIGHTS)], ["200"], [1], FURTHER)

    assert len(points) == 2 * len(FURTHER)
    assert {p.pooling for p in points} == {"mean", "tail-0.2"}
    mean = [p for p in points if p.pooling == "mean"]
    best = min((p for p in mean if p.log_loss is not None), key=lambda p: p.log_loss or 0.0)
    assert mean[0].chosen == best.penalty
    assert best.log_loss is not None
    assert best.log_loss < mean[0].prevalence_log_loss
    assert (mean[0].windows, mean[0].positives) == (200, 29)


def test_the_points_are_written_and_read_back_as_they_were(tmp_path: Path) -> None:
    rows = [point(0.001, None, chosen=None), point(100.0, 0.3712)]

    assert write(rows, tmp_path).name == CURVES
    assert read(tmp_path) == tuple(rows)


def test_the_table_places_the_choice_against_the_list_in_force_and_names_the_failures() -> None:
    inside = [point(10.0, 0.38), point(100.0, 0.3712), point(1000.0, 0.39)]
    strongest = [point(1e7, 0.4050, budget="50", chosen=1e7)]
    failed = [point(0.0001, None, budget="all", chosen=None)]

    rendered = render([*inside, *strongest, *failed])

    assert "| physionet-32 | mean | 200 | 1 | 28 | 100 | 100 (0.3712) | 0.4053 | inside | — |" in (
        rendered
    )
    assert "| 50 | 1 | 28 | 1e+07 | 1e+07 (0.4050) | 0.4053 | strongest | — |" in rendered
    assert "| all | 1 | 28 | failed | — | 0.4053 | failed | 0.0001 |" in rendered


def test_a_run_without_what_it_needs_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--backbone NAME KEY CHECKSUM are required"):
        main(["--out", str(tmp_path), "--manifest", "k", "sha256:" + "a" * 64])
    with pytest.raises(SystemExit, match="nothing to render"):
        main(["--report-only", str(tmp_path)])
