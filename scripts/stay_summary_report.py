"""What the intensive-care task reads at a few labels, and whether the frozen states carry it.

Three readings, all on the half of the tuning stays a selection holds out and all off arrays
stored once by ``stay_summary_states.py``, so nothing here reaches an accelerator or a registry.
First, a logistic regression over the per-channel summaries of a stay — levels, extremes, trends,
how often a channel was measured, and the static descriptors — fitted on the labels a campaign
would draw at each budget and seed. Second, how well a linear map recovers each of those
summaries from the frozen states of the stage reader's probe, learnt on every stay of the other
half, since a summary needs no label, and scored on the held-out half; beside every pretrained
encoder the same encoder at its initialisation, drawn as the campaign drew it. Third, the probe's
own head over the states with the summaries beside them, against the summaries alone.

Every head is the one the probe solves on an outcome: logistic regression whose penalty is chosen
by the folds' log-loss among the strengths the campaigns ran under. The probe over the states
alone is fitted too, and must answer as the registered cells answered: that is how a reader knows
the stays, the draws and the encoders here are the campaign's.

The fits are written to CSV first, in the shape ``campaign_pairs_report.py`` pairs, so every pair
is made by the stage's own pairing beside the registered cells; the tables are made from those
files, so a table can be reshaped without fitting anything again.

    uv run scripts/stay_summary_report.py DIR [--reference DIR]
    uv run scripts/stay_summary_report.py DIR --report-only [--reference DIR]
"""

import argparse
import csv
import json
import sys
import time
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from typing import ClassVar, Self

import numpy as np
import scipy.linalg
import torch
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.adapters.torch.logistic_solution import LogisticSolution
from emblema.evaluation.domain.heads.ridge_penalties import RidgePenalties
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.error_measure import ErrorMeasure
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.scoring.window_ranking import WindowRanking
from emblema.evaluation.domain.task.run_purpose import RunPurpose
from scripts.campaign_pairs_report import (
    CELLS,
    PREDICTIONS,
    SCORES,
    CellRow,
    PredictionRow,
    read_scores,
    write_cells,
    write_predictions,
)
from scripts.reporting import table

# What the storing step writes and this reads: the arrays, the stays they are rows of, the draws,
# the encoders the states are of, the channels the summaries are laid over, and the conditions.
STATES = "states.npz"
STAYS = "stays.csv"
DRAWS = "draws.csv"
ENCODERS = "encoders.csv"
CHANNELS = "channels.csv"
CONDITIONS = "conditions.json"
# Where the storing step leaves each encoder's mean state per channel, one file per encoder named
# by its digest, laid out channel by channel.
CHANNEL_STATES = "channel_states"
# What a stay's measured values are when it is embedded: as recorded, each channel's mean over the
# learnt half, or values drawn from another stay of the learnt half.
RECORDED, CONSTANT, TRANSPLANTED = "recorded", "constant", "transplanted"
VALUES = (RECORDED, CONSTANT, TRANSPLANTED)
# The arrays besides the states, which are keyed by their encoder's digest.
SUMMARIES = "summaries"
SUMMARY_CHANNELS = "summary_channels"
SUMMARY_STATISTICS = "summary_statistics"
# What this writes beside the pairing's two files.
FITS = "fits.csv"
DECODING = "decoding.csv"
REPORT = "report.md"

# The two halves of the tuning side, and the two ways an encoder enters.
LEARNT, SCORED = "learnt", "scored"
PRETRAINED, UNTRAINED = "pretrained", "untrained"
# What each reading is called in the pairing's files. A reading over the summaries alone has no
# encoder, so it sits under a campaign of its own name.
STATES_ALONE = "states"
STATES_AND_SUMMARIES = "states_and_summaries"
SUMMARIES_ALONE = "summaries"
SUMMARIES_SOURCE = "summaries"
UNTRAINED_PREFIX = "untrained_"
# What the registered probe cells are called, against which the reading over states alone is held.
REGISTERED = {PRETRAINED: "frozen_ridge@layer=concat", UNTRAINED: "untrained_ridge@layer=concat"}

# Each summary a classical model reads, by what it says about a channel over the stay.
LEVEL, EXTREMES, TREND, FREQUENCY, STATICS = "level", "extremes", "trend", "frequency", "statics"
FAMILIES = (LEVEL, EXTREMES, TREND, FREQUENCY, STATICS)
FAMILY_OF = {
    "count": FREQUENCY,
    "gap": FREQUENCY,
    "mean": LEVEL,
    "first": LEVEL,
    "last": LEVEL,
    "minimum": EXTREMES,
    "maximum": EXTREMES,
    "deviation": EXTREMES,
    "slope": TREND,
    "step": TREND,
}
# A static descriptor is one token, so its mean is its value and its count whether it was
# recorded; every other statistic of it repeats the value or is undefined.
STATIC_STATISTICS = ("count", "mean")
# Fewer stays than this on either half leave a summary's fit or its score to a handful of stays.
MINIMUM_STAYS = 50
# The share of variance a family's median summary must reach for the decision rule's two edges.
CARRIED, NOT_CARRIED = 0.5, 0.3


def family_of(statistic: str, *, timeless: bool) -> str | None:
    """The family a summary belongs to, or ``None`` for a statistic of a static that repeats one.

    Raises:
        ValueError: If the statistic is not one the summaries hold.
    """
    if statistic not in FAMILY_OF:
        raise ValueError(f"{statistic!r} is not a summary of a channel")
    if timeless:
        return STATICS if statistic in STATIC_STATISTICS else None
    return FAMILY_OF[statistic]


def _number(value: float) -> str:
    return repr(float(value))


def write_rows(path: Path, columns: Sequence[str], records: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True, kw_only=True)
class StoredStay:
    """One stay of the tuning side, as a row of the stored arrays.

    Attributes:
        row: Index of the stay's row in every stored array.
        half: Which half of the tuning side the stay is on: learnt from, or scored.
        unit: The stay.
        position: Index of its window in the published block.
        ends_at: When the window ends, in the corpus's time.
        target: The outcome, one for a death.
    """

    row: int
    half: str
    unit: str
    position: int
    ends_at: float
    target: float

    COLUMNS = ("row", "half", "unit", "position", "ends_at", "target")

    def record(self) -> dict[str, str]:
        return {
            "row": str(self.row),
            "half": self.half,
            "unit": self.unit,
            "position": str(self.position),
            "ends_at": _number(self.ends_at),
            "target": _number(self.target),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            row=int(record["row"]),
            half=record["half"],
            unit=record["unit"],
            position=int(record["position"]),
            ends_at=float(record["ends_at"]),
            target=float(record["target"]),
        )

    def answered(self, predicted: float) -> WindowPrediction:
        window = TaskWindow(unit=UnitKey(self.unit), position=self.position, ends_at=self.ends_at)
        return WindowPrediction(window=window, target=self.target, predicted=predicted)


@dataclass(frozen=True, kw_only=True)
class StoredDraw:
    """One stay of the labels a cell drew at a budget under a seed."""

    budget: str
    seed: int
    unit: str
    position: int

    COLUMNS = ("budget", "seed", "unit", "position")

    def record(self) -> dict[str, str]:
        return {
            "budget": self.budget,
            "seed": str(self.seed),
            "unit": self.unit,
            "position": str(self.position),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            budget=record["budget"],
            seed=int(record["seed"]),
            unit=record["unit"],
            position=int(record["position"]),
        )


@dataclass(frozen=True, kw_only=True)
class StoredEncoder:
    """The encoder one cell of a campaign read, named by the digest its states are stored under.

    Attributes:
        source: The name the campaign was given on the command line.
        campaign: The campaign's identifier in the registry.
        arm: Whether the cell read the pretrained encoder or the same shape at initialisation.
        seed: The cell's seed.
        digest: Digest of the encoder's and the pooling's weights; cells that read the same
            weights share it, and their states are stored once.
    """

    source: str
    campaign: str
    arm: str
    seed: int
    digest: str

    COLUMNS = ("source", "campaign", "arm", "seed", "digest")

    def record(self) -> dict[str, str]:
        return {
            "source": self.source,
            "campaign": self.campaign,
            "arm": self.arm,
            "seed": str(self.seed),
            "digest": self.digest,
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            source=record["source"],
            campaign=record["campaign"],
            arm=record["arm"],
            seed=int(record["seed"]),
            digest=record["digest"],
        )


@dataclass(frozen=True, kw_only=True)
class StoredChannel:
    """A channel the summaries are laid over: one the learnt half observed."""

    channel_id: int
    name: str
    timeless: bool

    COLUMNS = ("channel_id", "name", "timeless")

    def record(self) -> dict[str, str]:
        return {
            "channel_id": str(self.channel_id),
            "name": self.name,
            "timeless": str(self.timeless).lower(),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            channel_id=int(record["channel_id"]),
            name=record["name"],
            timeless=record["timeless"] == "true",
        )


def save_arrays(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> None:
    """Every array under its name, uncompressed: the states are noise to a compressor."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.ascontiguousarray(array))


@dataclass(frozen=True)
class Stored:
    """What the storing step left in a directory, read back."""

    stays: tuple[StoredStay, ...]
    draws: tuple[StoredDraw, ...]
    encoders: tuple[StoredEncoder, ...]
    channels: tuple[StoredChannel, ...]
    conditions: Mapping[str, object]
    arrays: Mapping[str, NDArray[np.generic]]

    @classmethod
    def read(cls, directory: Path) -> Self:
        with np.load(directory / STATES) as archive:
            arrays = {name: archive[name] for name in archive.files}
        return cls(
            stays=tuple(StoredStay.parse(row) for row in read_rows(directory / STAYS)),
            draws=tuple(StoredDraw.parse(row) for row in read_rows(directory / DRAWS)),
            encoders=tuple(StoredEncoder.parse(row) for row in read_rows(directory / ENCODERS)),
            channels=tuple(StoredChannel.parse(row) for row in read_rows(directory / CHANNELS)),
            conditions=json.loads((directory / CONDITIONS).read_text(encoding="utf-8")),
            arrays=arrays,
        )

    @property
    def penalties(self) -> RidgePenalties:
        penalties = self.conditions["penalties"]
        if not isinstance(penalties, list):
            raise ValueError("the conditions name no penalties")
        return RidgePenalties(tuple(float(each) for each in penalties))

    @property
    def tier(self) -> str:
        return str(self.conditions["tier"])

    def half(self, name: str) -> tuple[StoredStay, ...]:
        return tuple(stay for stay in self.stays if stay.half == name)

    def cells(self) -> list[tuple[str, int]]:
        """Every (budget, seed) drawn, in the order the budgets were asked for, seeds ascending."""
        found: dict[tuple[str, int], None] = {}
        for draw in self.draws:
            found.setdefault((draw.budget, draw.seed), None)
        budgets = list(dict.fromkeys(budget for budget, _ in found))
        return sorted(found, key=lambda cell: (budgets.index(cell[0]), cell[1]))

    def drawn(self, budget: str, seed: int) -> list[StoredStay]:
        """The stays one cell learnt from, in the order the draw gave them.

        Raises:
            ValueError: If a drawn stay is not a stored stay of the learnt half.
        """
        by_window = {(stay.unit, stay.position): stay for stay in self.half(LEARNT)}
        found = []
        for draw in self.draws:
            if draw.budget == budget and draw.seed == seed:
                stay = by_window.get((draw.unit, draw.position))
                if stay is None:
                    raise ValueError(
                        f"the draw at {budget} under seed {seed} holds {draw.unit}, which is no "
                        "stored stay of the learnt half"
                    )
                found.append(stay)
        return found

    def states(self, digest: str) -> NDArray[np.float32]:
        return np.asarray(self.arrays[digest], dtype=np.float32)

    @property
    def summaries(self) -> NDArray[np.float64]:
        return np.asarray(self.arrays[SUMMARIES], dtype=np.float64)

    def summary_columns(self) -> list[tuple[StoredChannel, str]]:
        """Each column of the summaries as the channel and the statistic it holds."""
        by_id = {channel.channel_id: channel for channel in self.channels}
        return [
            (by_id[int(channel)], str(statistic))
            for channel, statistic in zip(
                self.arrays[SUMMARY_CHANNELS].tolist(),
                self.arrays[SUMMARY_STATISTICS].tolist(),
                strict=True,
            )
        ]

    def encoder(self, source: str, arm: str, seed: int) -> StoredEncoder:
        for encoder in self.encoders:
            if (encoder.source, encoder.arm, encoder.seed) == (source, arm, seed):
                return encoder
        raise ValueError(f"no {arm} encoder of {source} under seed {seed} was stored")

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(encoder.source for encoder in self.encoders))


def prepared(
    sample: NDArray[np.float64], scored: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The summaries a linear head can read: gaps filled and ranges bounded by the drawn stays.

    A statistic a stay cannot support is filled with its mean over the drawn stays, so nothing of
    the scored half reaches the fit; a column the draw defines fewer than twice distinctly is
    dropped, since the head could only spend its penalty on it. A scored value is bounded to the
    range the drawn stays span: a column a few drawn stays define has, once filled, a spread
    near zero, and a scored stay that holds the channel would otherwise sit thousands of
    deviations out once the head scales its columns. Whether a channel was measured at all is its
    count, which is always defined.
    """
    varying = np.array(
        [np.unique(column[~np.isnan(column)]).size >= 2 for column in sample.T], dtype=bool
    )
    sample, scored = sample[:, varying], scored[:, varying]
    means = np.nanmean(sample, axis=0)
    bounded = np.clip(scored, np.nanmin(sample, axis=0), np.nanmax(sample, axis=0))
    filled_sample = np.where(np.isnan(sample), means, sample)
    filled_scored = np.where(np.isnan(bounded), means, bounded)
    return filled_sample, filled_scored


def logistic_answers(
    sample: NDArray[np.floating],
    outcomes: NDArray[np.float64],
    scored: NDArray[np.floating],
    penalties: RidgePenalties,
) -> tuple[NDArray[np.float64], LogisticSolution]:
    """The probe's head fitted on ``sample`` and its probabilities for every scored row."""
    solution = LogisticSolution.fitted(
        torch.from_numpy(np.asarray(sample)), torch.from_numpy(outcomes), penalties
    )
    raw = torch.from_numpy(np.asarray(scored, dtype=np.float64)) @ solution.weights
    return torch.sigmoid(raw + solution.intercept).numpy(), solution


@dataclass(frozen=True, kw_only=True)
class Reading:
    """One input a head is fitted over, under the campaign and candidate the pairing names it by.

    Attributes:
        source: The campaign the reading is filed under in the pairing's files.
        candidate: What the reading is called there.
        arm: The encoder whose states enter, if any.
        summaries: Whether the summaries enter.
    """

    source: str
    candidate: str
    arm: str | None
    summaries: bool


def readings_of(stored: Stored) -> list[Reading]:
    """The summaries alone, then per source the states with and without them, both arms."""
    found = [Reading(source=SUMMARIES_SOURCE, candidate=SUMMARIES_ALONE, arm=None, summaries=True)]
    for source in stored.sources:
        for arm in (PRETRAINED, UNTRAINED):
            prefix = UNTRAINED_PREFIX if arm == UNTRAINED else ""
            found.append(
                Reading(source=source, candidate=prefix + STATES_ALONE, arm=arm, summaries=False)
            )
            found.append(
                Reading(
                    source=source, candidate=prefix + STATES_AND_SUMMARIES, arm=arm, summaries=True
                )
            )
    return found


@dataclass(frozen=True, kw_only=True)
class Fit:
    """One head fitted on one draw, and how it scored the held-out half."""

    source: str
    candidate: str
    budget: str
    seed: int
    rows: int
    columns: int
    penalty: float
    auroc: float
    seconds: float

    COLUMNS = (
        "source",
        "candidate",
        "budget",
        "seed",
        "rows",
        "columns",
        "penalty",
        "auroc",
        "seconds",
    )

    def record(self) -> dict[str, str]:
        return {
            "source": self.source,
            "candidate": self.candidate,
            "budget": self.budget,
            "seed": str(self.seed),
            "rows": str(self.rows),
            "columns": str(self.columns),
            "penalty": _number(self.penalty),
            "auroc": _number(self.auroc),
            "seconds": _number(self.seconds),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            source=record["source"],
            candidate=record["candidate"],
            budget=record["budget"],
            seed=int(record["seed"]),
            rows=int(record["rows"]),
            columns=int(record["columns"]),
            penalty=float(record["penalty"]),
            auroc=float(record["auroc"]),
            seconds=float(record["seconds"]),
        )


@dataclass(frozen=True)
class Probed:
    """Every head's fit and answers, in the pairing's shape."""

    fits: tuple[Fit, ...]
    cells: tuple[CellRow, ...]
    predictions: tuple[PredictionRow, ...]


def probe(stored: Stored, readings: Sequence[Reading] | None = None) -> Probed:
    """Every reading fitted on every stored draw and scored on the held-out half.

    A reading whose input is the same arrays as one already fitted on the same draw — the same
    encoder under two campaigns — is fitted once and filed under both.
    """
    asked = readings_of(stored) if readings is None else readings
    penalties = stored.penalties
    fits: list[Fit] = []
    cells: list[CellRow] = []
    predictions: list[PredictionRow] = []
    done: dict[tuple[str | None, bool, str, int], _Head] = {}
    for budget, seed in stored.cells():
        drawn = stored.drawn(budget, seed)
        sample_rows = [stay.row for stay in drawn]
        outcomes = np.array([stay.target for stay in drawn], dtype=np.float64)
        for reading in asked:
            digest = (
                None
                if reading.arm is None
                else stored.encoder(reading.source, reading.arm, seed).digest
            )
            key = (digest, reading.summaries, budget, seed)
            if key not in done:
                done[key] = _fitted(
                    stored, digest, reading.summaries, sample_rows, outcomes, penalties
                )
            head = done[key]
            answered = head.answered
            fit = Fit(
                source=reading.source,
                candidate=reading.candidate,
                budget=budget,
                seed=seed,
                rows=len(drawn),
                columns=head.columns,
                penalty=head.penalty,
                auroc=WindowRanking.of(answered).auroc,
                seconds=head.seconds,
            )
            print(f"{fit.source} {fit.candidate} at {budget}, seed {seed}: {fit.auroc:.4f}")
            fits.append(fit)
            cells.extend(
                CellRow(
                    campaign=reading.source,
                    purpose=RunPurpose.SELECTION.value,
                    tier=stored.tier,
                    measure=ErrorMeasure.AUROC_SHORTFALL.value,
                    candidate=reading.candidate,
                    budget=budget,
                    seed=seed,
                    unit=str(error.unit),
                    squared_error=error.squared_error,
                    windows=error.windows,
                    seconds=fit.seconds,
                )
                for error in UnitError.per_unit(answered)
            )
            predictions.extend(
                PredictionRow(
                    campaign=reading.source,
                    candidate=reading.candidate,
                    budget=budget,
                    seed=seed,
                    unit=str(row.window.unit),
                    position=row.window.position,
                    ends_at=row.window.ends_at,
                    target=row.target,
                    predicted=row.predicted,
                )
                for row in answered
            )
    return Probed(tuple(fits), tuple(cells), tuple(predictions))


@dataclass(frozen=True)
class _Head:
    """One head fitted on one draw: its answers on the held-out half, and how it was fitted."""

    answered: list[WindowPrediction]
    columns: int
    penalty: float
    seconds: float


def _fitted(
    stored: Stored,
    digest: str | None,
    summaries: bool,
    sample_rows: Sequence[int],
    outcomes: NDArray[np.float64],
    penalties: RidgePenalties,
) -> _Head:
    """One head over the states of ``digest``, the summaries or both, and its scored answers."""
    started = time.perf_counter()
    scored = stored.half(SCORED)
    scored_rows = [stay.row for stay in scored]
    parts_sample: list[NDArray[np.float64]] = []
    parts_scored: list[NDArray[np.float64]] = []
    if digest is not None:
        states = stored.states(digest)
        parts_sample.append(states[sample_rows].astype(np.float64))
        parts_scored.append(states[scored_rows].astype(np.float64))
    if summaries:
        filled_sample, filled_scored = prepared(
            stored.summaries[sample_rows], stored.summaries[scored_rows]
        )
        parts_sample.append(filled_sample)
        parts_scored.append(filled_scored)
    sample_x = np.concatenate(parts_sample, axis=1)
    answers, solution = logistic_answers(
        sample_x, outcomes, np.concatenate(parts_scored, axis=1), penalties
    )
    return _Head(
        answered=[stay.answered(float(p)) for stay, p in zip(scored, answers, strict=True)],
        columns=sample_x.shape[1],
        penalty=solution.penalty,
        seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class JointRidge:
    """Ridge solved once for many targets over the same rows, each choosing its own penalty.

    The arithmetic of ``RidgeSolution`` — columns scaled by their spread, an unpenalised
    intercept, one singular value decomposition, the penalty of least leave-one-out error, the
    smaller on a tie — run for a matrix of targets at once, so a summary of every channel costs
    one decomposition per set of stays rather than one per summary.

    The decomposition is LAPACK's divide and conquer, as ``RidgeSolution``'s is; on the rare
    matrix that method fails to converge on — states of a few thousand stays whose spread spans
    eight orders of magnitude — the same decomposition is taken by the slower one-sided method,
    which converges where the first does not, and the fit says so.

    Attributes:
        weights: The map over the states as given, one column per target.
        intercepts: What each target's map adds.
        penalties: The penalty each target chose.
        decomposition: Which LAPACK driver decomposed the states.
    """

    weights: NDArray[np.float64]
    intercepts: NDArray[np.float64]
    penalties: NDArray[np.float64]
    decomposition: str

    DIVIDE_AND_CONQUER: ClassVar[str] = "gesdd"
    ONE_SIDED: ClassVar[str] = "gesvd"

    @classmethod
    def fitted(
        cls, states: NDArray[np.floating], targets: NDArray[np.float64], penalties: RidgePenalties
    ) -> Self:
        """The maps over ``states`` answering each column of ``targets``.

        Raises:
            ValueError: If there are fewer than two rows, which leaves nothing to leave out.
        """
        x = torch.from_numpy(np.asarray(states, dtype=np.float64))
        y = torch.from_numpy(np.asarray(targets, dtype=np.float64))
        count = x.shape[0]
        if count < 2:
            raise ValueError(f"a fit chosen leave-one-out needs at least two rows, got {count}")
        spread = x.std(dim=0, correction=0)
        scale = torch.where(spread > 0.0, spread, torch.ones_like(spread))
        scaled = x / scale
        x_mean, y_mean = scaled.mean(dim=0), y.mean(dim=0)
        centred, residual = scaled - x_mean, y - y_mean
        left, singular, right_t, decomposition = cls._decomposed(centred)
        projected = left.T @ residual
        squared = singular**2
        best = torch.full((y.shape[1],), torch.inf, dtype=torch.float64)
        coefficients = torch.zeros((x.shape[1], y.shape[1]), dtype=torch.float64)
        chosen = torch.zeros(y.shape[1], dtype=torch.float64)
        for penalty in penalties.values:
            shrink = squared / (squared + penalty)
            fitted = left @ (shrink.unsqueeze(1) * projected)
            leverage = (left**2 * shrink).sum(dim=1) + 1.0 / count
            errors = (((residual - fitted) / (1.0 - leverage).unsqueeze(1)) ** 2).mean(dim=0)
            better = errors < best
            coefficient = right_t.T @ ((singular / (squared + penalty)).unsqueeze(1) * projected)
            best = torch.where(better, errors, best)
            coefficients[:, better] = coefficient[:, better]
            chosen[better] = penalty
        return cls(
            weights=(coefficients / scale.unsqueeze(1)).numpy(),
            intercepts=(y_mean - (x_mean.unsqueeze(1) * coefficients).sum(dim=0)).numpy(),
            penalties=chosen.numpy(),
            decomposition=decomposition,
        )

    @classmethod
    def _decomposed(
        cls, centred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        try:
            left, singular, right_t = torch.linalg.svd(centred, full_matrices=False)
        # The linear-algebra error torch raises here is a RuntimeError its stubs do not name.
        except RuntimeError:
            u, s, vt = scipy.linalg.svd(
                centred.numpy(), full_matrices=False, lapack_driver=cls.ONE_SIDED
            )
            return torch.from_numpy(u), torch.from_numpy(s), torch.from_numpy(vt), cls.ONE_SIDED
        return left, singular, right_t, cls.DIVIDE_AND_CONQUER

    def answered(self, states: NDArray[np.floating]) -> NDArray[np.float64]:
        return np.asarray(states, dtype=np.float64) @ self.weights + self.intercepts


@dataclass(frozen=True, kw_only=True)
class Decoded:
    """How well one encoder's states recover one summary of one channel on the held-out half.

    Attributes:
        digest: The encoder, by the digest its states are stored under.
        channel_id: The channel.
        statistic: The summary of it.
        family: What the summary says about the channel.
        learnt: Stays of the learnt half the map was fitted on, where the summary is defined.
        scored: Stays of the held-out half it was scored on.
        penalty: The penalty the map chose.
        r2: One less the squared error over the variance around the held-out half's mean, the
            map's answers held within the range of the values it was fitted on.
        decomposition: Which LAPACK driver decomposed the states the map was fitted on.
    """

    digest: str
    channel_id: int
    statistic: str
    family: str
    learnt: int
    scored: int
    penalty: float
    r2: float
    decomposition: str

    COLUMNS = (
        "digest",
        "channel_id",
        "statistic",
        "family",
        "learnt",
        "scored",
        "penalty",
        "r2",
        "decomposition",
    )

    def record(self) -> dict[str, str]:
        return {
            "digest": self.digest,
            "channel_id": str(self.channel_id),
            "statistic": self.statistic,
            "family": self.family,
            "learnt": str(self.learnt),
            "scored": str(self.scored),
            "penalty": _number(self.penalty),
            "r2": _number(self.r2),
            "decomposition": self.decomposition,
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            digest=record["digest"],
            channel_id=int(record["channel_id"]),
            statistic=record["statistic"],
            family=record["family"],
            learnt=int(record["learnt"]),
            scored=int(record["scored"]),
            penalty=float(record["penalty"]),
            r2=float(record["r2"]),
            decomposition=record["decomposition"],
        )


def decoded_targets(stored: Stored, channels: Sequence[int] | None = None) -> list[int]:
    """The columns of the summaries a map is fitted to: every family's, on enough stays.

    A column is left out where its family is none, where either half defines it on fewer than
    ``MINIMUM_STAYS`` stays, or where it does not vary over the learnt half.
    """
    learnt = [stay.row for stay in stored.half(LEARNT)]
    scored = [stay.row for stay in stored.half(SCORED)]
    summaries = stored.summaries
    kept = []
    for column, (channel, statistic) in enumerate(stored.summary_columns()):
        if channels is not None and channel.channel_id not in channels:
            continue
        if family_of(statistic, timeless=channel.timeless) is None:
            continue
        on_learnt = summaries[learnt, column]
        on_learnt = on_learnt[np.isfinite(on_learnt)]
        on_scored = summaries[scored, column]
        on_scored = on_scored[np.isfinite(on_scored)]
        if min(len(on_learnt), len(on_scored)) < MINIMUM_STAYS:
            continue
        if on_learnt.std() == 0.0 or on_scored.std() == 0.0:
            continue
        kept.append(column)
    return kept


def recovered(
    learnt_states: NDArray[np.floating],
    learnt_values: NDArray[np.float64],
    scored_states: NDArray[np.floating],
    scored_values: NDArray[np.float64],
    penalties: RidgePenalties,
) -> tuple[JointRidge, NDArray[np.float64]]:
    """Maps from states to each column of values, and the share of variance each recovers.

    Fitted on ``learnt_values``, defined on every row, and scored on the rows of
    ``scored_values`` where each column is defined, by one less the squared error over the
    variance around those rows' mean. A map answers within the range of the values it was
    fitted on: a held-out row whose state lies far from every learnt one would otherwise be
    answered by extrapolation, and its one error could outweigh every other row's.
    """
    ridge = JointRidge.fitted(learnt_states, learnt_values, penalties)
    answers = np.clip(
        ridge.answered(scored_states), learnt_values.min(axis=0), learnt_values.max(axis=0)
    )
    shares = np.empty(learnt_values.shape[1], dtype=np.float64)
    for index in range(learnt_values.shape[1]):
        truth = scored_values[:, index]
        defined = np.isfinite(truth)
        error = truth[defined] - answers[defined, index]
        around = truth[defined] - truth[defined].mean()
        shares[index] = 1.0 - (error**2).sum() / (around**2).sum()
    return ridge, shares


def decode(stored: Stored, columns: Sequence[int]) -> list[Decoded]:
    """Every stored encoder's states mapped to each of ``columns``, fitted and scored by halves.

    Columns defined on the same stays share one decomposition; each is scored as ``recovered``
    scores it, which a channel read once under an encoder at its initialisation needs.
    """
    learnt = np.array([stay.row for stay in stored.half(LEARNT)])
    scored = np.array([stay.row for stay in stored.half(SCORED)])
    summaries = stored.summaries
    described = stored.summary_columns()
    penalties = stored.penalties
    groups: dict[bytes, list[int]] = {}
    for column in columns:
        groups.setdefault(np.isfinite(summaries[learnt, column]).tobytes(), []).append(column)
    found = []
    for digest in dict.fromkeys(encoder.digest for encoder in stored.encoders):
        started = time.perf_counter()
        states = stored.states(digest)
        for mask, members in groups.items():
            rows = learnt[np.frombuffer(mask, dtype=bool)]
            truths = summaries[np.ix_(scored, members)]
            ridge, shares = recovered(
                states[rows], summaries[np.ix_(rows, members)], states[scored], truths, penalties
            )
            for index, column in enumerate(members):
                channel, statistic = described[column]
                family = family_of(statistic, timeless=channel.timeless)
                if family is None:
                    raise ValueError(f"column {column} has no family to be decoded under")
                found.append(
                    Decoded(
                        digest=digest,
                        channel_id=channel.channel_id,
                        statistic=statistic,
                        family=family,
                        learnt=len(rows),
                        scored=int(np.isfinite(truths[:, index]).sum()),
                        penalty=float(ridge.penalties[index]),
                        r2=float(shares[index]),
                        decomposition=ridge.decomposition,
                    )
                )
        print(
            f"decoded {len(columns)} summaries off {digest[:12]} in "
            f"{time.perf_counter() - started:.0f} s",
            flush=True,
        )
    return found


def write_probed(directory: Path, probed: Probed) -> None:
    write_rows(directory / FITS, Fit.COLUMNS, (fit.record() for fit in probed.fits))
    write_cells(directory / CELLS, probed.cells)
    write_predictions(directory / PREDICTIONS, probed.predictions)


def write_decoded(directory: Path, decoded: Sequence[Decoded]) -> None:
    write_rows(directory / DECODING, Decoded.COLUMNS, (row.record() for row in decoded))


def spread(values: Sequence[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{fmean(values):.3f} [{min(values):.3f}; {max(values):.3f}]"


def render_fidelity(stored: Stored, fits: Sequence[Fit], reference: Path) -> tuple[str, float]:
    """The reading over states alone set beside the registered probe cells it stands in for.

    Returns:
        The table, and the largest difference in area over every cell compared.
    """
    campaigns = stored.conditions["campaigns"]
    if not isinstance(campaigns, dict):
        raise ValueError("the conditions name no campaigns")
    registered = {
        (row.campaign, row.candidate, row.budget, row.seed): row.score
        for row in read_scores(reference / SCORES)
    }
    rows = []
    largest = 0.0
    for source in stored.sources:
        for arm in (PRETRAINED, UNTRAINED):
            candidate = (UNTRAINED_PREFIX if arm == UNTRAINED else "") + STATES_ALONE
            for budget in dict.fromkeys(fit.budget for fit in fits):
                own = {
                    fit.seed: fit.auroc
                    for fit in fits
                    if (fit.source, fit.candidate, fit.budget) == (source, candidate, budget)
                }
                differences = []
                for seed, area in own.items():
                    key = (str(campaigns[source]), REGISTERED[arm], budget, seed)
                    if key not in registered:
                        raise ValueError(f"the reference holds no cell {key}")
                    differences.append(abs(area - registered[key]))
                largest = max([largest, *differences])
                rows.append(
                    [
                        source,
                        REGISTERED[arm],
                        budget,
                        str(len(differences)),
                        f"{max(differences):.2e}",
                    ]
                )
    header = ["source", "registered cell", "budget", "seeds", "largest |difference| in area"]
    return table(header, rows), largest


def render_areas(fits: Sequence[Fit]) -> str:
    """Area per reading and budget: the mean over seeds per source, spread over the sources."""
    budgets = list(dict.fromkeys(fit.budget for fit in fits))
    by_candidate: dict[str, dict[str, dict[str, list[float]]]] = {}
    for fit in fits:
        by_candidate.setdefault(fit.candidate, {}).setdefault(fit.budget, {}).setdefault(
            fit.source, []
        ).append(fit.auroc)
    rows = []
    for candidate, per_budget in by_candidate.items():
        cells = []
        for budget in budgets:
            sources = per_budget.get(budget, {})
            cells.append(spread([fmean(values) for values in sources.values()]))
        rows.append([candidate, *cells])
    return table(["reading", *budgets], rows)


def render(
    stored: Stored, fits: Sequence[Fit], decoded: Sequence[Decoded], reference: Path | None
) -> str:
    """The tables a note pastes: fidelity, area per reading and budget, recovery per family."""
    sections = [
        f"Commit `{stored.conditions['commit']}`, {stored.conditions['device']} "
        f"(torch {stored.conditions['torch']}), tier {stored.tier}; "
        f"{len(stored.half(LEARNT))} stays learnt, {len(stored.half(SCORED))} scored."
    ]
    if reference is not None:
        fidelity, largest = render_fidelity(stored, fits, reference)
        sections.append(
            f"### Fidelity to the registered probe cells (largest {largest:.2e})\n\n{fidelity}"
        )
    sections.append(
        "### Area under the ROC curve, mean over seeds (mean over sources [range])\n\n"
        + render_areas(fits)
    )
    arms = {encoder.digest: encoder.arm for encoder in stored.encoders}
    medians: dict[str, dict[str, list[float]]] = {}
    shares: dict[str, list[float]] = {}
    for digest in dict.fromkeys(row.digest for row in decoded):
        own = [row for row in decoded if row.digest == digest]
        arm = arms[digest]
        for family in FAMILIES:
            values = [row.r2 for row in own if row.family == family]
            if values:
                medians.setdefault(arm, {}).setdefault(family, []).append(median(values))
        shares.setdefault(arm, []).append(fmean(row.r2 >= CARRIED for row in own))
    family_rows = []
    for family in FAMILIES:
        pretrained = medians.get(PRETRAINED, {}).get(family, [])
        untrained = medians.get(UNTRAINED, {}).get(family, [])
        difference = (
            f"{fmean(pretrained) - fmean(untrained):+.3f}" if pretrained and untrained else "—"
        )
        count = len({(row.channel_id, row.statistic) for row in decoded if row.family == family})
        family_rows.append([family, str(count), spread(pretrained), spread(untrained), difference])
    family_rows.append(
        [
            f"share ≥ {CARRIED}",
            str(len({(row.channel_id, row.statistic) for row in decoded})),
            spread(shares.get(PRETRAINED, [])),
            spread(shares.get(UNTRAINED, [])),
            "—",
        ]
    )
    sections.append(
        "### Median R² per family of summaries on the held-out half "
        "(mean over encoders [range])\n\n"
        + table(
            ["family", "summaries", "pretrained", "at initialisation", "difference"], family_rows
        )
    )
    return "\n\n".join(sections) + "\n"


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="where the storing step left its arrays")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--report-only", action="store_true", help="render the stored fits")
    stage.add_argument(
        "--decode-only",
        action="store_true",
        help="keep the heads already fitted and map the summaries again",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        metavar="DIR",
        help="an export of the probe campaigns by campaign_pairs_report.py, to check against",
    )
    parser.add_argument(
        "--decode-channels",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="map only these channels' summaries, as a trial of the arithmetic does",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    directory: Path = arguments.directory
    stored = Stored.read(directory)
    # Each stage is written as soon as it ends, so a later one that fails costs only itself.
    if arguments.report_only or arguments.decode_only:
        fits = [Fit.parse(row) for row in read_rows(directory / FITS)]
    else:
        probed = probe(stored)
        write_probed(directory, probed)
        fits = list(probed.fits)
    if arguments.report_only:
        decoded = [Decoded.parse(row) for row in read_rows(directory / DECODING)]
    else:
        decoded = decode(stored, decoded_targets(stored, arguments.decode_channels))
        write_decoded(directory, decoded)
    report = render(stored, fits, decoded, arguments.reference)
    (directory / REPORT).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
