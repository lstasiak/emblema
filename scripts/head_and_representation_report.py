"""Where the network loses to the trees: the head, the representation, or the task itself.

Three diagnoses, all on the validation side and all off representations stored once by
``frozen_representations.py``, so nothing here reaches an accelerator. Every probe is a fitter
over an input: ridge or boosted trees over the hand-made statistics per channel, over the
statistics' last values alone, or over the frozen backbone's states pooled several ways — the
mean over the window, the mean over its tail, one state per channel — and, as a control, over
the states of an encoder that was never trained. Every probe learns from the same drawn labels a
campaign would give a candidate at that budget and seed, answers the same validation windows, and
is paired with every other over the same units by the bootstrap every verdict uses.

The predictions and the fits are written to CSV first; the comparisons and the tables are made
from those files, so a table can be reshaped without fitting anything again.

    uv run scripts/head_and_representation_report.py DIR
    uv run scripts/head_and_representation_report.py DIR --report-only
"""

import argparse
import csv
import json
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Self
from uuid import UUID

import numpy as np
import xgboost
from numpy.typing import NDArray
from sklearn.linear_model import RidgeCV
from threadpoolctl import threadpool_limits

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.contracts.identifiers import TaskId
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.label_budget import LabelBudget
from emblema.evaluation.domain.labels.label_sample import LabelSample
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.target_bins import TargetBins
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.unit_error import UnitError
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.statistics.error_over_repeats import ErrorOverRepeats
from emblema.evaluation.domain.statistics.paired_unit_bootstrap import PairedUnitBootstrap
from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors
from scripts.reporting import dated_heading, table

# What the embedding step stores and this reads: the arrays, the windows they are rows of, and
# the conditions they were made under.
REPRESENTATIONS = "representations.npz"
WINDOWS = "windows.csv"
CONDITIONS = "conditions.json"
HAND_FEATURES = "hand"
HAND_NAMES = "hand_names"
CHANNELS_READ = "channels_read"
TUNING, VALIDATION = "tuning", "validation"

# The two encoders the states come from, and the ways a window's states are pooled into one row.
PRETRAINED, FRESH = "pretrained", "fresh"
MEAN, CHANNEL_MEAN, CHANNEL_LAST = "mean", "channel_mean", "channel_last"
TAIL_SHARES = (0.02, 0.1, 0.2)

# What this writes: one row per answered window, one per fit, one per paired comparison.
PREDICTIONS, FITS, COMPARISONS = "predictions.csv", "fits.csv", "comparisons.csv"

# The inputs a probe may read. The hand-made statistics are the trees' own reading; the last
# values are the one column of them that says where each channel stands as the window ends.
HAND, LAST = "hand", "last"
LAST_SUFFIX = "_last"
FRESH_PREFIX = "fresh_"

RIDGE, TREES = "ridge", "trees"
BUDGETS = (50, 200)
SEEDS = (1, 2, 3)
# The penalties the ridge of the convolution baseline chooses among, by leave-one-out error.
PENALTIES = (0.001, 0.00464, 0.0215, 0.1, 0.464, 2.15, 10.0, 46.4, 215.0, 1000.0)
# The trees at each budget as the declared selection chose them for the trees per channel,
# recorded in ``campaigns/baselines-fd001.toml``: the library's depth at 50, three levels at 200.
# A budget the selection never ran at gets the library's depth, as an untuned baseline would.
LIBRARY_DEPTH = 6
DEPTH_AT = {50: LIBRARY_DEPTH, 200: 3}
THREADS = 4
# A column whose spread is this small a share of its largest value is constant up to rounding:
# a gap between readings on a regular corpus is one number written a thousand times.
CONSTANT_TOLERANCE = 1e-9
RESAMPLES = 10_000
TASK = TaskId(UUID(int=0))


def tail_name(share: float) -> str:
    """How a tail pooling is named, by the share of the window it keeps, in percent."""
    return f"tail_{round(share * 100)}"


def pooled_key(encoder: str, pooling: str) -> str:
    """The array a pooling of an encoder's states is stored under."""
    return f"{encoder}__{pooling}"


@dataclass(frozen=True, kw_only=True)
class Probe:
    """One fitter over one input, named by both.

    Attributes:
        input: What the fitter reads a window as.
        fitter: What is fitted on that reading.
    """

    input: str
    fitter: str

    @property
    def name(self) -> str:
        return f"{self.input}/{self.fitter}"

    @classmethod
    def parse(cls, name: str) -> Self:
        stated, slash, fitter = name.partition("/")
        if not slash or fitter not in (RIDGE, TREES):
            raise ValueError(f"{name!r} is not an input and a fitter")
        return cls(input=stated, fitter=fitter)


REFERENCE = Probe(input=HAND, fitter=TREES)
PROBES = (
    REFERENCE,
    Probe(input=HAND, fitter=RIDGE),
    Probe(input=LAST, fitter=RIDGE),
    Probe(input=LAST, fitter=TREES),
    Probe(input=MEAN, fitter=RIDGE),
    Probe(input=MEAN, fitter=TREES),
    Probe(input=tail_name(0.02), fitter=RIDGE),
    Probe(input=tail_name(0.1), fitter=RIDGE),
    Probe(input=tail_name(0.1), fitter=TREES),
    Probe(input=tail_name(0.2), fitter=RIDGE),
    Probe(input=CHANNEL_MEAN, fitter=RIDGE),
    Probe(input=CHANNEL_MEAN, fitter=TREES),
    Probe(input=CHANNEL_LAST, fitter=RIDGE),
    Probe(input=CHANNEL_LAST, fitter=TREES),
    Probe(input=FRESH_PREFIX + MEAN, fitter=RIDGE),
    Probe(input=FRESH_PREFIX + MEAN, fitter=TREES),
    Probe(input=FRESH_PREFIX + CHANNEL_LAST, fitter=RIDGE),
    Probe(input=FRESH_PREFIX + CHANNEL_LAST, fitter=TREES),
)
# The pairs the diagnoses read, beside every probe against the reference: the head's poolings
# against the mean under one linear head; the pretrained states against the untrained ones under
# one fitter; the last values against the trees' whole reading.
CONTRASTS = (
    (Probe(input=tail_name(0.1), fitter=RIDGE), Probe(input=MEAN, fitter=RIDGE)),
    (Probe(input=CHANNEL_MEAN, fitter=RIDGE), Probe(input=MEAN, fitter=RIDGE)),
    (Probe(input=CHANNEL_LAST, fitter=RIDGE), Probe(input=MEAN, fitter=RIDGE)),
    (Probe(input=MEAN, fitter=TREES), Probe(input=FRESH_PREFIX + MEAN, fitter=TREES)),
    (
        Probe(input=CHANNEL_LAST, fitter=TREES),
        Probe(input=FRESH_PREFIX + CHANNEL_LAST, fitter=TREES),
    ),
    (Probe(input=CHANNEL_LAST, fitter=RIDGE), Probe(input=LAST, fitter=RIDGE)),
)


@dataclass(frozen=True, kw_only=True)
class StoredWindow:
    """One row of the stored arrays: which window it is, which side, and its label.

    Attributes:
        row: Index of the window in every stored array.
        side: Whether the window is on the tuning side or the validation side.
        unit: Unit the window was cut from.
        position: Index the published block holds the window at.
        ends_at: Last moment the window covers.
        target: The label the task's scheme reads for it, in the task's unit.
    """

    row: int
    side: str
    unit: str
    position: int
    ends_at: float
    target: float

    COLUMNS = ("row", "side", "unit", "position", "ends_at", "target")

    def record(self) -> dict[str, str]:
        return {
            "row": str(self.row),
            "side": self.side,
            "unit": self.unit,
            "position": str(self.position),
            "ends_at": repr(self.ends_at),
            "target": repr(self.target),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            row=int(record["row"]),
            side=record["side"],
            unit=record["unit"],
            position=int(record["position"]),
            ends_at=float(record["ends_at"]),
            target=float(record["target"]),
        )

    def task_window(self) -> TaskWindow:
        return TaskWindow(unit=UnitKey(self.unit), position=self.position, ends_at=self.ends_at)

    def labelled(self) -> LabelledWindow:
        return LabelledWindow(window=self.task_window(), target=self.target)


def save_arrays(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> None:
    """``arrays`` in NumPy's own archive format, one member per name, nothing pickled."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w") as member:
                np.lib.format.write_array(member, np.ascontiguousarray(array), allow_pickle=False)


def write_windows(windows: Sequence[StoredWindow], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / WINDOWS).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=StoredWindow.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(window.record() for window in windows)
    return directory / WINDOWS


@dataclass(frozen=True)
class Stored:
    """What the embedding step left in a directory, read back as the probes' inputs.

    Attributes:
        directory: Where the files are.
        windows: The rows of every array, in row order.
        conditions: What the embedding step recorded about its run.
        arrays: The stored arrays by name.
    """

    directory: Path
    windows: tuple[StoredWindow, ...]
    conditions: dict[str, object]
    arrays: Mapping[str, NDArray[np.generic]]

    @classmethod
    def read(cls, directory: Path) -> Self:
        """The representations stored under ``directory``.

        Raises:
            SystemExit: If nothing is stored there.
        """
        for name in (REPRESENTATIONS, WINDOWS, CONDITIONS):
            if not (directory / name).is_file():
                raise SystemExit(f"{directory} holds no {name}; run frozen_representations first")
        with (directory / WINDOWS).open(newline="", encoding="utf-8") as handle:
            windows = tuple(StoredWindow.parse(row) for row in csv.DictReader(handle))
        with np.load(directory / REPRESENTATIONS, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        conditions: dict[str, object] = json.loads(
            (directory / CONDITIONS).read_text(encoding="utf-8")
        )
        return cls(directory=directory, windows=windows, conditions=conditions, arrays=arrays)

    @property
    def tuning(self) -> tuple[StoredWindow, ...]:
        return tuple(window for window in self.windows if window.side == TUNING)

    @property
    def validation(self) -> tuple[StoredWindow, ...]:
        return tuple(window for window in self.windows if window.side == VALIDATION)

    @property
    def strata(self) -> TargetBins:
        return TargetBins(int(str(self.conditions["strata"])))

    @property
    def scale(self) -> float:
        return float(str(self.conditions["target_scale"]))

    def input(self, name: str) -> NDArray[np.float64]:
        """The rows of ``name``, one per stored window, as the fitters read them.

        Raises:
            KeyError: If nothing stored is called that.
        """
        hand = np.asarray(self.arrays[HAND_FEATURES], dtype=np.float64)
        if name == HAND:
            return hand
        if name == LAST:
            names = [str(column) for column in self.arrays[HAND_NAMES]]
            kept = [index for index, column in enumerate(names) if column.endswith(LAST_SUFFIX)]
            return hand[:, kept]
        if name.startswith(FRESH_PREFIX):
            key = pooled_key(FRESH, name.removeprefix(FRESH_PREFIX))
        else:
            key = pooled_key(PRETRAINED, name)
        return np.asarray(self.arrays[key], dtype=np.float64)

    def sample(self, budget: int, seed: int) -> LabelSample:
        """The draw at ``budget`` under ``seed``, made as a campaign makes it.

        The draw ranks each window by its unit and position under the seed, so the same windows
        come out here as came out for every candidate of a campaign over this task.
        """
        pool = [window.labelled() for window in self.tuning]
        return LabelSample.drawn(TASK, pool, LabelBudget.of(budget), self.strata, seed)

    def rows_of(self, windows: Sequence[LabelledWindow]) -> list[int]:
        """The stored rows of ``windows``, in the order given."""
        by_window = {(window.unit, window.position): window.row for window in self.windows}
        return [
            by_window[str(labelled.window.unit), labelled.window.position] for labelled in windows
        ]


@dataclass(frozen=True, kw_only=True)
class Prediction:
    """What one probe said about one validation window under one budget and seed."""

    probe: str
    budget: int
    seed: int
    unit: str
    position: int
    target: float
    predicted: float

    COLUMNS = ("probe", "budget", "seed", "unit", "position", "target", "predicted")

    def record(self) -> dict[str, str]:
        return {
            "probe": self.probe,
            "budget": str(self.budget),
            "seed": str(self.seed),
            "unit": self.unit,
            "position": str(self.position),
            "target": repr(self.target),
            "predicted": repr(self.predicted),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            probe=record["probe"],
            budget=int(record["budget"]),
            seed=int(record["seed"]),
            unit=record["unit"],
            position=int(record["position"]),
            target=float(record["target"]),
            predicted=float(record["predicted"]),
        )

    def window_prediction(self) -> WindowPrediction:
        return WindowPrediction(
            window=TaskWindow(unit=UnitKey(self.unit), position=self.position, ends_at=0.0),
            target=self.target,
            predicted=self.predicted,
        )


@dataclass(frozen=True, kw_only=True)
class Fit:
    """One probe fitted under one budget and seed: what it read, what it chose, what it scored.

    Attributes:
        probe: The probe.
        budget: Labelled windows it learnt from.
        seed: Seed of the draw and of the fit.
        rows: Labelled windows the fit saw.
        columns: Columns it read after the ones it cannot read were dropped.
        chosen: The penalty a ridge chose, or the depth the trees were grown to.
        rmse: Error on the validation windows, in the task's unit.
        seconds: Wall time of the fit and the answers.
    """

    probe: str
    budget: int
    seed: int
    rows: int
    columns: int
    chosen: float
    rmse: float
    seconds: float

    COLUMNS = ("probe", "budget", "seed", "rows", "columns", "chosen", "rmse", "seconds")

    def record(self) -> dict[str, str]:
        return {
            "probe": self.probe,
            "budget": str(self.budget),
            "seed": str(self.seed),
            "rows": str(self.rows),
            "columns": str(self.columns),
            "chosen": repr(self.chosen),
            "rmse": repr(self.rmse),
            "seconds": repr(self.seconds),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            probe=record["probe"],
            budget=int(record["budget"]),
            seed=int(record["seed"]),
            rows=int(record["rows"]),
            columns=int(record["columns"]),
            chosen=float(record["chosen"]),
            rmse=float(record["rmse"]),
            seconds=float(record["seconds"]),
        )


@dataclass(frozen=True, kw_only=True)
class Comparison:
    """One probe against another at one budget, paired over the validation units.

    Attributes:
        candidate: The probe compared.
        rival: The probe it is compared against.
        budget: Labelled windows both learnt from.
        seeds: Repeats pooled on both sides.
        rival_rmse: The rival's error over the pooled repeats.
        candidate_rmse: The candidate's error over the pooled repeats.
        rival_sd: Spread of the rival's error between repeats.
        candidate_sd: Spread of the candidate's error between repeats.
        reduction: How much lower the candidate's error is than the rival's.
        relative_reduction: The reduction as a share of the rival's error.
        low: Lower end of the paired interval over units.
        high: Upper end of the paired interval over units.
        p_value: Two-sided p-value of the reduction under the resampling.
    """

    candidate: str
    rival: str
    budget: int
    seeds: int
    rival_rmse: float
    candidate_rmse: float
    rival_sd: float
    candidate_sd: float
    reduction: float
    relative_reduction: float
    low: float
    high: float
    p_value: float

    COLUMNS = (
        "candidate",
        "rival",
        "budget",
        "seeds",
        "rival_rmse",
        "candidate_rmse",
        "rival_sd",
        "candidate_sd",
        "reduction",
        "relative_reduction",
        "low",
        "high",
        "p_value",
    )

    def record(self) -> dict[str, str]:
        return {
            "candidate": self.candidate,
            "rival": self.rival,
            "budget": str(self.budget),
            "seeds": str(self.seeds),
            "rival_rmse": repr(self.rival_rmse),
            "candidate_rmse": repr(self.candidate_rmse),
            "rival_sd": repr(self.rival_sd),
            "candidate_sd": repr(self.candidate_sd),
            "reduction": repr(self.reduction),
            "relative_reduction": repr(self.relative_reduction),
            "low": repr(self.low),
            "high": repr(self.high),
            "p_value": repr(self.p_value),
        }

    @classmethod
    def parse(cls, record: Mapping[str, str]) -> Self:
        return cls(
            candidate=record["candidate"],
            rival=record["rival"],
            budget=int(record["budget"]),
            seeds=int(record["seeds"]),
            rival_rmse=float(record["rival_rmse"]),
            candidate_rmse=float(record["candidate_rmse"]),
            rival_sd=float(record["rival_sd"]),
            candidate_sd=float(record["candidate_sd"]),
            reduction=float(record["reduction"]),
            relative_reduction=float(record["relative_reduction"]),
            low=float(record["low"]),
            high=float(record["high"]),
            p_value=float(record["p_value"]),
        )

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0


class Fitted:
    """The answers of one fit over the scored rows, and what the fit chose and read."""

    def __init__(self, predicted: NDArray[np.float64], *, chosen: float, columns: int) -> None:
        self.predicted = predicted
        self.chosen = chosen
        self.columns = columns


def ridge(
    rows: NDArray[np.float64],
    targets: NDArray[np.float64],
    scored: NDArray[np.float64],
    *,
    penalties: Sequence[float] = PENALTIES,
) -> Fitted:
    """A linear map with an intercept, its penalty chosen by leave-one-out error.

    Scaled but not centred, as the convolution baseline's ridge is: the intercept absorbs the
    means. A column a ridge cannot read — one that is missing in a fitted row, or one that does
    not vary beyond the rounding of its own values — is dropped; a value missing in a scored row
    is filled with the fitted column's mean, which is the value that moves the answer least.
    """
    kept = ~np.isnan(rows).any(axis=0)
    present = np.where(kept, rows, 0.0)
    spread = present.std(axis=0)
    kept &= spread > CONSTANT_TOLERANCE * np.maximum(np.abs(present).max(axis=0), 1.0)
    fitted = rows[:, kept]
    scale = fitted.std(axis=0)
    means = fitted.mean(axis=0)
    answered = scored[:, kept]
    answered = np.where(np.isnan(answered), means, answered)
    with threadpool_limits(limits=THREADS):
        model = RidgeCV(alphas=list(penalties)).fit(fitted / scale, targets)
        predicted = np.asarray(model.predict(answered / scale), dtype=np.float64)
    return Fitted(predicted, chosen=float(model.alpha_), columns=int(kept.sum()))


def trees(
    rows: NDArray[np.float64],
    targets: NDArray[np.float64],
    scored: NDArray[np.float64],
    *,
    boosting: GradientBoostingSpec,
    seed: int,
) -> Fitted:
    """Gradient-boosted trees grown as the classical runtime grows them, on the same knobs.

    A missing value is read by the trees as the runtime reads it: as a direction learnt per
    split, so the rows go in as they are.
    """
    model = xgboost.XGBRegressor(
        n_estimators=boosting.rounds,
        max_depth=boosting.max_depth,
        learning_rate=boosting.learning_rate,
        subsample=boosting.row_share,
        colsample_bytree=boosting.feature_share,
        min_child_weight=boosting.min_leaf_weight,
        reg_lambda=boosting.l2_penalty,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        n_jobs=boosting.threads,
    )
    model.fit(rows, targets)
    predicted = np.asarray(model.predict(scored), dtype=np.float64)
    return Fitted(predicted, chosen=float(boosting.max_depth), columns=int(rows.shape[1]))


def boosting_at(budget: int) -> GradientBoostingSpec:
    """The trees' knobs at ``budget``: the library's, at the depth the selection chose."""
    return GradientBoostingSpec(
        rounds=100,
        max_depth=DEPTH_AT.get(budget, LIBRARY_DEPTH),
        learning_rate=0.3,
        row_share=1.0,
        feature_share=1.0,
        min_leaf_weight=1.0,
        l2_penalty=1.0,
        threads=THREADS,
    )


def fit_probe(stored: Stored, probe: Probe, budget: int, seed: int) -> tuple[Fit, list[Prediction]]:
    """One probe fitted on the draw at ``budget`` under ``seed``, answering the validation side.

    Targets are learnt in units of the task's scale, as every candidate of a campaign learns
    them, and the answers are multiplied back.
    """
    sample = stored.sample(budget, seed)
    matrix = stored.input(probe.input)
    validation = stored.validation
    fitted_rows = matrix[stored.rows_of(sample.windows)]
    scored_rows = matrix[[window.row for window in validation]]
    targets = np.array(
        [labelled.target / stored.scale for labelled in sample.windows], dtype=np.float64
    )
    started = time.perf_counter()
    if probe.fitter == RIDGE:
        fitted = ridge(fitted_rows, targets, scored_rows)
    else:
        fitted = trees(fitted_rows, targets, scored_rows, boosting=boosting_at(budget), seed=seed)
    seconds = time.perf_counter() - started
    predictions = [
        Prediction(
            probe=probe.name,
            budget=budget,
            seed=seed,
            unit=window.unit,
            position=window.position,
            target=window.target,
            predicted=float(answer) * stored.scale,
        )
        for window, answer in zip(validation, fitted.predicted.tolist(), strict=True)
    ]
    errors = np.array([row.predicted - row.target for row in predictions])
    fit = Fit(
        probe=probe.name,
        budget=budget,
        seed=seed,
        rows=len(sample.windows),
        columns=fitted.columns,
        chosen=fitted.chosen,
        rmse=float(np.sqrt(np.mean(errors**2))),
        seconds=seconds,
    )
    return fit, predictions


@dataclass(frozen=True)
class Measured:
    """Everything the fits produced, and the comparisons read from them."""

    fits: tuple[Fit, ...]
    predictions: tuple[Prediction, ...]
    comparisons: tuple[Comparison, ...]


def measure(
    stored: Stored,
    *,
    probes: Sequence[Probe] = PROBES,
    budgets: Sequence[int] = BUDGETS,
    seeds: Sequence[int] = SEEDS,
    resamples: int = RESAMPLES,
) -> Measured:
    """Every probe at every budget under every seed, then every declared pair of them."""
    fits: list[Fit] = []
    predictions: list[Prediction] = []
    for budget in budgets:
        for seed in seeds:
            for probe in probes:
                fit, answered = fit_probe(stored, probe, budget, seed)
                print(f"{fit.probe} at {budget} under seed {seed}: {fit.rmse:.2f}", flush=True)
                fits.append(fit)
                predictions.extend(answered)
    comparisons = compare(predictions, probes=probes, resamples=resamples)
    return Measured(tuple(fits), tuple(predictions), tuple(comparisons))


def compare(
    predictions: Sequence[Prediction],
    *,
    probes: Sequence[Probe] = PROBES,
    resamples: int = RESAMPLES,
) -> tuple[Comparison, ...]:
    """Every probe against the reference and every declared contrast, at every budget.

    Repeats pool per unit before the pair is made, so a comparison is one candidate over all of
    its answers against one rival over all of its own, on the same units.
    """
    by_cell: dict[tuple[str, int, int], list[WindowPrediction]] = {}
    for row in predictions:
        by_cell.setdefault((row.probe, row.budget, row.seed), []).append(row.window_prediction())
    budgets = sorted({budget for _, budget, _ in by_cell})
    pairs = [(probe, REFERENCE) for probe in probes if probe != REFERENCE]
    pairs += [pair for pair in CONTRASTS if pair not in pairs]
    bootstrap = PairedUnitBootstrap(resamples=resamples, seed=1)
    found = []
    for budget in budgets:
        for candidate, rival in pairs:
            seeds = sorted(
                seed
                for probe, held, seed in by_cell
                if probe == candidate.name
                and held == budget
                and (rival.name, budget, seed) in by_cell
            )
            if not seeds:
                continue
            rival_runs = [by_cell[rival.name, budget, seed] for seed in seeds]
            candidate_runs = [by_cell[candidate.name, budget, seed] for seed in seeds]
            paired = PairedUnitErrors.pooled(
                control=[UnitError.per_unit(run) for run in rival_runs],
                candidate=[UnitError.per_unit(run) for run in candidate_runs],
            )
            rival_error = ErrorOverRepeats.of(
                paired.rmse_control, [_rmse(run) for run in rival_runs]
            )
            candidate_error = ErrorOverRepeats.of(
                paired.rmse_candidate, [_rmse(run) for run in candidate_runs]
            )
            difference = bootstrap.compare(paired)
            found.append(
                Comparison(
                    candidate=candidate.name,
                    rival=rival.name,
                    budget=budget,
                    seeds=len(seeds),
                    rival_rmse=rival_error.pooled,
                    candidate_rmse=candidate_error.pooled,
                    rival_sd=rival_error.spread,
                    candidate_sd=candidate_error.spread,
                    reduction=difference.reduction,
                    relative_reduction=difference.relative_reduction,
                    low=difference.interval.low,
                    high=difference.interval.high,
                    p_value=difference.p_value,
                )
            )
    return tuple(found)


def _rmse(predictions: Sequence[WindowPrediction]) -> float:
    return float(np.sqrt(np.mean([row.squared_error for row in predictions])))


def write(measured: Measured, directory: Path) -> None:
    for name, columns, rows in (
        (FITS, Fit.COLUMNS, measured.fits),
        (PREDICTIONS, Prediction.COLUMNS, measured.predictions),
        (COMPARISONS, Comparison.COLUMNS, measured.comparisons),
    ):
        with (directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(row.record() for row in rows)


def read(directory: Path) -> Measured:
    """The fits, predictions and comparisons stored under ``directory``.

    Raises:
        SystemExit: If nothing is stored there.
    """
    if not (directory / FITS).is_file():
        raise SystemExit(f"{directory} holds no {FITS}; nothing to render")

    def rows(name: str) -> list[dict[str, str]]:
        with (directory / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    return Measured(
        fits=tuple(Fit.parse(row) for row in rows(FITS)),
        predictions=tuple(Prediction.parse(row) for row in rows(PREDICTIONS)),
        comparisons=tuple(Comparison.parse(row) for row in rows(COMPARISONS)),
    )


def render(measured: Measured, conditions: Mapping[str, object]) -> str:
    """The note's tables: the error of every probe per budget, then the paired comparisons."""
    budgets = sorted({fit.budget for fit in measured.fits})
    probes = list(dict.fromkeys(fit.probe for fit in measured.fits))
    by_probe: dict[tuple[str, int], list[Fit]] = {}
    for fit in measured.fits:
        by_probe.setdefault((fit.probe, fit.budget), []).append(fit)

    def error_cell(probe: str, budget: int) -> str:
        fits = by_probe.get((probe, budget), [])
        if not fits:
            return "—"
        errors = [fit.rmse for fit in fits]
        spread = f" ± {stdev(errors):.2f}" if len(errors) > 1 else ""
        return f"{mean(errors):.2f}{spread}"

    def columns_cell(probe: str) -> str:
        fits = [fit for fit in measured.fits if fit.probe == probe]
        return str(fits[0].columns) if fits else "—"

    errors = table(
        ("probe", "columns", *(str(budget) for budget in budgets)),
        (
            (probe, columns_cell(probe), *(error_cell(probe, budget) for budget in budgets))
            for probe in probes
        ),
    )
    comparisons = table(
        (
            "candidate",
            "rival",
            "budget",
            "candidate RMSE",
            "rival RMSE",
            "reduction",
            "95 % interval",
            "p",
        ),
        (
            (
                row.candidate,
                row.rival,
                str(row.budget),
                f"{row.candidate_rmse:.2f}",
                f"{row.rival_rmse:.2f}",
                f"{row.reduction:+.2f} ({row.relative_reduction:+.1%})",
                f"[{row.low:+.2f}, {row.high:+.2f}]",
                f"{row.p_value:.4f}",
            )
            for row in measured.comparisons
        ),
    )
    return "\n\n".join(
        [
            dated_heading(),
            f"Backbone {conditions.get('weights', '?')}, task {conditions.get('task', '?')}, "
            f"device {conditions.get('device', '?')}, commit {conditions.get('commit', '?')}. "
            f"RMSE on the validation units in the task's unit, mean ± SD over seeds; "
            f"comparisons paired over units with repeats pooled, {RESAMPLES:,} resamples.",
            errors,
            comparisons,
        ]
    )


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="where frozen_representations wrote")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="render the tables from the files written earlier",
    )
    parser.add_argument("--budget", type=int, action="append", help="a budget to fit at")
    parser.add_argument("--seed", type=int, action="append", help="a seed to draw under")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    stored = Stored.read(arguments.directory)
    if arguments.report_only:
        print(render(read(arguments.directory), stored.conditions))
        return
    measured = measure(
        stored,
        budgets=arguments.budget or BUDGETS,
        seeds=arguments.seed or SEEDS,
        resamples=arguments.resamples,
    )
    write(measured, arguments.directory)
    print(render(measured, stored.conditions))


if __name__ == "__main__":
    main()
