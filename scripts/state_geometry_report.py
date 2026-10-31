"""Where the events' signal sits in a frozen encoder's states, and which distance reads it.

A nearest-neighbour detector over the pretrained encoder's states trails the same detector over
the encoder at its initialisation on the satellite missions, and the raised windows showed why
the distance, not the states, loses: pretraining spreads the normal windows' variation over
more directions and encodes the drift between months. This report asks what follows for the
reading. It fits the campaign's detector once more over each encoder's states, then reads the
same states four other ways: the events' separation along each principal component of the
normal fitting states; the Mahalanobis distance over the components that carry most of the
variance against the rest; a nearest-neighbour distance over whitened states; and the
Mahalanobis distance from the window's own month's centre, so a month that drifted whole is
read against itself. The reference of the campaign's detector is read both ways too, against
the fitting side whole and against the window's own month.

``measure`` touches torch and stores every encoder's states of both sides, so a reading can be
corrected without embedding again; ``read`` computes everything from those arrays.

    uv run --env-file .env.r2 scripts/state_geometry_report.py measure
        --task esa-mission2-events --manifest KEY CHECKSUM
        --weights base1 KEY CHECKSUM --weights base2 KEY CHECKSUM --out DIR [--seeds 1 2 3]
    uv run scripts/state_geometry_report.py read --dir DIR [DIR ...] [--resamples 2000]
"""

import argparse
import csv
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.evaluation.adapters.detection.nearest_neighbour_detector import (
    NearestNeighbourDetector,
)
from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
from emblema.evaluation.domain.identifiers import UnitKey
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.anomaly_ranking import AnomalyRanking
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from scripts.reporting import table

Floats = NDArray[np.float64]
Ints = NDArray[np.int64]

WINDOWS = "windows.csv"
FITTING_UNITS = "fitting-units.csv"
CONDITIONS = "conditions.json"
PROFILE = "profile.csv"
SCORES = "scores.csv"
INITIALISATION = "initialisation"

# The campaign's detector, as its arms declare it.
NEIGHBOURS = 5
CONTAMINATION = 0.02
# A reading beyond this many deviations splits the events, as the earlier readings did.
FAR = 10.0
# The components that carry this share of the fitting states' variance are "the top"; the rest
# are "the residual", where defects and far-off inputs show up on pretrained image features.
TOP_SHARE = 0.9
# The declared rule: a scorer enters the screen when both pretrained encoders gain at least this
# much area over the campaign's distance, interval clear of zero, beyond every initialisation's.
LEAST_GAIN = 0.02
# Fewer windows than this leave a month no centre of its own; such a month is read against the
# fitting side's centre instead, so one thin month at a boundary does not stop the reading.
MIN_MONTH_WINDOWS = 10

# The scorers, by the name their columns carry.
DISTANCE = "knn"
PER_UNIT = "knn_unit"
WHITENED = "knn_whitened"
MAHALANOBIS = "mahalanobis"
TOP = "mahalanobis_top"
RESIDUAL = "mahalanobis_residual"
MONTH = "mahalanobis_month"
SCORERS = (DISTANCE, PER_UNIT, WHITENED, MAHALANOBIS, TOP, RESIDUAL, MONTH)
# Those the rule reads against the control: the mission's own campaign detector, `knn` where the
# campaign reads against the fitting side whole and `knn_unit` where it reads each window
# against its own month, so a scorer never enters by re-admitting the campaign's detector.
CANDIDATES = (WHITENED, MONTH, PER_UNIT, DISTANCE)
CONTROLS = (DISTANCE, PER_UNIT)


# --- statistics ------------------------------------------------------------------------------


def shrunk_covariance(rows: Floats) -> Floats:
    """The Ledoit-Wolf shrinkage of the rows' covariance towards a scaled identity.

    The shrinkage intensity is Ledoit and Wolf's closed form (2004), the one scikit-learn's
    ``LedoitWolf`` computes, so a covariance of fewer rows than dimensions still inverts.

    Raises:
        ValueError: If there are fewer than two rows.
    """
    if len(rows) < 2:
        raise ValueError(f"a covariance needs two rows, got {len(rows)}")
    centred = rows - rows.mean(axis=0)
    count, width = centred.shape
    sample = centred.T @ centred / count
    mu = float(np.trace(sample)) / width
    squared = centred**2
    beta_sum = float((squared.T @ squared).sum()) / count - float((sample**2).sum())
    delta = float(((sample - mu * np.eye(width)) ** 2).sum())
    beta = min(beta_sum / count, delta) if delta > 0 else 0.0
    shrinkage = 0.0 if delta == 0 else beta / delta
    shrunk: Floats = (1 - shrinkage) * sample + shrinkage * mu * np.eye(width)
    return shrunk


def inverse_root(covariance: Floats) -> Floats:
    """The symmetric inverse square root of a covariance, whitening what it multiplies."""
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, np.finfo(float).tiny, None)
    root: Floats = vectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ vectors.T
    return root


def mahalanobis(rows: Floats, centre: Floats, covariance: Floats) -> Floats:
    """Each row's squared Mahalanobis distance from ``centre`` under ``covariance``."""
    whitened = (rows - centre) @ inverse_root(covariance)
    squared: Floats = np.einsum("ij,ij->i", whitened, whitened)
    return squared


@dataclass(frozen=True)
class RankedScores:
    """One score vector sorted once, so a resample of units costs a pass and no sort.

    Attributes:
        order: Indices sorting the scores ascending.
        positive: Whether each window of ``order`` is an event's.
        starts: Where each group of tied scores starts in ``order``.
    """

    order: Ints
    positive: NDArray[np.bool_]
    starts: Ints

    @classmethod
    def of(cls, scores: Floats, positive: NDArray[np.bool_]) -> "RankedScores":
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
        return cls(order=order, positive=positive[order], starts=starts)

    def auroc(self, weights: Floats | None = None) -> float:
        """The chance an event's window outscores a normal one, ties half, windows weighted.

        ``weights`` counts each window that many times, as a resample of units does; ``None``
        counts every window once. Not a number when a side has no weight.
        """
        counted = np.ones(len(self.order)) if weights is None else weights[self.order]
        positive_weight = np.where(self.positive, counted, 0.0)
        negative_weight = np.where(self.positive, 0.0, counted)
        positive_in_group = np.add.reduceat(positive_weight, self.starts)
        negative_in_group = np.add.reduceat(negative_weight, self.starts)
        negatives_below = np.cumsum(negative_in_group) - negative_in_group
        total_positive = positive_weight.sum()
        total_negative = negative_weight.sum()
        if total_positive == 0 or total_negative == 0:
            return float("nan")
        credit = (positive_in_group * (negatives_below + 0.5 * negative_in_group)).sum()
        return float(credit / (total_positive * total_negative))


def auroc(scores: Floats, positive: NDArray[np.bool_]) -> float:
    """The chance an event's window outscores a normal one, ties counted half."""
    return RankedScores.of(scores, positive).auroc()


def event_weighted_area(
    scores: Floats, units: Sequence[str], positions: Ints, ends_at: Floats, targets: Floats
) -> float:
    """The register's measure: the area under the precision-recall curve, an event weighing one."""
    return AnomalyRanking.of(
        WindowPrediction(
            window=TaskWindow(unit=UnitKey(unit), position=int(position), ends_at=float(end)),
            target=float(target),
            predicted=float(score),
        )
        for score, unit, position, end, target in zip(
            scores, units, positions, ends_at, targets, strict=True
        )
    ).average_precision


# --- the scorers -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """The events' separation along each principal component of the normal fitting states.

    Attributes:
        share: The share of the fitting states' variance each component carries, descending.
        separation: The area of ``|projection|`` of events against normal windows per component.
        top: How many leading components carry ``TOP_SHARE`` of the variance.
    """

    share: Floats
    separation: Floats
    top: int


def scorers_of(
    fitting: Floats,
    scored: Floats,
    *,
    fitting_units: Sequence[str],
    scored_units: Sequence[str],
    positive: NDArray[np.bool_],
) -> tuple[dict[str, Floats], Profile]:
    """Every scorer's score of every scored window, and the per-component profile.

    The states are standardised over the fitting side as the campaign's detector standardises
    them, so every scorer reads the same coordinates and the campaign's distance is reproduced
    exactly. The month's centre is the coordinate-wise median of all its scored windows, events
    among them, as the campaign's per-unit reference reads them: a median stands up to the few
    events a month holds, where a month's own covariance would not, so the covariance stays the
    fitting side's. A month of fewer than ``MIN_MONTH_WINDOWS`` windows is centred on the
    fitting side instead.
    """
    # The campaign's two detectors as they are, alarm and all, so a check against the campaign's
    # exported scores is exact; an alarm's level moves no area.
    detector = NearestNeighbourDetector.fitted_on(
        fitting, neighbours=NEIGHBOURS, contamination=CONTAMINATION
    )
    standardised_fitting = detector.fitted
    standardised_scored = (scored - detector.centre) / detector.scale
    width = standardised_fitting.shape[1]
    scores: dict[str, Floats] = {DISTANCE: detector.scores(scored)}
    per_unit = NearestNeighbourDetector.fitted_on(
        fitting, neighbours=NEIGHBOURS, contamination=CONTAMINATION, units=fitting_units
    )
    scores[PER_UNIT] = per_unit.scores(scored, units=scored_units)

    covariance = shrunk_covariance(standardised_fitting)
    centre = standardised_fitting.mean(axis=0)
    scores[MAHALANOBIS] = mahalanobis(standardised_scored, centre, covariance)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    descending = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[descending], vectors[:, descending]
    share = eigenvalues / eigenvalues.sum()
    top = int(np.searchsorted(np.cumsum(share), TOP_SHARE) + 1)
    projected = (standardised_scored - centre) @ vectors
    whitened_squares = projected**2 / eigenvalues
    scores[TOP] = whitened_squares[:, :top].sum(axis=1)
    scores[RESIDUAL] = whitened_squares[:, top:].sum(axis=1)
    separation = np.array([auroc(np.abs(projected[:, j]), positive) for j in range(width)])

    root = inverse_root(covariance)
    whitened = NearestNeighbourDetector(
        centre=np.zeros(width),
        scale=np.ones(width),
        fitted=(standardised_fitting - centre) @ root,
        neighbours=detector.neighbours,
        alarm=ScoreAlarm(level=0.0, per_unit=False, unit_spread=1.0),
    )
    scores[WHITENED] = whitened.scores((standardised_scored - centre) @ root)

    centred = np.empty_like(standardised_scored)
    units = np.asarray(scored_units)
    for unit in np.unique(units):
        rows = units == unit
        own = (
            np.median(standardised_scored[rows], axis=0)
            if rows.sum() >= MIN_MONTH_WINDOWS
            else centre
        )
        centred[rows] = standardised_scored[rows] - own
    month_whitened = centred @ root
    scores[MONTH] = np.einsum("ij,ij->i", month_whitened, month_whitened)
    return scores, Profile(share=share, separation=separation, top=top)


# --- measure ---------------------------------------------------------------------------------


def measure(arguments: argparse.Namespace) -> None:  # pragma: no cover - torch, store, corpora
    import torch

    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_store
    from emblema.entrypoints.source_revision import SourceRevision
    from emblema.pretraining.adapters.training.devices import available_device
    from emblema.shared.adapters.arrays.binned_tokens import BinnedTokens
    from scripts.detection_states import (
        FrozenStates,
        detection_sides,
        detector_recipe,
        encoders_to_read,
        ref_of,
        shared_shape,
    )

    device = available_device() if arguments.device is None else arguments.device
    store = configured_store(Settings())
    manifest = ref_of(*arguments.manifest)
    layout = BinnedTokens(shared_shape(store, arguments.weights).bins)
    sides = detection_sides(
        arguments.task,
        manifest,
        store=store,
        layout=layout,
        workspace=arguments.workspace,
        corpora=arguments.corpora,
    )
    print(
        f"{len(sides.fitting)} fitting, {len(sides.scored)} scored "
        f"({int((sides.targets > 0).sum())} in events); {device}",
        flush=True,
    )
    frozen = FrozenStates(store, sides, device=device, batch_size=arguments.batch_size)
    arguments.out.mkdir(parents=True, exist_ok=True)
    seconds: dict[str, float] = {}
    for encoder in encoders_to_read(arguments.weights, arguments.seeds):
        started = time.perf_counter()
        # The detector's reference does not enter the states; the campaign's recipe is asked for
        # its seeding and reading alone.
        recipe = detector_recipe(encoder, per_unit=False)
        embedding = frozen.embedding(encoder, recipe)
        np.savez(
            arguments.out / f"states-{encoder.name}.npz",
            fitting=frozen.states(embedding, sides.laid_fitting),
            scored=frozen.states(embedding, sides.laid_scored),
        )
        seconds[encoder.name] = time.perf_counter() - started
        print(f"{encoder.name}: {seconds[encoder.name]:.0f} s", flush=True)
    write_csv(
        arguments.out / WINDOWS,
        ["position", "unit", "ends_at", "target", "largest"],
        (
            [int(p), u, float(e), float(t), f"{largest:.6g}"]
            for p, u, e, t, largest in zip(
                sides.positions,
                sides.scored_units,
                sides.ends_at,
                sides.targets,
                sides.largest,
                strict=True,
            )
        ),
    )
    write_csv(arguments.out / FITTING_UNITS, ["unit"], ([unit] for unit in sides.fitting_units))
    conditions: Mapping[str, object] = {
        "commit": SourceRevision().current(),
        "device": device,
        "torch": torch.__version__,
        "task": sides.task,
        "manifest": f"{manifest.key}@{manifest.checksum}",
        "weights": {label: f"{key}@{checksum}" for label, key, checksum in arguments.weights},
        "initialisation_seeds": list(arguments.seeds),
        "bins": layout.bins,
        "vocabulary": sides.vocabulary,
        "windows": {"fitting": len(sides.fitting), "scored": len(sides.scored)},
        "seconds": seconds,
    }
    (arguments.out / CONDITIONS).write_text(json.dumps(conditions, indent=2) + "\n")
    print(f"stored under {arguments.out}")


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


# --- read ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Measured:
    """What ``measure`` stored for one task."""

    task: str
    units: list[str]
    positions: Ints
    ends_at: Floats
    targets: Floats
    largest: Floats
    fitting_units: list[str]
    states: dict[str, tuple[Floats, Floats]]

    @classmethod
    def read(cls, directory: Path) -> "Measured":
        conditions = json.loads((directory / CONDITIONS).read_text())
        with (directory / WINDOWS).open() as handle:
            rows = list(csv.DictReader(handle))
        with (directory / FITTING_UNITS).open() as handle:
            fitting_units = [row["unit"] for row in csv.DictReader(handle)]
        states = {}
        for path in sorted(directory.glob("states-*.npz")):
            with np.load(path) as archive:
                states[path.stem.removeprefix("states-")] = (
                    np.asarray(archive["fitting"], dtype=np.float64),
                    np.asarray(archive["scored"], dtype=np.float64),
                )
        return cls(
            task=conditions["task"],
            units=[row["unit"] for row in rows],
            positions=np.array([int(row["position"]) for row in rows]),
            ends_at=np.array([float(row["ends_at"]) for row in rows]),
            targets=np.array([float(row["target"]) for row in rows]),
            largest=np.array([float(row["largest"]) for row in rows]),
            fitting_units=fitting_units,
            states=states,
        )

    @property
    def positive(self) -> NDArray[np.bool_]:
        return self.targets > 0

    @property
    def pretrained(self) -> list[str]:
        return [name for name in self.states if not name.startswith(INITIALISATION)]

    @property
    def initialisations(self) -> list[str]:
        return [name for name in self.states if name.startswith(INITIALISATION)]


@dataclass(frozen=True)
class Gain:
    """A scorer's gain in area over the campaign's distance, with its interval over months."""

    point: float
    low: float
    high: float


@dataclass(frozen=True)
class Reading:
    """One task read: every encoder's scores, profile, areas and gains over the control."""

    measured: Measured
    control: str
    scores: dict[str, dict[str, Floats]]
    profiles: dict[str, Profile]
    gains: dict[tuple[str, str], Gain]

    @property
    def ruled(self) -> tuple[str, ...]:
        """The scorers the rule reads: every candidate but the control."""
        return tuple(scorer for scorer in CANDIDATES if scorer != self.control)

    @classmethod
    def of(cls, measured: Measured, *, control: str, resamples: int, seed: int) -> "Reading":
        """Read ``measured`` against ``control``, one of ``CONTROLS``.

        Raises:
            ValueError: If the control is not a campaign detector.
        """
        if control not in CONTROLS:
            raise ValueError(f"the control is one of {CONTROLS}, not {control!r}")
        scores: dict[str, dict[str, Floats]] = {}
        profiles: dict[str, Profile] = {}
        for name, (fitting, scored) in measured.states.items():
            scores[name], profiles[name] = scorers_of(
                fitting,
                scored,
                fitting_units=measured.fitting_units,
                scored_units=measured.units,
                positive=measured.positive,
            )
        return cls(
            measured=measured,
            control=control,
            scores=scores,
            profiles=profiles,
            gains=gains_of(measured, scores, control=control, resamples=resamples, seed=seed),
        )


def gains_of(
    measured: Measured,
    scores: Mapping[str, Mapping[str, Floats]],
    *,
    control: str,
    resamples: int,
    seed: int,
) -> dict[tuple[str, str], Gain]:
    """Each candidate scorer's gain in area over the control, per encoder.

    The interval resamples the months (units) with replacement, the same draw for every encoder
    and scorer, and counts each window as many times as its month was drawn; a resample that
    leaves no event or no normal window is dropped.
    """
    units = np.asarray(measured.units)
    names, inverse = np.unique(units, return_inverse=True)
    draws = np.random.default_rng(seed)
    multiplicity = [
        np.bincount(draws.integers(0, len(names), len(names)), minlength=len(names))
        for _ in range(resamples)
    ]
    gains: dict[tuple[str, str], Gain] = {}
    ruled = tuple(scorer for scorer in CANDIDATES if scorer != control)
    for encoder, by_scorer in scores.items():
        ranked = {s: RankedScores.of(by_scorer[s], measured.positive) for s in CANDIDATES}
        base = ranked[control]
        for scorer in ruled:
            point = ranked[scorer].auroc() - base.auroc()
            spread = []
            for counts in multiplicity:
                weights = counts[inverse].astype(np.float64)
                spread.append(ranked[scorer].auroc(weights) - base.auroc(weights))
            low, high = np.nanquantile(spread, [0.025, 0.975])
            gains[(encoder, scorer)] = Gain(point, float(low), float(high))
    return gains


def enters(reading: Reading, scorer: str) -> str:
    """The declared rule for one scorer: ``yes``, ``no``, or why it is void.

    A scorer enters when both pretrained encoders gain at least ``LEAST_GAIN`` over the control
    with the interval clear of zero, and each gain is larger than every initialisation's gain
    under the same scorer (a point, not an interval). The whitened distance needs besides that
    the residual components separate the events at least as well as the top ones under both
    pretrained encoders, since that is the mechanism by which whitening helps. The rule reads
    what a scorer does for the pretrained states beyond what it does for the drawn ones, not
    which scorer is the best detector.

    Raises:
        ValueError: If the scorer is the control.
    """
    if scorer == reading.control:
        raise ValueError(f"{scorer} is the control; it cannot enter against itself")
    measured = reading.measured
    if len(measured.pretrained) < 2 or len(measured.initialisations) < 2:
        return "void: two pretrained encoders and two initialisations are needed"
    drawn_ceiling = max(reading.gains[(name, scorer)].point for name in measured.initialisations)
    for name in measured.pretrained:
        gain = reading.gains[(name, scorer)]
        if gain.point < LEAST_GAIN or not gain.low > 0.0 or gain.point <= drawn_ceiling:
            return "no"
        if scorer == WHITENED:
            positive = measured.positive
            if auroc(reading.scores[name][RESIDUAL], positive) < auroc(
                reading.scores[name][TOP], positive
            ):
                return "no: the top components separate the events better than the residual"
    return "yes"


def areas_table(reading: Reading) -> str:
    """Every scorer's area under every encoder: events against normal windows, by the reading."""
    measured = reading.measured
    far = measured.positive & (measured.largest > FAR)
    near = measured.positive & (measured.largest <= FAR)
    normal = ~measured.positive
    lines = []
    for encoder, by_scorer in reading.scores.items():
        for scorer in SCORERS:
            scores = by_scorer[scorer]
            cells = [
                f"{auroc(scores[group | normal], measured.positive[group | normal]):.3f}"
                if group.any()
                else "-"
                for group in (measured.positive, far, near)
            ]
            weighted = event_weighted_area(
                scores, measured.units, measured.positions, measured.ends_at, measured.targets
            )
            lines.append([encoder, scorer, *cells, f"{weighted:.3f}"])
    return table(
        ["Encoder", "Scorer", "All events", f"Beyond {FAR:g}", f"Within {FAR:g}", "PR-AUC"], lines
    )


def profile_table(reading: Reading, *, leading: int = 5) -> str:
    """The leading components' variance share and separation, and the residual's, per encoder."""
    lines = []
    for encoder, profile in reading.profiles.items():
        positive = reading.measured.positive
        scores = reading.scores[encoder]
        lines.append(
            [
                encoder,
                str(profile.top),
                " ".join(f"{s:.2f}" for s in profile.share[:leading]),
                " ".join(f"{a:.2f}" for a in profile.separation[:leading]),
                f"{profile.separation[: profile.top].max():.3f}",
                f"{profile.separation[profile.top :].max():.3f}"
                if profile.top < len(profile.separation)
                else "-",
                f"{auroc(scores[TOP], positive):.3f}",
                f"{auroc(scores[RESIDUAL], positive):.3f}",
            ]
        )
    return table(
        [
            "Encoder",
            f"Top ({TOP_SHARE:.0%})",
            f"Share, first {leading}",
            f"Separation, first {leading}",
            "Best top",
            "Best residual",
            "Mahalanobis top",
            "Mahalanobis residual",
        ],
        lines,
    )


def gains_table(reading: Reading) -> str:
    lines = [
        [encoder, scorer, f"{gain.point:+.3f} [{gain.low:+.3f}; {gain.high:+.3f}]"]
        for (encoder, scorer), gain in reading.gains.items()
    ]
    return table(["Encoder", "Scorer", f"Gain over {reading.control}"], lines)


def write_profile(reading: Reading, directory: Path) -> None:
    write_csv(
        directory / PROFILE,
        ["encoder", "component", "share", "separation"],
        (
            [encoder, j, f"{profile.share[j]:.6g}", f"{profile.separation[j]:.6g}"]
            for encoder, profile in reading.profiles.items()
            for j in range(len(profile.share))
        ),
    )


def write_scores(reading: Reading, directory: Path) -> None:
    measured = reading.measured
    write_csv(
        directory / SCORES,
        ["encoder", "scorer", "position", "unit", "target", "score"],
        (
            [encoder, scorer, int(p), u, float(t), f"{s:.10g}"]
            for encoder, by_scorer in reading.scores.items()
            for scorer, scores in by_scorer.items()
            for p, u, t, s in zip(
                measured.positions, measured.units, measured.targets, scores, strict=True
            )
        ),
    )


def check_against_campaign(
    reading: Reading, predictions: Path, campaign: str, label: str, scorer: str
) -> str:
    """How closely ``scorer`` reproduces the campaign's scores, window by window.

    The backbone ``label`` is matched to the campaign's states detector, each initialisation to
    its detector at the initialisation of the same seed.
    """
    measured = reading.measured
    theirs: dict[tuple[str, int], dict[int, float]] = {}
    with predictions.open() as handle:
        for row in csv.DictReader(handle):
            if row["campaign"] != campaign:
                continue
            candidate = row["candidate"].split("@")[0]
            if candidate in ("untrained_knn", "frozen_knn"):
                theirs.setdefault((candidate, int(row["seed"])), {})[int(row["position"])] = float(
                    row["predicted"]
                )
    lines = []
    for (candidate, seed), by_position in sorted(theirs.items()):
        name = f"{INITIALISATION}-{seed}" if candidate == "untrained_knn" else label
        if name not in reading.scores or (candidate == "frozen_knn" and seed != 1):
            continue
        ours = reading.scores[name][scorer]
        shared = [i for i, p in enumerate(measured.positions) if int(p) in by_position]
        if not shared:
            continue
        a = ours[shared]
        b = np.array([by_position[int(measured.positions[i])] for i in shared])
        lines.append(
            [
                name,
                candidate,
                str(seed),
                f"{len(shared)} of {len(by_position)}",
                f"{float(np.max(np.abs(a - b))):.2e}",
                f"{float(np.corrcoef(a, b)[0, 1]):.6f}",
            ]
        )
    return table(
        ["Encoder", "Candidate", "Seed", "Windows", "Largest difference", "Correlation"], lines
    )


def read(arguments: argparse.Namespace) -> None:
    for directory in arguments.dir:
        measured = Measured.read(directory)
        reading = Reading.of(
            measured, control=arguments.control, resamples=arguments.resamples, seed=arguments.seed
        )
        write_profile(reading, directory)
        write_scores(reading, directory)
        for predictions, campaign, label, scorer in arguments.campaign_predictions or ():
            print(f"Against the campaign, {scorer}:\n")
            print(check_against_campaign(reading, Path(predictions), campaign, label, scorer))
            print()
        print(f"## {measured.task} ({len(measured.units)} scored windows)\n")
        print("Where the events' signal sits (components of the normal fitting states):\n")
        print(profile_table(reading))
        print("\nAreas (window-level AUROC; PR-AUC event-weighted):\n")
        print(areas_table(reading))
        print(f"\nGains over the control {reading.control} (95 % over resampled months):\n")
        print(gains_table(reading))
        print()
        for scorer in reading.ruled:
            print(f"{scorer} enters the screen: {enters(reading, scorer)}")
        print()


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    measuring = steps.add_parser("measure")
    measuring.add_argument("--task", required=True)
    measuring.add_argument("--manifest", nargs=2, metavar=("KEY", "CHECKSUM"), required=True)
    measuring.add_argument(
        "--weights",
        nargs=3,
        action="append",
        metavar=("LABEL", "KEY", "CHECKSUM"),
        required=True,
        help="a pretrained backbone; the first one's shape is the initialisation's",
    )
    measuring.add_argument("--out", type=Path, required=True)
    measuring.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    measuring.add_argument("--batch-size", type=int, default=64)
    measuring.add_argument("--workspace", type=Path, default=Path("data/workspace"))
    measuring.add_argument("--corpora", type=Path, default=Path("data/raw"))
    measuring.add_argument("--device", default=None)
    reading = steps.add_parser("read")
    reading.add_argument("--dir", type=Path, nargs="+", required=True)
    reading.add_argument(
        "--control",
        choices=CONTROLS,
        default=DISTANCE,
        help="the mission's campaign detector the candidates are read against",
    )
    reading.add_argument("--resamples", type=int, default=2000)
    reading.add_argument("--seed", type=int, default=1)
    reading.add_argument(
        "--campaign-predictions",
        nargs=4,
        action="append",
        metavar=("PREDICTIONS", "CAMPAIGN", "LABEL", "SCORER"),
        help="check a scorer against a campaign's exported predictions",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.step == "measure":
        measure(arguments)
    else:
        read(arguments)


if __name__ == "__main__":
    main()
