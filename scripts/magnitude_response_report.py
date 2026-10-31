"""Whether a frozen encoder's states keep the size of a reading, as a detector reads them.

A detector over an encoder's states scores a window by its mean distance to the nearest fitting
windows. On the satellite missions the pretrained states trail the same encoder at its
initialisation on the events whose window holds a reading beyond ten deviations, and on the second
mission on the rest too. Two causes would look alike in those areas: the pretrained states lose
how large a reading is (the response to a larger reading saturates), or they are worse for the
detector at every size (a duller response, or a reference the validation windows lie far from).
This script separates them by a test under control rather than by the events' labels: normal
windows of the validation side are raised, their largest reading's cell, its channel or the whole
window multiplied by a factor, and each encoder's distance is read before and after. The windows'
distances unraised, and the geometry of the fitting states, are kept beside.

``measure`` touches torch and writes, per task, ``scored.csv`` (every validation window under
every encoder, as the campaign scores it and as the raw distance), ``responses.csv`` (the raised
windows' distances), ``geometry.csv`` (the fitting states per encoder) and ``conditions.json``.
``read`` computes the tables a note pastes from those files and nothing else, and checks the
scores against a campaign's exported predictions when given them.

    uv run --env-file .env.r2 scripts/magnitude_response_report.py measure
        --task esa-mission2-events --manifest KEY CHECKSUM
        --weights base1 KEY CHECKSUM --weights base2 KEY CHECKSUM --out DIR
        [--reference unit] [--seeds 1 2 3] [--normal 300] [--factors 1 3 10 30]
    uv run scripts/magnitude_response_report.py read --dir DIR
        [--campaign-predictions PREDICTIONS.csv CAMPAIGN LABEL ...]
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.shared.kernel.ordering import seeded_rank
from emblema.shared.kernel.tokens import Token, TokenWindow
from scripts.reporting import table

Floats = NDArray[np.float64]

SCORED = "scored.csv"
RESPONSES = "responses.csv"
GEOMETRY = "geometry.csv"
CONDITIONS = "conditions.json"

SCOPES = ("cell", "channel", "window")
INITIALISATION = "initialisation"
# A reading beyond this many deviations splits the events, as the reading after the screen did.
FAR = 10.0
# Fitting windows whose distance to the others is read, for the scale of the reference.
FITTING_DRAWN = 2000
RESAMPLES = 2000
# The declared rule: the sensitivity at the large raise over that at the moderate one, on the
# scope an event is local to, falls to at most this share with its interval below one.
MODERATE = 3.0
LARGE = 10.0
TILT = 0.75
DULL = 0.8
RULED = "channel"
DISPLACEMENT = "displacement"
DISTANCE = "distance"
# The rule reads how far the state moves, which no neighbour enters; the detector's distance, which
# also carries how far the window lay from its neighbours before it was raised, is read beside.
MEASURES = {
    DISPLACEMENT: "the state's displacement, in the fitting side's spread per feature",
    DISTANCE: "the detector's distance, in the median unraised distance",
}


def raised(window: TokenWindow, *, scope: str, factor: float, bins: int) -> TokenWindow:
    """``window`` with the readings of ``scope`` around its largest one multiplied by ``factor``.

    The largest reading is the timed token of largest magnitude, the first of a tie. ``cell`` takes
    the readings of its channel in its bin, as the layout cuts the window into ``bins`` spans;
    ``channel`` every reading of its channel; ``window`` every timed reading. Static features stay.

    Raises:
        ValueError: If the scope is not one of ``SCOPES``.
    """
    values = np.asarray(window.values, dtype=np.float64)
    timed = ~np.asarray(window.timeless, dtype=bool)
    channels = np.asarray(window.channel_ids, dtype=np.int64)
    times = np.asarray(window.times, dtype=np.float64)
    peak = int(np.argmax(np.where(timed, np.abs(values), -np.inf)))
    same_channel = timed & (channels == channels[peak])
    if scope == "window":
        chosen = timed
    elif scope == "channel":
        chosen = same_channel
    elif scope == "cell":
        cells = np.minimum(np.floor(times * bins).astype(np.int64), bins - 1)
        chosen = same_channel & (cells == cells[peak])
    else:
        raise ValueError(f"a scope is one of {SCOPES}, not {scope!r}")
    scaled = np.where(chosen, values * factor, values)
    return TokenWindow.of(
        Token(channel_id=int(c), value=float(v), time=float(t), gap=float(g), timeless=bool(s))
        for c, v, t, g, s in zip(
            window.channel_ids, scaled, window.times, window.gaps, window.timeless, strict=True
        )
    )


def drawn_normals(positions: Sequence[int], *, seed: int, count: int) -> list[int]:
    """``count`` of ``positions`` in the order ``seed`` ranks them, all of them if fewer."""
    ranked = sorted(positions, key=lambda position: (seeded_rank(seed, position), position))
    return ranked[:count]


def participation_ratio(rows: Floats) -> float:
    """How many directions the rows' variance spreads over: (Σλ)² / Σλ² of their covariance."""
    centred = rows - rows.mean(axis=0)
    eigenvalues = np.clip(np.linalg.eigvalsh(centred.T @ centred / max(len(rows) - 1, 1)), 0, None)
    total = float(eigenvalues.sum())
    return total * total / float((eigenvalues**2).sum()) if total > 0 else 0.0


def auroc(positive: Floats, negative: Floats) -> float:
    """The chance a positive outscores a negative, ties counted half."""
    scores = np.concatenate((positive, negative))
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    # Average ranks over ties, so a tie counts half.
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + end + 1) / 2.0
    count = len(positive)
    return float((ranks[:count].sum() - count * (count + 1) / 2) / (count * len(negative)))


# --- measure ---------------------------------------------------------------------------------


def measure(arguments: argparse.Namespace) -> None:  # pragma: no cover - torch, store, corpora
    import torch

    from emblema.config.settings import Settings
    from emblema.entrypoints.configured import configured_store
    from emblema.entrypoints.source_revision import SourceRevision
    from emblema.evaluation.adapters.detection.nearest_neighbour_detector import (
        NearestNeighbourDetector,
    )
    from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
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
    targets, largest = sides.targets, sides.largest
    normals = drawn_normals(
        [
            index
            for index in range(len(sides.scored))
            if targets[index] == 0 and largest[index] <= FAR
        ],
        seed=arguments.draw_seed,
        count=arguments.normal,
    )
    per_unit = arguments.reference == "unit"
    fitting_units = sides.fitting_units if per_unit else None
    scored_units = sides.scored_units if per_unit else None
    factors = sorted(set(arguments.factors) | {1.0})
    # Every window is laid out once and its cells kept, so each encoder reads the same arrays;
    # a raised window is laid out as soon as it is made and only its cells are kept.
    raised_keys: list[tuple[int, str, float]] = []
    laid_raised = []
    for index in normals:
        for scope in SCOPES:
            for factor in factors:
                if factor == 1.0:
                    continue
                window = raised(sides.scored[index], scope=scope, factor=factor, bins=layout.bins)
                raised_keys.append((index, scope, factor))
                laid_raised.extend(layout.lay([window]))
    print(
        f"{len(sides.fitting)} fitting, {len(sides.scored)} scored "
        f"({int((targets > 0).sum())} in events), "
        f"{len(normals)} normal windows raised {len(raised_keys)} ways; {device}",
        flush=True,
    )
    frozen = FrozenStates(store, sides, device=device, batch_size=arguments.batch_size)
    arguments.out.mkdir(parents=True, exist_ok=True)
    scored_rows: list[list[object]] = []
    response_rows: list[list[object]] = []
    geometry_rows: list[list[object]] = []
    seconds: dict[str, float] = {}
    for encoder in encoders_to_read(arguments.weights, arguments.seeds):
        started = time.perf_counter()
        label, seed = encoder.label, encoder.seed
        recipe = detector_recipe(encoder, per_unit=per_unit)
        embedding = frozen.embedding(encoder, recipe)
        fitted_states = frozen.states(embedding, sides.laid_fitting)
        detector = NearestNeighbourDetector.fitted_on(
            fitted_states,
            neighbours=recipe.nearest(),
            contamination=recipe.contamination,
            units=fitting_units,
        )
        # The same rows, neighbours and standardisation with the alarm at zero and no unit: the
        # mean distance itself, which a raised window and its original are compared by.
        distance = replace(detector, alarm=ScoreAlarm(level=0.0, per_unit=False, unit_spread=1.0))
        scored_states = frozen.states(embedding, sides.laid_scored)
        campaign_scores = detector.scores(scored_states, units=scored_units)
        raw = distance.scores(scored_states)
        for index in range(len(sides.scored)):
            scored_rows.append(
                [
                    label,
                    seed,
                    int(sides.positions[index]),
                    sides.scored_units[index],
                    float(sides.ends_at[index]),
                    float(targets[index]),
                    f"{largest[index]:.6g}",
                    f"{campaign_scores[index]:.10g}",
                    f"{raw[index]:.10g}",
                ]
            )
        raised_states = frozen.states(embedding, laid_raised)
        raised_distances = distance.scores(raised_states)
        moved = np.linalg.norm(
            (raised_states - scored_states[[index for index, _, _ in raised_keys]])
            / detector.standardisation.scale,
            axis=1,
        )
        for (index, scope, factor), value, shift in zip(
            raised_keys, raised_distances, moved, strict=True
        ):
            response_rows.append(
                [
                    label,
                    seed,
                    int(sides.positions[index]),
                    scope,
                    factor,
                    f"{value:.10g}",
                    f"{shift:.10g}",
                ]
            )
        # A fitting window's distance to the others: one more neighbour asked for, itself at no
        # distance among them, and the mean put back over the others.
        drawn_fitting = drawn_normals(
            list(range(len(sides.fitting))), seed=arguments.draw_seed, count=FITTING_DRAWN
        )
        among = (
            replace(distance, neighbours=distance.neighbours + 1).scores(
                fitted_states[drawn_fitting]
            )
            * (distance.neighbours + 1)
            / distance.neighbours
        )
        geometry_rows.append(
            [
                label,
                seed,
                fitted_states.shape[1],
                f"{participation_ratio(detector.fitted):.6g}",
                f"{participation_ratio(fitted_states):.6g}",
                f"{float(np.median(among)):.10g}",
                f"{float(np.mean(np.linalg.norm(fitted_states, axis=1))):.6g}",
            ]
        )
        seconds[encoder.name] = time.perf_counter() - started
        print(f"{label} seed {seed}: {seconds[encoder.name]:.0f} s", flush=True)
    write_csv(
        arguments.out / SCORED,
        [
            "encoder",
            "seed",
            "position",
            "unit",
            "ends_at",
            "target",
            "largest",
            "score",
            "distance",
        ],
        scored_rows,
    )
    write_csv(
        arguments.out / RESPONSES,
        ["encoder", "seed", "position", "scope", "factor", "distance", "displacement"],
        response_rows,
    )
    write_csv(
        arguments.out / GEOMETRY,
        [
            "encoder",
            "seed",
            "width",
            "participation_standardised",
            "participation_raw",
            "fitting_distance",
            "state_norm",
        ],
        geometry_rows,
    )
    conditions: Mapping[str, object] = {
        "commit": SourceRevision().current(),
        "device": device,
        "torch": torch.__version__,
        "task": sides.task,
        "manifest": f"{manifest.key}@{manifest.checksum}",
        "weights": {label: f"{key}@{checksum}" for label, key, checksum in arguments.weights},
        "initialisation_seeds": list(arguments.seeds),
        "reference": arguments.reference,
        "bins": layout.bins,
        "vocabulary": sides.vocabulary,
        "factors": factors,
        "normal_windows": len(normals),
        "draw_seed": arguments.draw_seed,
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
class Responses:
    """The raised windows' distances, one array per encoder, scope and factor, windows aligned.

    Attributes:
        windows: The positions of the raised windows, in the order every array holds them.
        distances: ``(encoder, scope, factor)`` to the windows' distances; factor 1 under every
            scope is the window unraised.
        displacements: The same keys to how far each window's standardised state moved.
    """

    windows: tuple[int, ...]
    distances: Mapping[tuple[str, str, float], Floats]
    displacements: Mapping[tuple[str, str, float], Floats]

    @classmethod
    def read(cls, directory: Path) -> "Responses":
        """The responses ``measure`` wrote under ``directory``, with the unraised distances."""
        unraised: dict[tuple[str, int], float] = {}
        with (directory / SCORED).open() as handle:
            for row in csv.DictReader(handle):
                unraised[(encoder_name(row), int(row["position"]))] = float(row["distance"])
        rows: dict[tuple[str, str, float], dict[int, tuple[float, float]]] = defaultdict(dict)
        with (directory / RESPONSES).open() as handle:
            for row in csv.DictReader(handle):
                rows[(encoder_name(row), row["scope"], float(row["factor"]))][
                    int(row["position"])
                ] = (float(row["distance"]), float(row["displacement"]))
        windows = tuple(sorted({position for by in rows.values() for position in by}))
        distances: dict[tuple[str, str, float], Floats] = {}
        displacements: dict[tuple[str, str, float], Floats] = {}
        for (encoder, scope, factor), by in rows.items():
            distances[(encoder, scope, factor)] = np.array([by[w][0] for w in windows])
            displacements[(encoder, scope, factor)] = np.array([by[w][1] for w in windows])
            distances[(encoder, scope, 1.0)] = np.array([unraised[(encoder, w)] for w in windows])
            displacements[(encoder, scope, 1.0)] = np.zeros(len(windows))
        return cls(windows=windows, distances=distances, displacements=displacements)

    @property
    def encoders(self) -> list[str]:
        return sorted({encoder for encoder, _, _ in self.distances})

    @property
    def factors(self) -> list[float]:
        return sorted({factor for _, _, factor in self.distances if factor != 1.0})

    def rise(
        self, encoder: str, scope: str, factor: float, rows: NDArray[np.int64], measure: str
    ) -> float:
        """How far ``rows`` moved when raised, their median, by ``measure``.

        ``distance``: the detector's distance less the unraised one, in the median unraised
        distance. ``displacement``: how far the state moved, in the fitting side's spread per
        feature, which no neighbour enters.

        Raises:
            ValueError: If the measure is neither.
        """
        if measure == DISPLACEMENT:
            return float(np.median(self.displacements[(encoder, scope, factor)][rows]))
        if measure != DISTANCE:
            raise ValueError(f"a rise is measured by {MEASURES}, not {measure!r}")
        before = self.distances[(encoder, scope, 1.0)][rows]
        after = self.distances[(encoder, scope, factor)][rows]
        return float(np.median(after - before) / np.median(before))


def encoder_name(row: Mapping[str, str]) -> str:
    """An encoder by its label, an initialisation by its seed too."""
    return f"{row['encoder']}-{row['seed']}" if row["encoder"] == INITIALISATION else row["encoder"]


def sensitivity(
    responses: Responses,
    encoder: str,
    scope: str,
    factor: float,
    rows: NDArray[np.int64],
    measure: str,
) -> tuple[float, bool]:
    """The encoder's rise over the mean rise of the initialisations other than itself.

    An initialisation read this way against the others shows how far the draw of weights alone
    moves the measure, the reference a pretrained encoder's departure is read against. Beside it,
    whether both rises were above zero, since a ratio of rises means nothing otherwise.
    """
    others = [
        name for name in responses.encoders if name.startswith(INITIALISATION) and name != encoder
    ]
    reference = float(np.mean([responses.rise(n, scope, factor, rows, measure) for n in others]))
    own = responses.rise(encoder, scope, factor, rows, measure)
    positive = own > 0 and reference > 0
    return (own / reference if reference != 0 else float("nan")), positive


@dataclass(frozen=True)
class Reading:
    """One encoder's sensitivity at one scope, with its tilt from the moderate to the large raise.

    Attributes:
        sensitivity: Per factor, the point and the 95 % interval over resampled windows.
        tilt: The sensitivity at ``LARGE`` over that at ``MODERATE``, point and interval, from the
            same resamples: below one, the encoder falls behind the initialisations as the reading
            grows, which a duller response at every size does not do.
        positive: Whether every rise the tilt is made of lay above zero, at the point and in all
            but 2.5 % of the resamples.
    """

    sensitivity: Mapping[float, tuple[float, float, float]]
    tilt: tuple[float, float, float]
    positive: bool


def readings(
    responses: Responses, measure: str, *, seed: int = 1
) -> dict[tuple[str, str], Reading]:
    """Every encoder's reading at every scope, intervals over one draw of resampled windows.

    Raises:
        ValueError: If the windows were not raised by both factors the rule reads, or fewer than
            two initialisations were measured, so none could be read against the others.
    """
    if not {MODERATE, LARGE} <= set(responses.factors):
        raise ValueError(
            f"the rule reads ×{MODERATE:g} and ×{LARGE:g}; raised by {responses.factors}"
        )
    if sum(name.startswith(INITIALISATION) for name in responses.encoders) < 2:
        raise ValueError("an initialisation is read against the others; measure at least two")
    draws = np.random.default_rng(seed)
    count = len(responses.windows)
    resamples = [np.arange(count)] + [draws.integers(0, count, count) for _ in range(RESAMPLES)]
    moderate = responses.factors.index(MODERATE)
    large = responses.factors.index(LARGE)
    found: dict[tuple[str, str], Reading] = {}
    for encoder in responses.encoders:
        for scope in SCOPES:
            read = [
                [
                    sensitivity(responses, encoder, scope, f, rows, measure)
                    for f in responses.factors
                ]
                for rows in resamples
            ]
            values = np.array([[value for value, _ in row] for row in read])
            signs = np.array([[ok for _, ok in row] for row in read])
            ruled = signs[:, [moderate, large]].all(axis=1)
            by_factor = {}
            for column, factor in enumerate(responses.factors):
                low, high = np.nanquantile(values[1:, column], [0.025, 0.975])
                by_factor[factor] = (float(values[0, column]), float(low), float(high))
            tilts = values[:, large] / values[:, moderate]
            low, high = np.nanquantile(tilts[1:], [0.025, 0.975])
            found[(encoder, scope)] = Reading(
                sensitivity=by_factor,
                tilt=(float(tilts[0]), float(low), float(high)),
                positive=bool(ruled[0]) and float(ruled[1:].mean()) >= 0.975,
            )
    return found


def sensitivity_table(responses: Responses, found: Mapping[tuple[str, str], Reading]) -> str:
    """Every encoder's sensitivity and tilt, scope by scope."""
    lines = []
    for (encoder, scope), reading in found.items():
        cells = [
            f"{point:.2f} [{low:.2f}; {high:.2f}]"
            for point, low, high in (reading.sensitivity[f] for f in responses.factors)
        ]
        point, low, high = reading.tilt
        lines.append(
            [
                encoder,
                scope,
                *cells,
                f"{point:.2f} [{low:.2f}; {high:.2f}]",
                "yes" if reading.positive else "no",
            ]
        )
    return table(
        [
            "Encoder",
            "Scope",
            *(f"×{factor:g}" for factor in responses.factors),
            f"Tilt ×{LARGE:g} / ×{MODERATE:g}",
            "Rises above zero",
        ],
        lines,
    )


def verdict(found: Mapping[tuple[str, str], Reading]) -> str:
    """The declared rule over one task and one measure, on the channel scope.

    A pretrained encoder falls when its tilt is at most ``TILT``, its interval below one and its
    point below every initialisation's own; it is duller when its sensitivity at the moderate
    raise is at most ``DULL``, its interval below one and its point below every
    initialisation's. The size is lost when every pretrained encoder falls, the response duller
    when every one is duller. The rule is void when a rise it divides is not above zero, or when
    an initialisation's own tilt is at most ``TILT`` with its interval below one.
    """
    ruled = {e: r for (e, s), r in found.items() if s == RULED}
    drawn = [r for e, r in ruled.items() if e.startswith(INITIALISATION)]
    pretrained = [r for e, r in ruled.items() if not e.startswith(INITIALISATION)]
    if not all(r.positive for r in ruled.values()):
        return f"void on the {RULED} scope: a rise the rule divides is not above zero"
    if any(r.tilt[0] <= TILT and r.tilt[2] < 1.0 for r in drawn):
        return f"void on the {RULED} scope: an initialisation's own tilt falls"
    tilt_floor = min(r.tilt[0] for r in drawn)
    dull_floor = min(r.sensitivity[MODERATE][0] for r in drawn)

    def falls(r: Reading) -> bool:
        return r.tilt[0] <= TILT and r.tilt[2] < 1.0 and r.tilt[0] < tilt_floor

    def duller(r: Reading) -> bool:
        point, _, high = r.sensitivity[MODERATE]
        return point <= DULL and high < 1.0 and point < dull_floor

    lost = "yes" if pretrained and all(falls(r) for r in pretrained) else "no"
    dull = "yes" if pretrained and all(duller(r) for r in pretrained) else "no"
    return f"size lost on the {RULED} scope: {lost}; duller at ×{MODERATE:g}: {dull}"


def rise_table(responses: Responses, measure: str) -> str:
    """Every encoder's median rise at every scope and factor, by ``measure``."""
    every = np.arange(len(responses.windows))
    lines = [
        [
            encoder,
            scope,
            *(
                f"{responses.rise(encoder, scope, factor, every, measure):.3f}"
                for factor in responses.factors
            ),
        ]
        for encoder in responses.encoders
        for scope in SCOPES
    ]
    return table(["Encoder", "Scope", *(f"×{factor:g}" for factor in responses.factors)], lines)


def events_table(directory: Path) -> str:
    """Each encoder's window-level area of events against normal windows, by the largest reading."""
    by: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    with (directory / SCORED).open() as handle:
        for row in csv.DictReader(handle):
            by[encoder_name(row)].append(
                (float(row["target"]), float(row["largest"]), float(row["score"]))
            )
    lines = []
    for encoder, rows in sorted(by.items()):
        values = np.array(rows)
        normal = values[:, 0] == 0
        far = (values[:, 0] > 0) & (values[:, 1] > FAR)
        near = (values[:, 0] > 0) & (values[:, 1] <= FAR)
        cells = [
            f"{auroc(values[group, 2], values[normal, 2]):.3f} (n {int(group.sum())})"
            if group.any()
            else "-"
            for group in (far | near, far, near)
        ]
        lines.append([encoder, *cells])
    return table(["Encoder", "All events", f"Beyond {FAR:g}", f"Within {FAR:g}"], lines)


def reference_table(directory: Path) -> str:
    """How far the validation's normal windows lie from the fitting side, against its own."""
    normal: dict[str, list[float]] = defaultdict(list)
    with (directory / SCORED).open() as handle:
        for row in csv.DictReader(handle):
            if float(row["target"]) == 0:
                normal[encoder_name(row)].append(float(row["distance"]))
    lines = []
    with (directory / GEOMETRY).open() as handle:
        for row in csv.DictReader(handle):
            name = encoder_name(row)
            validation = float(np.median(normal[name]))
            fitting = float(row["fitting_distance"])
            lines.append(
                [
                    name,
                    row["width"],
                    f"{float(row['participation_standardised']):.1f}",
                    f"{float(row['participation_raw']):.1f}",
                    f"{fitting:.3f}",
                    f"{validation:.3f}",
                    f"{validation / fitting:.2f}",
                ]
            )
    return table(
        [
            "Encoder",
            "Width",
            "Directions, standardised",
            "Directions, raw",
            "Fitting distance",
            "Validation normal distance",
            "Ratio",
        ],
        lines,
    )


def check_against_campaign(directory: Path, predictions: Path, campaign: str, label: str) -> str:
    """How closely the scores of ``label`` reproduce the campaign's, seed by seed.

    A pretrained encoder is matched to the campaign's states detector, an initialisation to its
    detector at its initialisation, each by the window's position.
    """
    ours: dict[tuple[str, int, int], float] = {}
    with (directory / SCORED).open() as handle:
        for row in csv.DictReader(handle):
            if row["encoder"] in (label, INITIALISATION):
                candidate = "untrained_knn" if row["encoder"] == INITIALISATION else "frozen_knn"
                ours[(candidate, int(row["seed"]), int(row["position"]))] = float(row["score"])
    theirs: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    with predictions.open() as handle:
        for row in csv.DictReader(handle):
            if row["campaign"] != campaign:
                continue
            candidate = row["candidate"].split("@")[0]
            if candidate in ("untrained_knn", "frozen_knn"):
                theirs[(candidate, int(row["seed"]))][int(row["position"])] = float(
                    row["predicted"]
                )
    lines = []
    for (candidate, seed), by in sorted(theirs.items()):
        if candidate == "frozen_knn" and seed != 1:
            continue
        shared = [p for p in by if (candidate, seed, p) in ours]
        if not shared:
            continue
        a = np.array([ours[(candidate, seed, p)] for p in shared])
        b = np.array([by[p] for p in shared])
        lines.append(
            [
                label,
                candidate,
                str(seed),
                f"{len(shared)} of {len(by)}",
                f"{float(np.max(np.abs(a - b))):.2e}",
                f"{float(np.corrcoef(a, b)[0, 1]):.6f}",
            ]
        )
    return table(
        ["Label", "Candidate", "Seed", "Windows", "Largest difference", "Correlation"], lines
    )


def read(arguments: argparse.Namespace) -> None:
    for directory in arguments.dir:
        conditions = json.loads((directory / CONDITIONS).read_text())
        print(f"## {conditions['task']} ({conditions['normal_windows']} normal windows raised)\n")
        responses = Responses.read(directory)
        for measure in MEASURES:
            declared = " (the declared rule)" if measure == DISPLACEMENT else " (descriptive)"
            print(f"### {measure.capitalize()}{declared}\n")
            print(f"Rise by {MEASURES[measure]}:\n")
            print(rise_table(responses, measure))
            found = readings(responses, measure)
            print("\nSensitivity, rise over the other initialisations' mean rise:\n")
            print(sensitivity_table(responses, found))
            print(f"\n{verdict(found)}\n")
        print("\nEvents against normal windows (window-level AUROC, as the campaign scores):\n")
        print(events_table(directory))
        print("\nThe reference:\n")
        print(reference_table(directory))
        for predictions, campaign, label in arguments.campaign_predictions or ():
            print("\nAgainst the campaign:\n")
            print(check_against_campaign(directory, Path(predictions), campaign, label))
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
    measuring.add_argument("--reference", choices=("window", "unit"), default="window")
    measuring.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    measuring.add_argument("--normal", type=int, default=300)
    measuring.add_argument("--factors", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0])
    measuring.add_argument("--draw-seed", type=int, default=1)
    measuring.add_argument("--batch-size", type=int, default=64)
    measuring.add_argument("--workspace", type=Path, default=Path("data/workspace"))
    measuring.add_argument("--corpora", type=Path, default=Path("data/raw"))
    measuring.add_argument("--device", default=None)
    reading = steps.add_parser("read")
    reading.add_argument("--dir", type=Path, nargs="+", required=True)
    reading.add_argument(
        "--campaign-predictions",
        nargs=3,
        action="append",
        metavar=("PREDICTIONS", "CAMPAIGN", "LABEL"),
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
