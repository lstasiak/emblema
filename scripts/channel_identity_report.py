"""Where the row of a channel lies among the rows a backbone learnt for another corpus's channels.

A backbone gives every channel of every corpus one row of its channel table, and a row fitted
for a channel of a corpus the backbone never read is what let its pretext reach that corpus
(``docs/verification/zero-shot-pretext.md``, 2026-10-17). This script asks what such a row is:
whether it lies nearer the row the backbone learnt for the same physical quantity in another
corpus — the heart rate of one intensive-care corpus beside the heart rate of the other — than
the rows of that corpus's other channels. A map of counterparts names the pairs; for each, the
counterpart's rank by cosine among every row of its corpus, and the cosine itself, in three
readings: the rows as they are; the rows centred by the mean of the candidate corpus's rows,
which takes out the component every learnt row shares; and, for a backbone continued from
stored weights, the fitted row's shift from where the run started it, since a row moved little
from the draw it started at still carries that draw. Every row of a corpus can be listed with
its nearest candidate and the largest cosine it reaches, for a corpus with no counterpart at
all, beside rows drawn as the table draws them, which say what the largest cosine of a row
that means nothing comes to.

A reading of a backbone that read both corpora says whether its own learnt rows put like
quantities together; it is a reference for the fitted rows, not a bound on them.

Measuring and reporting are two steps: every pair, every query and every summary go to CSV,
and the tables are rendered from those files.

    uv run --env-file .env.r2 scripts/channel_identity_report.py --backbone <id>
        --map docs/verification/channel-counterparts.csv --out DIR [--start]
        [--queries physionet2019 --against physionet2012]
    uv run scripts/channel_identity_report.py --report-only DIR
"""

import argparse
import csv
import random
import re
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from uuid import UUID

import numpy as np
import torch
from numpy.typing import NDArray

# Run from anywhere: the sibling script modules live in this directory's package at the repository
# root. The imports below follow, which is why this file is exempt from the import-order rule in
# the lint configuration.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.config.settings import Settings
from emblema.entrypoints.configured import configured_store
from emblema.entrypoints.source_revision import SourceRevision
from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape
from emblema.pretraining.adapters.training.trained_model import CHANNEL_TABLE, TrainedModel
from emblema.pretraining.domain.backbone.backbone import Backbone
from emblema.pretraining.domain.identifiers import BackboneId
from emblema.pretraining.ports.backbone_repository import BackboneRepository
from emblema.shared.ports.artifact_store import ArtifactStore
from scripts.reporting import machine, table

WORKSPACE = REPO_ROOT / "data" / "report" / "channel-identity"
PAIRS = "pairs.csv"
QUERIES = "queries.csv"
RANDOM = "random.csv"
SUMMARY = "summary.csv"
SETTINGS = "settings.csv"
# The readings of a row: as it is, centred by the candidates' mean, and its shift from the start.
READINGS = ("raw", "centred", "shifted")
# A counterpart within the first this many candidates counts as found: the pressures of a corpus
# lie near one another, so the nearest row alone would miss a counterpart that is one of them.
TOP = 3
RESAMPLES = 2_000
PERMUTATIONS = 2_000
RANDOM_ROWS = 1_000
SEED = 1
Vector = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class VocabularyRow:
    """One row of the channel table: the channel it belongs to.

    Attributes:
        channel_id: Identifier the row is indexed by.
        corpus: Corpus of the channel.
        channel: The channel's name within its corpus.
        timeless: Whether the channel is a static feature of its units rather than a measurement;
            such a row is not the identity of a quantity read over time and is left out.
    """

    channel_id: int
    corpus: str
    channel: str
    timeless: bool = False


@dataclass(frozen=True)
class ChannelTable:
    """A backbone's channel table with the vocabulary that indexes it.

    Attributes:
        rows: One row per identifier, padding's zero row first.
        entries: The channel behind each identifier, entry ``i`` carrying ``i + 1``.
    """

    rows: Vector
    entries: tuple[VocabularyRow, ...]

    def __post_init__(self) -> None:
        if self.rows.shape[0] != len(self.entries) + 1:
            raise ValueError(
                f"a table of {self.rows.shape[0]} rows does not index {len(self.entries)} channels"
            )
        for index, entry in enumerate(self.entries):
            if entry.channel_id != index + 1:
                raise ValueError(f"entry {index} must carry id {index + 1}, got {entry.channel_id}")

    def id_of(self, corpus: str, channel: str) -> int:
        """The row of a measured channel.

        Raises:
            ValueError: If the table has no such row, or the channel is a static feature.
        """
        for entry in self.entries:
            if entry.corpus == corpus and entry.channel == channel:
                if entry.timeless:
                    raise ValueError(f"channel {channel!r} of {corpus!r} is a static feature")
                return entry.channel_id
        raise ValueError(f"the table has no row for channel {channel!r} of {corpus!r}")

    def ids_of(self, corpus: str) -> list[int]:
        """The rows of a corpus's measured channels, its static features left out."""
        found = [
            entry.channel_id
            for entry in self.entries
            if entry.corpus == corpus and not entry.timeless
        ]
        if not found:
            raise ValueError(f"the table has no rows of measured channels of corpus {corpus!r}")
        return found

    def entry(self, channel_id: int) -> VocabularyRow:
        return self.entries[channel_id - 1]


@dataclass(frozen=True, kw_only=True)
class Counterpart:
    """One line of the map: a channel and the channel of another corpus it is the quantity of.

    Attributes:
        corpus: Corpus of the channel asked about.
        channel: The channel asked about.
        counterpart_corpus: Corpus whose rows are the candidates.
        counterpart_channel: The candidate that is the same quantity.
        note: Why the pair is doubtful, when it is; empty otherwise.
    """

    corpus: str
    channel: str
    counterpart_corpus: str
    counterpart_channel: str
    note: str = ""


def read_map(path: Path) -> list[Counterpart]:
    """The counterparts a map file names, one per line, with ``note`` blank where absent."""
    with path.open(newline="") as handle:
        return [
            Counterpart(
                corpus=record["corpus"],
                channel=record["channel"],
                counterpart_corpus=record["counterpart_corpus"],
                counterpart_channel=record["counterpart_channel"],
                note=record.get("note") or "",
            )
            for record in csv.DictReader(handle)
        ]


def families_of(table: ChannelTable, corpus: str, pattern: str) -> list[Counterpart]:
    """Every ordered pair of measured channels of ``corpus`` whose names share ``pattern``'s group.

    A corpus can carry one quantity under several conditions as several channels, the sensors
    of the turbofan under each flight condition among them; the group the pattern captures
    names the quantity, and each channel of a family is read against each of the others.

    Raises:
        ValueError: If the pattern captures no group, matches no channel, or every family is one.
    """
    compiled = re.compile(pattern)
    if compiled.groups < 1:
        raise ValueError(f"the family pattern {pattern!r} captures no group")
    families: dict[str, list[str]] = {}
    for channel_id in table.ids_of(corpus):
        entry = table.entry(channel_id)
        found = compiled.fullmatch(entry.channel)
        if found is not None:
            families.setdefault(found.group(1), []).append(entry.channel)
    pairs = [
        Counterpart(
            corpus=corpus,
            channel=channel,
            counterpart_corpus=corpus,
            counterpart_channel=sibling,
            note=family,
        )
        for family, members in families.items()
        for channel in members
        for sibling in members
        if sibling != channel
    ]
    if not pairs:
        raise ValueError(f"the family pattern {pattern!r} gives no family of two in {corpus!r}")
    return pairs


def swapped(pairs: Iterable[Counterpart]) -> list[Counterpart]:
    """The pairs read the other way round: each counterpart asked about among the query's corpus."""
    return [
        Counterpart(
            corpus=pair.counterpart_corpus,
            channel=pair.counterpart_channel,
            counterpart_corpus=pair.corpus,
            counterpart_channel=pair.channel,
            note=pair.note,
        )
        for pair in pairs
    ]


@dataclass(frozen=True, kw_only=True)
class Placement:
    """Where one vector lies among the candidates, by cosine.

    Attributes:
        rank: One plus the number of candidates nearer than the counterpart; ``0`` without one.
        cosine: Cosine to the counterpart; ``nan`` without one.
        nearest: Identifier of the nearest candidate.
        largest: Cosine to the nearest candidate.
    """

    rank: int
    cosine: float
    nearest: int
    largest: float


def cosines(query: Vector, candidates: Vector) -> Vector:
    norms = np.linalg.norm(candidates, axis=1) * np.linalg.norm(query)
    return np.asarray(candidates @ query / np.where(norms == 0.0, np.nan, norms), dtype=np.float64)


def placed(
    query: Vector, candidates: Vector, ids: Sequence[int], counterpart: int | None
) -> Placement:
    """``query`` among the ``candidates`` whose identifiers are ``ids``."""
    found = cosines(query, candidates)
    nearest = int(np.nanargmax(found))
    if counterpart is None:
        return Placement(
            rank=0, cosine=float("nan"), nearest=ids[nearest], largest=float(found[nearest])
        )
    own = found[ids.index(counterpart)]
    return Placement(
        rank=1 + int(np.sum(found > own)),
        cosine=float(own),
        nearest=ids[nearest],
        largest=float(found[nearest]),
    )


@dataclass(frozen=True, kw_only=True)
class RowReadings:
    """One row placed among the candidates under each reading, and the sizes involved.

    Attributes:
        candidates: How many rows the query was placed among.
        readings: The placement under each reading, keyed by its name.
        norm: Length of the query's row.
        start_norm: Length of the row the run started the query at; ``nan`` without a start.
        shift: Length of the query's shift from its start; ``nan`` without a start.
    """

    candidates: int
    readings: dict[str, Placement]
    norm: float
    start_norm: float
    shift: float


@dataclass(frozen=True, kw_only=True)
class PairResult:
    """One counterpart's row read among the rows of the counterpart's corpus."""

    pair: Counterpart
    row: RowReadings


def row_readings(
    table: ChannelTable,
    query_id: int,
    candidate_ids: Sequence[int],
    counterpart: int | None,
    start: ChannelTable | None,
) -> RowReadings:
    """The query's row among the candidates: as it is, centred by their mean, and its shift.

    Raises:
        ValueError: If fewer than two candidates are given, since centring by the mean of one
            leaves nothing to place the query among.
    """
    if len(candidate_ids) < 2:
        raise ValueError(f"a row is placed among at least two candidates, not {len(candidate_ids)}")
    query = table.rows[query_id]
    candidates = table.rows[list(candidate_ids)]
    centre = candidates.mean(axis=0)
    readings = {
        "raw": placed(query, candidates, candidate_ids, counterpart),
        "centred": placed(query - centre, candidates - centre, candidate_ids, counterpart),
    }
    start_norm = shift = float("nan")
    if start is not None:
        began = start.rows[query_id]
        readings["shifted"] = placed(query - began, candidates - centre, candidate_ids, counterpart)
        start_norm = float(np.linalg.norm(began))
        shift = float(np.linalg.norm(query - began))
    return RowReadings(
        candidates=len(candidate_ids),
        readings=readings,
        norm=float(np.linalg.norm(query)),
        start_norm=start_norm,
        shift=shift,
    )


def compare(
    table: ChannelTable, pairs: Iterable[Counterpart], start: ChannelTable | None = None
) -> list[PairResult]:
    """Each counterpart placed among every row of its corpus but the query's own.

    Raises:
        ValueError: If a pair names a channel the table has no row for, or if ``start`` is a
            table of another vocabulary.
    """
    if start is not None and start.entries != table.entries:
        raise ValueError("the start table must index the same vocabulary as the table read")
    results = []
    for pair in pairs:
        query_id = table.id_of(pair.corpus, pair.channel)
        counterpart = table.id_of(pair.counterpart_corpus, pair.counterpart_channel)
        candidate_ids = [one for one in table.ids_of(pair.counterpart_corpus) if one != query_id]
        if counterpart == query_id:
            raise ValueError(f"{pair.corpus}/{pair.channel} is named as its own counterpart")
        results.append(
            PairResult(
                pair=pair, row=row_readings(table, query_id, candidate_ids, counterpart, start)
            )
        )
    return results


@dataclass(frozen=True, kw_only=True)
class QueryResult:
    """One row of a corpus against every row of another: its nearest, and how near."""

    corpus: str
    channel: str
    row: RowReadings


def queries_of(
    table: ChannelTable, corpus: str, against: str, start: ChannelTable | None = None
) -> list[QueryResult]:
    """Every row of ``corpus`` placed among the rows of ``against``, counterpart or none."""
    candidate_ids = table.ids_of(against)
    results = []
    for query_id in table.ids_of(corpus):
        candidates = [one for one in candidate_ids if one != query_id]
        entry = table.entry(query_id)
        results.append(
            QueryResult(
                corpus=entry.corpus,
                channel=entry.channel,
                row=row_readings(table, query_id, candidates, None, start),
            )
        )
    return results


def random_largest(table: ChannelTable, against: str, count: int, seed: int) -> list[float]:
    """The largest centred cosine each of ``count`` rows drawn as the table draws them reaches.

    The channel table draws a row from the standard normal, which is what a row no pretraining
    set is; it is read as a query under the centred reading, itself less the candidates' mean.
    """
    candidates = table.rows[table.ids_of(against)]
    centre = candidates.mean(axis=0)
    generator = torch.Generator().manual_seed(seed)
    drawn = torch.randn(count, candidates.shape[1], generator=generator, dtype=torch.float64)
    return [float(np.nanmax(cosines(row - centre, candidates - centre))) for row in drawn.numpy()]


@dataclass(frozen=True, kw_only=True)
class Summary:
    """The pairs under one reading: how many counterparts are found, and how surely.

    Attributes:
        reading: The reading summarised.
        pairs: Counterparts read.
        candidates: Candidates a counterpart was placed among, the same for every pair or the
            median where they differ.
        found: Share of pairs whose counterpart ranks within the first ``TOP``.
        found_low: Lower edge of the share's bootstrap interval over pairs.
        found_high: Upper edge.
        median_rank: Median rank of the counterpart.
        rank_low: Lower edge of the median rank's bootstrap interval.
        rank_high: Upper edge.
        cosine: Mean cosine to the counterpart.
        cosine_low: Lower edge of the mean cosine's bootstrap interval.
        cosine_high: Upper edge.
        p_value: Share of permutations, each placing every counterpart at a random candidate,
            that find at least as many as observed; one added to both counts.
    """

    reading: str
    pairs: int
    candidates: int
    found: float
    found_low: float
    found_high: float
    median_rank: float
    rank_low: float
    rank_high: float
    cosine: float
    cosine_low: float
    cosine_high: float
    p_value: float


def summarise(
    results: Sequence[PairResult],
    *,
    resamples: int = RESAMPLES,
    permutations: int = PERMUTATIONS,
    seed: int = SEED,
) -> list[Summary]:
    """One summary per reading the results carry.

    The interval is a percentile bootstrap over pairs; the permutation draws every pair's rank
    uniformly from its candidates, which is the rank of a counterpart that is a random candidate.
    """
    if not results:
        return []
    draws = random.Random(seed)
    summaries = []
    for reading in READINGS:
        if any(reading not in result.row.readings for result in results):
            continue
        ranks = [result.row.readings[reading].rank for result in results]
        cosine_values = [result.row.readings[reading].cosine for result in results]
        counts = [result.row.candidates for result in results]
        found = _share_found(ranks)
        shares, medians, means = [], [], []
        for _ in range(resamples):
            chosen = [draws.randrange(len(results)) for _ in results]
            shares.append(_share_found([ranks[i] for i in chosen]))
            medians.append(statistics.median(ranks[i] for i in chosen))
            means.append(statistics.fmean(cosine_values[i] for i in chosen))
        at_least = sum(
            _share_found([draws.randint(1, count) for count in counts]) >= found
            for _ in range(permutations)
        )
        summaries.append(
            Summary(
                reading=reading,
                pairs=len(results),
                candidates=int(statistics.median(counts)),
                found=found,
                found_low=_quantile(shares, 0.025),
                found_high=_quantile(shares, 0.975),
                median_rank=statistics.median(ranks),
                rank_low=_quantile(medians, 0.025),
                rank_high=_quantile(medians, 0.975),
                cosine=statistics.fmean(cosine_values),
                cosine_low=_quantile(means, 0.025),
                cosine_high=_quantile(means, 0.975),
                p_value=(1 + at_least) / (1 + permutations),
            )
        )
    return summaries


def _share_found(ranks: Sequence[int]) -> float:
    return sum(rank <= TOP for rank in ranks) / len(ranks)


def _quantile(values: Sequence[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def _number(value: float) -> str:
    return "nan" if value != value else f"{value:.6g}"


PAIR_FIELDS = (
    "corpus",
    "channel",
    "counterpart_corpus",
    "counterpart_channel",
    "note",
    "candidates",
    "norm",
    "start_norm",
    "shift",
    *(
        f"{reading}_{field}"
        for reading in READINGS
        for field in ("rank", "cosine", "nearest", "largest")
    ),
)
QUERY_FIELDS = (
    "corpus",
    "channel",
    "norm",
    "start_norm",
    "shift",
    *(f"{reading}_{field}" for reading in READINGS for field in ("nearest", "largest")),
)


def _placement_record(
    table: ChannelTable, readings: dict[str, Placement], with_rank: bool
) -> dict[str, str]:
    record = {}
    for reading in READINGS:
        placement = readings.get(reading)
        if with_rank:
            record[f"{reading}_rank"] = "" if placement is None else str(placement.rank)
            record[f"{reading}_cosine"] = "" if placement is None else _number(placement.cosine)
        record[f"{reading}_nearest"] = (
            "" if placement is None else table.entry(placement.nearest).channel
        )
        record[f"{reading}_largest"] = "" if placement is None else _number(placement.largest)
    return record


def write_pairs(directory: Path, table: ChannelTable, results: Iterable[PairResult]) -> None:
    records = [
        {
            "corpus": result.pair.corpus,
            "channel": result.pair.channel,
            "counterpart_corpus": result.pair.counterpart_corpus,
            "counterpart_channel": result.pair.counterpart_channel,
            "note": result.pair.note,
            "candidates": str(result.row.candidates),
            "norm": _number(result.row.norm),
            "start_norm": _number(result.row.start_norm),
            "shift": _number(result.row.shift),
            **_placement_record(table, result.row.readings, with_rank=True),
        }
        for result in results
    ]
    _write(directory / PAIRS, PAIR_FIELDS, records)


def write_queries(directory: Path, table: ChannelTable, results: Iterable[QueryResult]) -> None:
    records = [
        {
            "corpus": result.corpus,
            "channel": result.channel,
            "norm": _number(result.row.norm),
            "start_norm": _number(result.row.start_norm),
            "shift": _number(result.row.shift),
            **_placement_record(table, result.row.readings, with_rank=False),
        }
        for result in results
    ]
    _write(directory / QUERIES, QUERY_FIELDS, records)


def write_random(directory: Path, largest: Iterable[float]) -> None:
    _write(directory / RANDOM, ("largest",), [{"largest": _number(one)} for one in largest])


def write_summary(directory: Path, summaries: Iterable[Summary]) -> None:
    names = [field.name for field in fields(Summary)]
    _write(
        directory / SUMMARY,
        names,
        [
            {
                name: (value if isinstance(value, str) else _number(float(value)))
                for name, value in ((f.name, getattr(summary, f.name)) for f in fields(Summary))
            }
            for summary in summaries
        ],
    )


def write_settings(directory: Path, settings: dict[str, str]) -> None:
    _write(
        directory / SETTINGS,
        ("setting", "value"),
        [{"setting": key, "value": value} for key, value in settings.items()],
    )


def _write(path: Path, names: Sequence[str], records: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(names), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_summary(directory: Path) -> list[Summary]:
    return [
        Summary(
            reading=record["reading"],
            pairs=int(record["pairs"]),
            candidates=int(record["candidates"]),
            **{
                name: float(record[name])
                for name in (f.name for f in fields(Summary))
                if name not in ("reading", "pairs", "candidates")
            },
        )
        for record in read_rows(directory / SUMMARY)
    ]


def render(directory: Path) -> str:
    """The tables a note pastes, from the files a run stored."""
    settings = {record["setting"]: record["value"] for record in read_rows(directory / SETTINGS)}
    parts = [f"Backbone {settings['backbone']}, weights `sha256:{settings['weights'][:8]}…`."]
    summary_path = directory / SUMMARY
    if summary_path.exists():
        summaries = read_summary(directory)
        parts.append(
            table(
                (
                    "Reading",
                    "Pairs",
                    "Candidates",
                    f"Found within {TOP}",
                    "Median rank",
                    "Cosine to counterpart",
                    "p",
                ),
                [
                    (
                        one.reading,
                        str(one.pairs),
                        str(one.candidates),
                        f"{one.found:.2f} [{one.found_low:.2f}; {one.found_high:.2f}]",
                        f"{one.median_rank:.1f} [{one.rank_low:.1f}; {one.rank_high:.1f}]",
                        f"{one.cosine:+.3f} [{one.cosine_low:+.3f}; {one.cosine_high:+.3f}]",
                        f"{one.p_value:.3f}",
                    )
                    for one in summaries
                ],
            )
        )
        readings = [one.reading for one in summaries]
        parts.append(
            table(
                (
                    "Channel",
                    "Counterpart",
                    "Note",
                    *(f"{r}: rank, cosine, nearest" for r in readings),
                    "Shift / start",
                ),
                [
                    (
                        f"{record['corpus']}/{record['channel']}",
                        record["counterpart_channel"],
                        record["note"],
                        *(_pair_text(record, r) for r in readings),
                        _shift_text(record),
                    )
                    for record in read_rows(directory / PAIRS)
                ],
            )
        )
    queries_path = directory / QUERIES
    if queries_path.exists():
        records = read_rows(queries_path)
        readings = [r for r in READINGS if records and records[0][f"{r}_largest"]]
        parts.append(
            table(
                ("Channel", *(f"{r}: nearest, largest" for r in readings), "Shift / start"),
                [
                    (
                        f"{record['corpus']}/{record['channel']}",
                        *(
                            f"{record[f'{r}_nearest']}, {float(record[f'{r}_largest']):+.2f}"
                            for r in readings
                        ),
                        _shift_text(record),
                    )
                    for record in records
                ],
            )
        )
    random_path = directory / RANDOM
    if random_path.exists():
        largest = sorted(float(record["largest"]) for record in read_rows(random_path))
        parts.append(
            f"Largest centred cosine of {len(largest)} rows drawn as the table draws them: median "
            f"{_quantile(largest, 0.5):+.3f}, 95th percentile {_quantile(largest, 0.95):+.3f}, "
            f"largest {largest[-1]:+.3f}."
        )
    return "\n\n".join(parts)


def _pair_text(record: dict[str, str], reading: str) -> str:
    cosine = float(record[f"{reading}_cosine"])
    return f"{record[f'{reading}_rank']}, {cosine:+.2f}, {record[f'{reading}_nearest']}"


def _shift_text(record: dict[str, str]) -> str:
    if record["shift"] == "nan":
        return "—"
    return f"{float(record['shift']):.2f} / {float(record['start_norm']):.2f}"


def table_of(backbone: Backbone, store: ArtifactStore) -> ChannelTable:
    """The backbone's channel table under the vocabulary its last input published.

    A published manifest carries the whole vocabulary its chain had reached, so the input with
    the largest vocabulary names every row of the table.
    """
    if backbone.artifact is None:
        raise ValueError(f"backbone {backbone.id} is not ready: it has no weights")
    stored = TrainedModel.read(store.get(backbone.artifact))
    return ChannelTable(_table_rows(stored.weights), _entries_of(backbone, store))


def start_table_of(backbone: Backbone, store: ArtifactStore) -> ChannelTable:
    """The channel table the run started from, drawn from its seed with its parent carried in.

    Raises:
        ValueError: If the backbone was drawn from its seed, so its start is its initialisation,
            which is not a row anything fitted moved from.
    """
    configuration = backbone.configuration
    if configuration.start is None:
        raise ValueError(f"backbone {backbone.id} started from its seed, not from stored weights")
    vocabulary_size = max(stated.vocabulary_size for stated in backbone.inputs)
    torch.manual_seed(configuration.budget.seed)
    model = ObjectiveShape.of(configuration, vocabulary_size).build(dropout=configuration.dropout)
    TrainedModel.read(store.get(configuration.start)).carried_into(model)
    return ChannelTable(_table_rows(model.state_dict()), _entries_of(backbone, store))


def _table_rows(weights: dict[str, torch.Tensor]) -> Vector:
    for key, value in weights.items():
        if key.endswith(CHANNEL_TABLE):
            return np.asarray(value.detach().to("cpu").to(torch.float64).numpy(), dtype=np.float64)
    raise ValueError(f"the weights hold no tensor ending in {CHANNEL_TABLE!r}")


def _entries_of(backbone: Backbone, store: ArtifactStore) -> tuple[VocabularyRow, ...]:
    widest = max(backbone.inputs, key=lambda stated: stated.vocabulary_size)
    manifest = PublishedCorpusManifestJson().decode(store.get(widest.manifest))
    return tuple(
        VocabularyRow(
            channel_id=channel.channel_id,
            corpus=channel.corpus,
            channel=channel.channel,
            timeless=channel.timeless,
        )
        for channel in manifest.channels
    )


def _registry() -> BackboneRepository:  # pragma: no cover - environment
    from emblema.entrypoints.configured import configured_engine
    from emblema.pretraining.adapters.persistence.backbone_repository import (
        SqlAlchemyBackboneRepository,
    )

    return SqlAlchemyBackboneRepository(configured_engine(Settings()))


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", metavar="ID", help="a backbone the registry holds ready")
    parser.add_argument("--map", type=Path, metavar="FILE", help="the counterparts, one per line")
    parser.add_argument("--swap", action="store_true", help="read the map the other way round")
    parser.add_argument(
        "--family",
        metavar="CORPUS:PATTERN",
        help="counterparts within one corpus: channels whose names share the pattern's group",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="read each row's shift from where the run started it, for a continued backbone",
    )
    parser.add_argument(
        "--queries", metavar="CORPUS", help="list every row of this corpus with its nearest"
    )
    parser.add_argument(
        "--against", metavar="CORPUS", help="the corpus whose rows --queries and --random meet"
    )
    parser.add_argument(
        "--random",
        type=int,
        metavar="ROWS",
        help=f"rows drawn as the table draws them, read against --against; {RANDOM_ROWS} with it",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--report-only", type=Path, metavar="DIR", help="render a stored reading")
    parser.add_argument(
        "--out",
        type=Path,
        help="where the files go; data/report/channel-identity/<backbone> unless given",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    if arguments.report_only is not None:
        print(render(arguments.report_only))
        return
    if arguments.backbone is None:
        raise SystemExit("--backbone is needed to read anything")
    if arguments.map is None and arguments.family is None and arguments.queries is None:
        raise SystemExit("--map, --family or --queries is needed: nothing to place")
    if arguments.family is not None and ":" not in arguments.family:
        raise SystemExit("--family takes CORPUS:PATTERN")
    if (arguments.queries is not None or arguments.random) and arguments.against is None:
        raise SystemExit("--queries and --random need --against, the corpus whose rows they meet")
    random_rows = arguments.random
    if random_rows is None:
        random_rows = RANDOM_ROWS if arguments.against is not None else 0
    try:
        backbone_id = BackboneId(UUID(arguments.backbone))
    except ValueError as error:
        raise SystemExit(f"--backbone must be a backbone's id: {error}") from error
    registry = _registry()
    backbone = registry.get(backbone_id)
    store = configured_store(Settings())
    try:
        read = table_of(backbone, store)
        start = start_table_of(backbone, store) if arguments.start else None
        pairs = read_map(arguments.map) if arguments.map is not None else []
        if arguments.swap:
            pairs = swapped(pairs)
        if arguments.family is not None:
            corpus, pattern = arguments.family.split(":", 1)
            pairs += families_of(read, corpus, pattern)
        results = compare(read, pairs, start)
        queries = (
            queries_of(read, arguments.queries, arguments.against, start)
            if arguments.queries is not None
            else []
        )
        largest = (
            random_largest(read, arguments.against, random_rows, arguments.seed)
            if random_rows
            else []
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    directory = arguments.out or WORKSPACE / str(backbone.id)
    settings = {
        "backbone": str(backbone.id),
        "weights": backbone.artifact.checksum.digest if backbone.artifact else "",
        "experiment": backbone.configuration.name,
        "parent": str(backbone.parent) if backbone.parent is not None else "",
        "start": "true" if start is not None else "false",
        "map": str(arguments.map) if arguments.map is not None else "",
        "swap": "true" if arguments.swap else "false",
        "family": arguments.family or "",
        "queries": arguments.queries or "",
        "against": arguments.against or "",
        "random": str(random_rows),
        "top": str(TOP),
        "resamples": str(arguments.resamples),
        "permutations": str(arguments.permutations),
        "seed": str(arguments.seed),
        "machine": machine(),
        "torch": version("torch"),
        "revision": SourceRevision().current(),
        "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
    }
    write_settings(directory, settings)
    if results:
        write_pairs(directory, read, results)
        write_summary(
            directory,
            summarise(
                results,
                resamples=arguments.resamples,
                permutations=arguments.permutations,
                seed=arguments.seed,
            ),
        )
    if queries:
        write_queries(directory, read, queries)
    if largest:
        write_random(directory, largest)
    print(render(directory))
    print(f"\nstored under {directory}")


if __name__ == "__main__":
    main()
