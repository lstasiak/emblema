import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingClassifier

from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
from emblema.evaluation.adapters.detection.standardisation import Standardisation
from emblema.evaluation.adapters.detection.taught_rows import TaughtRows
from emblema.evaluation.domain.exceptions import UnreadableFittedCandidateError

_LEAF = -1


@dataclass(frozen=True)
class TreeArrays:
    """Boosted trees as the arrays of their nodes, every tree's nodes after the last tree's.

    The form the trees are kept and read in, so the document outlives the release of the
    library that grew them: a node's children, the feature and threshold it splits on and the
    value a leaf adds, as the library exposes them, with the children already offset into the
    one array. A row's log-odds is the learning rate times the sum, over the trees, of the leaf
    it falls to.

    Attributes:
        offsets: ``[trees + 1]``: where each tree's nodes start, and where the last tree's end.
        left: ``[nodes]``: index of each node's left child, ``-1`` at a leaf.
        right: ``[nodes]``: index of each node's right child, ``-1`` at a leaf.
        feature: ``[nodes]``: the feature a node splits on; a row at or below the threshold goes
            left.
        threshold: ``[nodes]``: where the split falls.
        value: ``[nodes]``: what a leaf adds to the log-odds.
        learning_rate: The share of each tree's correction that is kept.
    """

    offsets: NDArray[np.int64]
    left: NDArray[np.int64]
    right: NDArray[np.int64]
    feature: NDArray[np.int64]
    threshold: NDArray[np.float64]
    value: NDArray[np.float64]
    learning_rate: float

    @classmethod
    def of(cls, model: GradientBoostingClassifier) -> Self:
        """The arrays of ``model``'s trees, grown for one outcome."""
        trees = [estimator.tree_ for estimator in model.estimators_[:, 0]]
        counts = np.array([tree.node_count for tree in trees], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        children = [
            cls._offset(np.asarray(getattr(tree, side)), int(at))
            for tree, at in zip(trees, offsets[:-1], strict=True)
            for side in ("children_left", "children_right")
        ]
        return cls(
            offsets=offsets,
            left=np.concatenate(children[0::2]).astype(np.int64),
            right=np.concatenate(children[1::2]).astype(np.int64),
            feature=np.concatenate([np.asarray(tree.feature) for tree in trees]).astype(np.int64),
            threshold=np.concatenate([np.asarray(tree.threshold) for tree in trees]).astype(
                np.float64
            ),
            value=np.concatenate([np.asarray(tree.value)[:, 0, 0] for tree in trees]).astype(
                np.float64
            ),
            learning_rate=float(model.learning_rate),
        )

    @staticmethod
    def _offset(children: NDArray[np.int64], at: int) -> NDArray[np.int64]:
        """A tree's children moved to where its nodes lie in the one array; a leaf stays a leaf."""
        moved: NDArray[np.int64] = np.where(children == _LEAF, _LEAF, children + at)
        return moved

    def log_odds(self, rows: NDArray[np.float64]) -> NDArray[np.float64]:
        """What the trees add up to for every row: the raw prediction before any probability.

        The rows are read in single precision, as the library grew and reads the trees, so a
        value at a threshold falls to the leaf the library sends it to.
        """
        reduced = np.asarray(rows, dtype=np.float32)
        total = np.zeros(len(rows), dtype=np.float64)
        places = np.arange(len(rows))
        for start in self.offsets[:-1]:
            node = np.full(len(rows), start, dtype=np.int64)
            inside = self.left[node] != _LEAF
            while inside.any():
                at = node[inside]
                goes_left = reduced[places[inside], self.feature[at]] <= self.threshold[at]
                node[inside] = np.where(goes_left, self.left[at], self.right[at])
                inside = self.left[node] != _LEAF
            total += self.value[node]
        return total * self.learning_rate


@dataclass(frozen=True)
class TaughtTreesDetector:
    """A detector taught a few events, scoring a row by trees told the events apart from the rest.

    The field's comparator with the same labels: gradient-boosted trees over the same
    standardised features, the taught rows against every other fitting row, which stays
    unlabelled whatever events it holds unknowingly. Each taught event weighs as much as every
    other, over its windows, and the taught side together weighs as much as the rest, so the
    trees learn the events rather than their rarity. The trees are grown by one rule, stated
    here and never chosen on the scored side, from a seed, and a row's score is the log-odds
    they answer; the alarm is set over every fitting row's own score and read against a unit as
    the other detectors' are.

    Attributes:
        standardisation: How every row is centred and scaled, from the rows outside the events.
        trees: The trees, as the arrays of their nodes.
        alarm: Where a score raises an alarm.
    """

    FORMAT: ClassVar[str] = "taught-trees-npz"
    ROUNDS: ClassVar[int] = 100
    DEPTH: ClassVar[int] = 3
    LEARNING_RATE: ClassVar[float] = 0.1

    standardisation: Standardisation
    trees: TreeArrays
    alarm: ScoreAlarm

    @classmethod
    def fitted_on(
        cls,
        rows: NDArray[np.float64],
        taught: TaughtRows,
        *,
        seed: int,
        contamination: float,
        units: Sequence[str] | None = None,
    ) -> Self:
        """The detector fitted on ``rows``, one per fitting window, taught the rows ``taught`` name.

        Raises:
            InvalidTaughtEventsError: If a taught row lies past the rows.
            ValueError: If no row lies outside the taught events, or ``units`` does not name one
                unit per row.
            InvalidScoredOutcomeError: If units are given and none holds two rows.
        """
        rows = np.asarray(rows, dtype=np.float64)
        is_taught = taught.mask(len(rows))
        if is_taught.all():
            raise ValueError("a detector taught events needs a fitting row outside them")
        standardisation = Standardisation.fitted_on(rows[~is_taught])
        standardised = standardisation.of(rows)
        weights = np.ones(len(rows), dtype=np.float64)
        weights[list(taught.positions)] = taught.weights() * (
            float((~is_taught).sum()) / taught.event_count
        )
        model = GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=cls.ROUNDS,
            learning_rate=cls.LEARNING_RATE,
            max_depth=cls.DEPTH,
            max_features="sqrt",
            init="zero",
            random_state=seed,
        )
        model.fit(standardised, is_taught.astype(np.int64), sample_weight=weights)
        trees = TreeArrays.of(model)
        return cls(
            standardisation=standardisation,
            trees=trees,
            alarm=ScoreAlarm.set_over(
                trees.log_odds(standardised), contamination=contamination, units=units
            ),
        )

    def scores(
        self, rows: NDArray[np.float64], *, units: Sequence[str] | None = None
    ) -> NDArray[np.float64]:
        """The log-odds above the alarm of every row, in the order given.

        Raises:
            ValueError: If the rows come with units and the detector reads none, or the other
                way round, or ``units`` does not name one unit per row.
            InvalidScoredOutcomeError: If units are given and none holds two rows.
        """
        self.alarm.require(units)
        return self.alarm.above(self.trees.log_odds(self.standardisation.of(rows)), units=units)

    def kept(self) -> RepresentationBytes:
        """The detector as it is kept, under the format ``from_bytes`` reads back."""
        return RepresentationBytes(self.FORMAT, self.to_bytes())

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez(
            buffer,
            centre=self.standardisation.centre,
            scale=self.standardisation.scale,
            offsets=self.trees.offsets,
            left=self.trees.left,
            right=self.trees.right,
            feature=self.trees.feature,
            threshold=self.trees.threshold,
            value=self.trees.value,
            learning_rate=np.asarray(self.trees.learning_rate),
            alarm=np.asarray(self.alarm.level),
            per_unit=np.asarray(self.alarm.per_unit),
            unit_spread=np.asarray(self.alarm.unit_spread),
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, content: bytes) -> Self:
        """The detector kept as ``content``.

        Raises:
            UnreadableFittedCandidateError: If the bytes are not an archive this reads.
        """
        try:
            with np.load(io.BytesIO(content), allow_pickle=False) as archive:
                return cls(
                    standardisation=Standardisation(
                        centre=np.asarray(archive["centre"], dtype=np.float64),
                        scale=np.asarray(archive["scale"], dtype=np.float64),
                    ),
                    trees=TreeArrays(
                        offsets=np.asarray(archive["offsets"], dtype=np.int64),
                        left=np.asarray(archive["left"], dtype=np.int64),
                        right=np.asarray(archive["right"], dtype=np.int64),
                        feature=np.asarray(archive["feature"], dtype=np.int64),
                        threshold=np.asarray(archive["threshold"], dtype=np.float64),
                        value=np.asarray(archive["value"], dtype=np.float64),
                        learning_rate=float(archive["learning_rate"]),
                    ),
                    alarm=ScoreAlarm(
                        level=float(archive["alarm"]),
                        per_unit=bool(archive["per_unit"]),
                        unit_spread=float(archive["unit_spread"]),
                    ),
                )
        except (KeyError, ValueError, OSError, EOFError) as error:
            raise UnreadableFittedCandidateError(
                "the bytes are not a taught trees detector's archive"
            ) from error
