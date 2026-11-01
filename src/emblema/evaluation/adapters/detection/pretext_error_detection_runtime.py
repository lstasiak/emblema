import io
import logging
import time
from collections.abc import Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.blocks.published_corpus_blocks import PublishedCorpusBlocks
from emblema.evaluation.adapters.blocks.read_corpus import ReadCorpus
from emblema.evaluation.adapters.detection.channel_error_scales import ChannelErrorScales
from emblema.evaluation.adapters.detection.channel_errors import ChannelErrors
from emblema.evaluation.adapters.detection.hidden_cells import HiddenCells
from emblema.evaluation.adapters.detection.linear_cell_answer import LinearCellAnswer
from emblema.evaluation.adapters.detection.masked_cells import MaskedCells
from emblema.evaluation.adapters.detection.pretext_source import PretextSource
from emblema.evaluation.adapters.detection.score_alarm import ScoreAlarm
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.detection.detector_features import DetectorFeatures
from emblema.evaluation.domain.detection.detector_recipe import DetectorRecipe
from emblema.evaluation.domain.detection.error_normalisation import ErrorNormalisation
from emblema.evaluation.domain.detection.pretext_asking import PretextAsking
from emblema.evaluation.domain.detection.score_reference import ScoreReference
from emblema.evaluation.domain.exceptions import (
    CandidateNotRetainableError,
    InvalidScoredOutcomeError,
    UnreadablePretextError,
    UnsupportedDetectorFeaturesError,
)
from emblema.evaluation.domain.labels.labelled_window import LabelledWindow
from emblema.evaluation.domain.labels.task_window import TaskWindow
from emblema.evaluation.domain.scoring.scored_outcome import ScoredOutcome
from emblema.evaluation.domain.scoring.window_prediction import WindowPrediction
from emblema.evaluation.domain.task.downstream_task import DownstreamTask
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.ordering import seeded_rank
from emblema.shared.ports.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)

# Windows asked of the backbone at once; each is asked once per draw in the same call.
_CHUNK = 32

type _Answerer = Callable[[HiddenCells, MaskedCells], NDArray[np.float64]]


class PretextErrorDetectionRuntime:
    """Scores a window by the error of what is filled in where a backbone's masks hide its cells.

    Every window, fitting and scored alike, is hidden ``draws`` times under the backbone's own
    masks, each draw seeded by the cell's seed and the window's unit, position and draw, or once
    in its tail, so the pretrained backbone and the answers that learnt nothing fill in the same
    cells. A window's error is the backbone's loss on the five values in normalised units over the
    hidden timed cells that held a reading in all its draws: one mean over them, or, read per
    channel, the mean over the channels of each channel's error over its typical error on the
    fitting side. A
    window without such a cell has no error to read, so it scores the fitting side's median
    error, neither an alarm nor the most normal window there is, and how many did is logged per
    side. The alarm is the one every detector sets.

    The answers: the backbone's decoder; a ridge per channel over the same visible cells, fitted
    on the fitting side's draws; or each channel's line between its visible bins. What a fit
    keeps, where asked, is its alarm and the ridge's weights; the backbone is the campaign's.
    """

    FORMAT = "pretext-error-npz"

    FEATURES = frozenset(
        {
            DetectorFeatures.PRETEXT_ERROR,
            DetectorFeatures.LINEAR_ANSWER_ERROR,
            DetectorFeatures.INTERPOLATION_ERROR,
        }
    )

    def __init__(
        self,
        source: PretextSource,
        blocks: PublishedCorpusBlocks,
        *,
        device: str = "cpu",
        store: ArtifactStore | None = None,
    ) -> None:
        self._source = source
        self._blocks = blocks
        self._device = device
        self._kept_candidates = None if store is None else KeptCandidates(store)

    def fit(
        self,
        recipe: DetectorRecipe,
        task: DownstreamTask,
        fitting: Sequence[TaskWindow],
        scored: Sequence[LabelledWindow],
        *,
        retain: bool,
    ) -> ScoredOutcome:
        if recipe.features not in self.FEATURES:
            raise UnsupportedDetectorFeaturesError(
                f"this runtime asks a backbone's pretext, not {recipe.features}"
            )
        if retain and self._kept_candidates is None:
            raise CandidateNotRetainableError(
                "this runtime was asked to keep what it fitted and was given no store"
            )
        if not fitting:
            raise InvalidScoredOutcomeError("a detector needs a window to fit on")
        if not scored:
            raise InvalidScoredOutcomeError("there is no window to answer")
        if self._source.bins == 0:
            raise UnreadablePretextError("the backbone reads one token per reading, not cells")
        asking = recipe.asked()
        read = ReadCorpus.every(self._blocks, (task,))[task.manifest]
        started = time.perf_counter()
        answerer, ridge = self._answerer(recipe, asking, read, fitting)
        windows = [labelled.window for labelled in scored]
        per_unit = recipe.reference is ScoreReference.OWN_UNIT
        channels = self._channels(read)
        fitting_errors = self._errors(
            read, fitting, recipe, asking, answerer, channels, side="fitting"
        )
        scales = (
            ChannelErrorScales.fitted_on(fitting_errors)
            if recipe.normalisation is ErrorNormalisation.PER_CHANNEL
            else None
        )
        fitted = fitting_errors.raw() if scales is None else scales.read(fitting_errors)
        unread = fitting_errors.unread
        neutral = float(np.median(fitted[~unread])) if (~unread).any() else 0.0
        fitted[unread] = neutral
        scored_errors = self._errors(
            read, windows, recipe, asking, answerer, channels, side="scored"
        )
        errors = scored_errors.raw() if scales is None else scales.read(scored_errors)
        errors[scored_errors.unread] = neutral
        alarm = ScoreAlarm.set_over(
            fitted,
            contamination=recipe.contamination,
            units=[str(window.unit) for window in fitting] if per_unit else None,
        )
        answers = alarm.above(
            errors, units=[str(window.unit) for window in windows] if per_unit else None
        )
        return ScoredOutcome(
            predictions=tuple(
                WindowPrediction(
                    window=labelled.window, target=labelled.target, predicted=float(answer)
                )
                for labelled, answer in zip(scored, answers.tolist(), strict=True)
            ),
            seconds=time.perf_counter() - started,
            artifact=self._kept(alarm, ridge, scales, task) if retain else None,
        )

    @staticmethod
    def _channels(read: ReadCorpus) -> NDArray[np.int64]:
        """The corpus's timed channels, the columns every window's errors are summed into."""
        manifest = read.manifest
        return np.array(
            sorted(
                channel.channel_id
                for channel in manifest.channels
                if channel.corpus == manifest.corpus and not channel.timeless
            ),
            dtype=np.int64,
        )

    def _kept(
        self,
        alarm: ScoreAlarm,
        ridge: LinearCellAnswer | None,
        scales: ChannelErrorScales | None,
        task: DownstreamTask,
    ) -> ArtifactRef:
        """The alarm, the ridge and the channels' scales this fit produced, stored for the campaign.

        Raises:
            CandidateNotRetainableError: If the runtime was given nowhere to keep them.
        """
        if self._kept_candidates is None:  # pragma: no cover - refused before the fit
            raise CandidateNotRetainableError("this runtime was given no store")
        buffer = io.BytesIO()
        weights = {} if ridge is None else ridge.weights
        arrays: dict[str, NDArray[np.generic]] = {
            "alarm": np.asarray(alarm.level),
            "per_unit": np.asarray(alarm.per_unit),
            "unit_spread": np.asarray(alarm.unit_spread),
            # The columns a ridge's features stand for and the bins, which rebuild them.
            "columns": np.asarray(() if ridge is None else ridge.channels, dtype=np.int64),
            "bins": np.asarray(self._source.bins),
            "channels": np.asarray(sorted(weights), dtype=np.int64),
        }
        arrays.update({f"weights_{channel}": weights[channel] for channel in sorted(weights)})
        if scales is not None:
            arrays["scale_channels"] = scales.channels
            arrays["scales"] = scales.scales
        np.savez(buffer, *(), **arrays)  # type: ignore[arg-type]  # numpy's stub types one keyword
        return self._kept_candidates.keep(
            CandidateKind.DETECTOR,
            corpus_manifest=task.manifest,
            measured=RepresentationBytes(self.FORMAT, buffer.getvalue()),
        )

    def _answerer(
        self,
        recipe: DetectorRecipe,
        asking: PretextAsking,
        read: ReadCorpus,
        fitting: Sequence[TaskWindow],
    ) -> tuple[_Answerer, LinearCellAnswer | None]:
        if recipe.features is DetectorFeatures.PRETEXT_ERROR:
            return self._decoded, None
        if recipe.features is DetectorFeatures.INTERPOLATION_ERROR:
            return (lambda cells, _: cells.interpolation()), None
        ridge = LinearCellAnswer.fitted(
            self._channels(read).tolist(),
            self._source.bins,
            (
                (hidden, self._half(recipe.seed, fitting[at]))
                for _, hidden, at in self._drawn(read, fitting, recipe, asking, answered=False)
            ),
        )
        return (lambda cells, _: ridge.answers(cells)), ridge

    @staticmethod
    def _decoded(cells: HiddenCells, masked: MaskedCells) -> NDArray[np.float64]:
        if masked.answers is None:
            raise UnreadablePretextError("the backbone was asked and gave no answers")
        timed = ~masked.timeless
        bins, width = cells.scored.shape
        return masked.answers[timed].reshape(bins, width, 5)

    def _errors(
        self,
        read: ReadCorpus,
        windows: Sequence[TaskWindow],
        recipe: DetectorRecipe,
        asking: PretextAsking,
        answerer: _Answerer,
        channels: NDArray[np.int64],
        *,
        side: str,
    ) -> ChannelErrors:
        """Every window's loss over its draws' scored cells, summed per channel, in the order given.

        A window lays out the channels it holds, so its grid's columns are looked up among the
        corpus's ``channels``.

        Raises:
            UnreadablePretextError: If a window holds a channel the corpus does not name.
        """
        column_of = np.full(int(channels.max()) + 1 if channels.size else 1, -1, dtype=np.int64)
        column_of[channels] = np.arange(len(channels))
        missed = np.zeros((len(windows), len(channels)))
        counted = np.zeros((len(windows), len(channels)), dtype=np.int64)
        answered = recipe.features is DetectorFeatures.PRETEXT_ERROR
        for masked, hidden, at in self._drawn(read, windows, recipe, asking, answered=answered):
            if not hidden.scored.any():
                continue
            if hidden.channels.max() >= len(column_of) or (column_of[hidden.channels] < 0).any():
                raise UnreadablePretextError("a window holds a channel the corpus does not name")
            columns = column_of[hidden.channels]
            error = answerer(hidden, masked) - hidden.targets
            loss = np.where(hidden.scored, self._loss(error).mean(axis=-1), 0.0)
            missed[at, columns] += loss.sum(axis=0)
            counted[at, columns] += hidden.scored.sum(axis=0)
        errors = ChannelErrors(channels=channels, missed=missed, counted=counted)
        if errors.unread.any():
            logger.info(
                "%d of %d %s windows held no hidden cell with a reading in any draw; "
                "they score the fitting side's median error",
                int(errors.unread.sum()),
                len(windows),
                side,
            )
        return errors

    def _loss(self, error: NDArray[np.float64]) -> NDArray[np.float64]:
        """The backbone's own reading of a miss: squared, or Huber's past its knee."""
        knee = self._source.huber_delta
        if knee == 0.0:
            return error * error
        magnitude = np.abs(error)
        return np.where(magnitude <= knee, 0.5 * error * error, knee * (magnitude - 0.5 * knee))

    def _drawn(
        self,
        read: ReadCorpus,
        windows: Sequence[TaskWindow],
        recipe: DetectorRecipe,
        asking: PretextAsking,
        *,
        answered: bool,
    ) -> Iterator[tuple[MaskedCells, HiddenCells, int]]:
        """Every draw of every window beside its place in ``windows``, a chunk at a time."""
        for start in range(0, len(windows), _CHUNK):
            chunk = windows[start : start + _CHUNK]
            # Read once a chunk: the block reads a window again at every index asked of it.
            tokens = list(read.windows(chunk))
            asked = [
                (start + at, tokens[at], self._seed(recipe.seed, window, draw))
                for draw in range(asking.draws)
                for at, window in enumerate(chunk)
            ]
            masked = self._source.masked(
                asking.backbone,
                [token for _, token, _ in asked],
                [seed for _, _, seed in asked],
                answered=answered,
                device=self._device,
                hidden=asking.hidden,
            )
            for (at, _, _), cells in zip(asked, masked, strict=True):
                yield cells, HiddenCells.of(cells, self._source.bins), at

    @staticmethod
    def _seed(seed: int, window: TaskWindow, draw: int) -> int:
        """The seed of one draw of one window: a digest of both, within what a generator takes."""
        digest = seeded_rank(seed, "pretext", str(window.unit), window.position, draw)
        return int.from_bytes(digest[:8]) >> 1

    @staticmethod
    def _half(seed: int, window: TaskWindow) -> int:
        """Which half of the fitting units a window's unit falls in, for choosing a penalty."""
        return seeded_rank(seed, "half", str(window.unit))[0] & 1
