from emblema.config.worker_settings import WorkerSettings
from emblema.evaluation.domain.classical.gradient_boosting_spec import GradientBoostingSpec
from emblema.evaluation.domain.classical.minirocket_spec import MiniRocketSpec
from emblema.evaluation.domain.classical.random_convolutions import RandomConvolutions
from emblema.evaluation.domain.classical.ridge_spec import RidgeSpec
from emblema.evaluation.domain.patching.patch_model_spec import PatchModelSpec
from emblema.evaluation.domain.transfer.adaptation_schedule import AdaptationSchedule
from emblema.evaluation.domain.transfer.lora_spec import LoraSpec


class DeclaredWorker:
    """A worker's settings read back into the values a process is assembled from.

    The one place scalars from the environment become the value objects that judge them, so
    every process that reads the same group gets the same reading. Here rather than on the
    settings, which know nothing of any context, and here rather than on a composition root,
    because the roots that run a grid carry the stacks that run it and a process which only
    declares a campaign must carry neither.
    """

    def __init__(self, settings: WorkerSettings) -> None:
        self._settings = settings

    def schedule(self) -> AdaptationSchedule:
        """How every cell of this process's campaigns learns.

        Raises:
            ValueError: If nothing is configured.
            InvalidAdaptationScheduleError: If what is configured is not a schedule that
                stands up.
        """
        declared = self._settings.require_schedule()
        return AdaptationSchedule(
            epochs=declared.epochs,
            min_steps=declared.min_steps,
            batch_size=declared.batch_size,
            learning_rate=declared.learning_rate,
            weight_decay=declared.weight_decay,
            warmup_fraction=declared.warmup_fraction,
            final_lr_fraction=declared.final_lr_fraction,
        )

    def lora(self) -> LoraSpec:
        """The low-rank updates the arm of that name adds.

        Raises:
            ValueError: If nothing is configured.
            InvalidLoraSpecError: If what is configured is not a specification that stands up.
        """
        declared = self._settings.require_lora()
        return LoraSpec(
            rank=declared.rank,
            alpha=declared.alpha,
            dropout=declared.dropout,
            targets=tuple(
                target.strip() for target in declared.targets.split(",") if target.strip()
            ),
        )

    def boosting(self) -> GradientBoostingSpec:
        """How hard the classical candidates of this process's campaigns fit.

        Raises:
            ValueError: If nothing is configured.
            InvalidGradientBoostingSpecError: If what is configured is not a fit that stands up.
        """
        declared = self._settings.require_boosting()
        return GradientBoostingSpec(
            rounds=declared.rounds,
            max_depth=declared.max_depth,
            learning_rate=declared.learning_rate,
            row_share=declared.row_share,
            feature_share=declared.feature_share,
            min_leaf_weight=declared.min_leaf_weight,
            l2_penalty=declared.l2_penalty,
            threads=declared.threads,
        )

    def convolutions(self) -> RandomConvolutions:
        """How the MiniRocket candidate of this process's campaigns reads and fits.

        Raises:
            ValueError: If nothing is configured, or a penalty is not a number.
            InvalidMiniRocketSpecError: If the count of convolutions does not stand up.
            InvalidRidgeSpecError: If the penalties or the threads do not stand up.
        """
        declared = self._settings.require_convolutions()
        return RandomConvolutions(
            convolutions=MiniRocketSpec(features=declared.features),
            ridge=RidgeSpec(
                penalties=tuple(
                    float(penalty)
                    for penalty in declared.ridge_penalties.split(",")
                    if penalty.strip()
                ),
                threads=declared.threads,
            ),
        )

    def patch(self) -> PatchModelSpec:
        """How the patch model of this process's campaigns reads a window and how large it is.

        Raises:
            ValueError: If nothing is configured.
            InvalidPatchModelSpecError: If what is configured is not a shape that stands up.
        """
        declared = self._settings.require_patch()
        return PatchModelSpec(
            patch_length=declared.patch_length,
            stride=declared.stride,
            width=declared.width,
            heads=declared.heads,
            layers=declared.layers,
            feedforward_width=declared.feedforward_width,
            dropout=declared.dropout,
            grid_resolution=declared.grid_resolution,
        )
