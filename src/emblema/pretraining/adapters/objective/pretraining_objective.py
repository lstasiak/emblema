from abc import ABC, abstractmethod

from torch import nn

from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder
from emblema.pretraining.adapters.objective.objective_score import ObjectiveScore
from emblema.pretraining.adapters.objective.token_masks import TokenMasks
from emblema.pretraining.domain.training.objective_loss import ObjectiveLoss
from emblema.shared.adapters.tensors.token_tensors import TokenTensors


class PretrainingObjective(nn.Module, ABC):
    """What a run trains: an encoder and whatever is thrown away once it is trained.

    Every objective hides part of a window, encodes the rest and predicts something about the
    hidden tokens; what it predicts, and against what, is the subclass's. The runtime asks only
    for the score of a batch and tells the objective when a step has been taken, so a second
    objective is a second subclass and the runtime stays as it was. Not a port: an objective
    speaks in tensors, which no interface of the application could, and the run it belongs to
    is already behind one, ``TrainingRuntime``.

    Attributes:
        encoder: The backbone being pretrained; what remains when the objective is done.
    """

    encoder: SetEncoder

    @abstractmethod
    def score(self, batch: TokenTensors, masks: TokenMasks, loss: ObjectiveLoss) -> ObjectiveScore:
        """The error of the hidden tokens that were observed under ``loss``, and the trivial one.

        The reading is passed rather than held so that the weights an artifact stores are the
        objective alone, and a diagnostic may score them by any reading it names.
        """

    def after_step(self, step: int, steps: int) -> None:
        """What the objective does once the optimiser has taken step ``step`` of ``steps``.

        Nothing, unless a subclass keeps state the optimiser does not move. Called for every
        step the run counts, including one a gradient scaler skipped: the schedule counts those
        too, and a resumed run must take the same path.
        """
        return None
