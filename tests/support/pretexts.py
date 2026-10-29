"""A backbone over cells stored the way a run stores one, for the detectors that ask its pretext.

The objective is drawn under a fixed seed and stored with the masks and the loss of the test
experiment, so a detector asking it is asked over the same draws every time.
"""

import torch

from emblema.pretraining.adapters.objective.objective_shape import ObjectiveShape
from emblema.pretraining.adapters.training.trained_model import TrainedModel
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.ports.artifact_store import ArtifactStore
from tests.support.experiments import CELLS, CHANNELS, OBSERVATION, configuration

PRETEXT_SEED = 6


def stored_pretext(store: ArtifactStore) -> ArtifactRef:
    """A small objective over cells, stored in ``store`` with what it hid and how it scored."""
    stated = configuration(architecture=CELLS, observation=OBSERVATION)
    torch.manual_seed(PRETEXT_SEED)
    trained = TrainedModel.of(stated, CHANNELS, ObjectiveShape.of(stated, CHANNELS).build())
    return store.put(trained.to_bytes())
