"""Every form a runtime keeps is named by a format the manifest finds it under.

The loader another context builds reads a kept candidate by the manifest alone: it looks a form
up by format and hands the bytes to the codec that owns the name. So the contract every codec
meets is one and the same, a name of its own and bytes that come back through the manifest as
what went in, and it is stated once here rather than once per codec.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest

torch = pytest.importorskip("torch")
xgboost = pytest.importorskip("xgboost")

from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates  # noqa: E402
from emblema.evaluation.adapters.artifacts.representation_bytes import (  # noqa: E402
    RepresentationBytes,
)
from emblema.evaluation.adapters.features.channel_aggregated_features import (  # noqa: E402
    ChannelAggregatedFeatures,
)
from emblema.evaluation.adapters.minirocket.fitted_convolutions import (  # noqa: E402
    FittedConvolutions,
)
from emblema.evaluation.adapters.onnx.inference_graph import InferenceGraph  # noqa: E402
from emblema.evaluation.adapters.torch.adapted_backbone import AdaptedBackbone  # noqa: E402
from emblema.evaluation.adapters.torch.fitted_candidate import FittedCandidate  # noqa: E402
from emblema.evaluation.adapters.torch.fitted_patch_model import FittedPatchModel  # noqa: E402
from emblema.evaluation.adapters.torch.grid_reading import GridReading  # noqa: E402
from emblema.evaluation.adapters.torch.patch_transformer import PatchTransformer  # noqa: E402
from emblema.evaluation.adapters.torch.regression_head import RegressionHead  # noqa: E402
from emblema.evaluation.adapters.xgboost.fitted_baseline import FittedBaseline  # noqa: E402
from emblema.evaluation.contracts.candidate_kind import CandidateKind  # noqa: E402
from emblema.pretraining.adapters.encoder.set_encoder import SetEncoder  # noqa: E402
from emblema.shared.adapters.in_memory.artifact_store import InMemoryArtifactStore  # noqa: E402
from emblema.shared.kernel.artifacts import ArtifactRef  # noqa: E402
from emblema.shared.kernel.checksums import Checksum  # noqa: E402
from tests.evaluation.adapters.features.support import timed, window  # noqa: E402
from tests.evaluation.support import convolutions, patch_plan, plan, recipe  # noqa: E402
from tests.support.experiments import CHANNELS, TINY  # noqa: E402

pytestmark = pytest.mark.ml

type Fitted = FittedCandidate | FittedBaseline | FittedConvolutions | FittedPatchModel

CORPUS = ArtifactRef(key="durable/corpus", checksum=Checksum.of_bytes(b"corpus"))
SCALE = 125.0


def fitted_candidate() -> FittedCandidate:
    torch.manual_seed(1)
    encoder = SetEncoder.for_vocabulary(TINY, CHANNELS)
    candidate = AdaptedBackbone(encoder, RegressionHead(TINY.width, starting_at=0.5))
    return FittedCandidate.of(plan(), candidate, vocabulary_size=CHANNELS, target_scale=SCALE)


def fitted_baseline() -> FittedBaseline:
    features = ChannelAggregatedFeatures()
    rows = [[float(row + column) for column in range(features.width)] for row in range(6)]
    model = xgboost.XGBRegressor(n_estimators=4, max_depth=2, n_jobs=1, random_state=1)
    model.fit(rows, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    return FittedBaseline.of(recipe(), model, feature_names=features.names(), target_scale=SCALE)


def fitted_convolutions() -> FittedConvolutions:
    times = np.sort(np.random.default_rng(5).uniform(0.0, 1.0, 50))
    windows = [
        window(
            timed(1, [(float(np.sin(2 * np.pi * cycles * t)), float(t)) for t in times]),
            timed(2, [(float(t), float(t)) for t in times[::3]]),
        )
        for cycles in (1.0, 2.0, 3.0) * 2
    ]
    targets = np.array((1.0, 2.0, 3.0) * 2) / 10.0
    return FittedConvolutions.fitted(
        recipe(method=convolutions()), convolutions(), 2, 16, windows, targets, SCALE
    )


def fitted_patch_model() -> FittedPatchModel:
    torch.manual_seed(1)
    reading = GridReading(steps=16, channels=3, held=(0, 2))
    model = PatchTransformer(
        patch_plan().spec, channels=len(reading.held), steps=reading.steps, starting_at=0.0
    )
    return FittedPatchModel.of(patch_plan(), model, reading=reading, target_scale=SCALE)


class Kept(NamedTuple):
    """A codec of a measured form, a way to make one, and the kind of candidate it belongs to."""

    codec: type[Fitted]
    make: Callable[[], Fitted]
    kind: CandidateKind


KEPT = (
    Kept(FittedCandidate, fitted_candidate, CandidateKind.NEURAL),
    Kept(FittedBaseline, fitted_baseline, CandidateKind.CLASSICAL),
    Kept(FittedConvolutions, fitted_convolutions, CandidateKind.CLASSICAL),
    Kept(FittedPatchModel, fitted_patch_model, CandidateKind.NEURAL),
)
CODECS = (*(kept.codec for kept in KEPT), InferenceGraph)


def test_every_codec_names_its_form_and_no_two_share_a_name() -> None:
    names = [codec.FORMAT for codec in CODECS]

    assert all(name and name == name.strip() for name in names)
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("kept", KEPT, ids=[kept.codec.FORMAT for kept in KEPT])
def test_a_form_kept_under_its_format_reads_back_through_the_manifest(kept: Kept) -> None:
    store = InMemoryArtifactStore()
    fitted = kept.make()

    manifest = KeptCandidates(store).keep(
        kept.kind,
        corpus_manifest=CORPUS,
        measured=RepresentationBytes(kept.codec.FORMAT, fitted.to_bytes()),
    )

    form = KeptCandidates(store).read(manifest).find(kept.codec.FORMAT)
    assert form is not None
    read = kept.codec.read(store.get(form.artifact))
    assert (read.parameters, read.target_scale) == (fitted.parameters, fitted.target_scale)
