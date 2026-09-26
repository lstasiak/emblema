"""Boosted trees as a campaign keeps them, for the runtime tests: the document and its corpus.

Apparatus. A few shallow trees fitted on invented rows under the reading that summarises
across channels, kept the way the classical runtime keeps them, over the shared test corpus.
"""

import numpy as np
import xgboost

from emblema.catalog.contracts.published_corpus_manifest_json import PublishedCorpusManifestJson
from emblema.evaluation.adapters.artifacts.kept_candidates import KeptCandidates
from emblema.evaluation.adapters.artifacts.representation_bytes import RepresentationBytes
from emblema.evaluation.adapters.features.channel_aggregated_features import (
    ChannelAggregatedFeatures,
)
from emblema.evaluation.adapters.xgboost.fitted_baseline import FittedBaseline
from emblema.evaluation.contracts.candidate_kind import CandidateKind
from emblema.evaluation.domain.classical.feature_scheme import FeatureScheme
from emblema.shared.kernel.artifacts import ArtifactRef
from emblema.shared.kernel.checksums import Checksum
from emblema.shared.kernel.tokens import TokenWindow
from emblema.shared.ports.artifact_store import ArtifactStore
from tests.evaluation.adapters.features.support import timed, window
from tests.evaluation.support import recipe
from tests.support.published import manifest_of

SCALE = 125.0
BLOCK = ArtifactRef(key="durable/trees-block", checksum=Checksum.of_bytes(b"trees block"))


def trees_windows(count: int) -> list[TokenWindow]:
    """``count`` windows over the two measured channels of the shared test corpus."""
    return [
        window(
            timed(1, [(0.2 * n + 0.1 * i, 0.1 * (i + 1)) for i in range(4)]),
            timed(2, [(1.0 - 0.1 * n, 0.3), (0.5 + 0.1 * n, 0.9)]),
        )
        for n in range(count)
    ]


def fitted_trees(scheme: FeatureScheme = FeatureScheme.CHANNEL_AGGREGATED) -> FittedBaseline:
    features = ChannelAggregatedFeatures()
    rows = features.of(trees_windows(6))
    model = xgboost.XGBRegressor(n_estimators=4, max_depth=2, n_jobs=1, random_state=1)
    model.fit(rows, np.linspace(0.1, 0.6, 6))
    return FittedBaseline.of(
        recipe(scheme), model, feature_names=features.names(), target_scale=SCALE
    )


def trees_corpus(store: ArtifactStore) -> ArtifactRef:
    return store.put(PublishedCorpusManifestJson().encode(manifest_of(BLOCK)))


def keep_trees(store: ArtifactStore) -> ArtifactRef:
    """The manifest of kept trees: the document as their only form."""
    return KeptCandidates(store).keep(
        CandidateKind.CLASSICAL,
        corpus_manifest=trees_corpus(store),
        measured=RepresentationBytes(FittedBaseline.FORMAT, fitted_trees().to_bytes()),
    )


def trees_answers(fitted: FittedBaseline, windows: list[TokenWindow]) -> list[float]:
    """What the trees answer for ``windows``, read the way the runtime reads them back."""
    rows = ChannelAggregatedFeatures().of(windows)
    return [float(a) for a in (fitted.booster().inplace_predict(rows) * fitted.target_scale)]
