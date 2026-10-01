from dataclasses import dataclass

from emblema.catalog.contracts.window_tokeniser import WindowTokeniser
from emblema.evaluation.ports.campaign_listing import CampaignListing
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.evaluation.ports.verdict_memo import VerdictMemo
from emblema.serving.ports.inference_runtime import InferenceRuntime
from emblema.serving.ports.promotable_artifact_repository import PromotableArtifactRepository
from emblema.serving.ports.served_model_listing import ServedModelListing
from emblema.serving.ports.served_model_repository import ServedModelRepository
from emblema.shared.ports.artifact_store import ArtifactStore
from emblema.shared.ports.clock import Clock
from emblema.shared.ports.id_generator import IdGenerator
from emblema.shared.ports.identity_provider import IdentityProvider


@dataclass(frozen=True)
class Adapters:
    """Every port implementation the process runs on, so that what a use case got can be read.

    The tokeniser is not a port of this process's contexts but the service the Catalog
    publishes; it is listed with the adapters because it is composed in the same way and a
    reader asks the same question of it. The identity provider is consumed by the HTTP edge,
    not by a use case, and is listed for the same reason.
    """

    store: ArtifactStore
    identity: IdentityProvider
    served: ServedModelRepository
    served_listing: ServedModelListing
    promotables: PromotableArtifactRepository
    campaigns: EvaluationCampaignRepository
    campaign_listing: CampaignListing
    verdicts: VerdictMemo
    runtime: InferenceRuntime
    tokeniser: WindowTokeniser
    clock: Clock
    ids: IdGenerator
