from collections.abc import Callable
from dataclasses import dataclass

from emblema.evaluation.application.use_cases.list_campaign_runs import ListCampaignRuns
from emblema.evaluation.application.use_cases.list_campaigns import ListCampaigns
from emblema.evaluation.application.use_cases.view_campaign import ViewCampaign
from emblema.serving.application.read_models.embeddings import Embeddings
from emblema.serving.application.read_models.predictions import Predictions
from emblema.serving.application.use_cases.embed_windows import EmbedWindowsQuery
from emblema.serving.application.use_cases.list_served_models import ListServedModels
from emblema.serving.application.use_cases.predict_windows import PredictWindowsQuery
from emblema.serving.application.use_cases.promote_artifact import PromoteArtifact
from emblema.serving.application.use_cases.view_served_model import ViewServedModel
from emblema.serving.application.use_cases.withdraw_served_model import WithdrawServedModel


@dataclass(frozen=True)
class Services:
    """The use cases the HTTP process answers with, each already holding its dependencies.

    The two that answer windows are typed as the driving ports they are, a callable from query
    to result, because the root may hand them over counted rather than bare.
    """

    predict_windows: Callable[[PredictWindowsQuery], Predictions]
    embed_windows: Callable[[EmbedWindowsQuery], Embeddings]
    view_served_model: ViewServedModel
    list_served_models: ListServedModels
    promote_artifact: PromoteArtifact
    withdraw_served_model: WithdrawServedModel
    view_campaign: ViewCampaign
    list_campaigns: ListCampaigns
    list_campaign_runs: ListCampaignRuns
