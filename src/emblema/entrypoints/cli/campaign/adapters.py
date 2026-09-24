from dataclasses import dataclass

from emblema.evaluation.ports.campaign_handoff import CampaignHandoff
from emblema.evaluation.ports.candidate_catalogue import CandidateCatalogue
from emblema.evaluation.ports.corpus_windows import CorpusWindows
from emblema.evaluation.ports.downstream_task_repository import DownstreamTaskRepository
from emblema.evaluation.ports.evaluation_campaign_repository import EvaluationCampaignRepository
from emblema.shared.ports.job_queue import JobQueue


@dataclass(frozen=True)
class Adapters:
    """The port implementations this command line runs on.

    The corpus is here and not only behind a use case because naming a task's units needs the
    division of the corpus they are named in, and only the corpus can say what that is. The
    register turns that division into the command; nothing is decided here. The candidates are
    a catalogue and not a provider, because this process asks what each competitor is and never
    runs one.
    """

    corpus: CorpusWindows
    candidates: CandidateCatalogue
    tasks: DownstreamTaskRepository
    campaigns: EvaluationCampaignRepository
    jobs: JobQueue
    handoff: CampaignHandoff
