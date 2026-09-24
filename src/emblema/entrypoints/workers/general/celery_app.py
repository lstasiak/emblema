"""The Celery application the ``general`` worker is started with, and the job it answers.

Module level because that is the shape Celery discovers: ``celery -A`` imports this module and
expects to find the application and the tasks registered on it. The worker is built on first
use rather than at import, so a process that imports this only to send a task needs
neither the boosting knobs nor a database.
"""

from celery.signals import worker_init

from emblema.config.settings import Settings
from emblema.entrypoints.workers.campaign_worker import CampaignWorker
from emblema.entrypoints.workers.general.composition_root import CompositionRoot
from emblema.evaluation.application.use_cases.advance_campaign import RUN_CAMPAIGN_CELL
from emblema.shared.adapters.queues.celery_application import celery_application
from emblema.shared.adapters.queues.celery_job_queue import CeleryJobQueue
from emblema.shared.jobs.job_argument import JobArgument

app = celery_application(Settings().require_broker().url())

_worker: CampaignWorker | None = None


def worker() -> CampaignWorker:  # pragma: no cover - environment
    """The worker this process runs jobs with, built once, submitting through this module's app.

    A second Celery application in the process would take over as the current one and carry
    none of the configuration this one was given.
    """
    global _worker
    if _worker is None:
        root = CompositionRoot.from_environment(CeleryJobQueue(app))
        _worker = CampaignWorker(root.services.run_campaign_cell)
    return _worker


@app.task(name=RUN_CAMPAIGN_CELL)
def run_campaign_cell(**arguments: JobArgument) -> None:  # pragma: no cover - environment
    """Fit a classical candidate for one cell of a campaign."""
    worker().run_campaign_cell(arguments)


@worker_init.connect
def build_on_startup(**_: object) -> None:  # pragma: no cover - environment
    """Assemble the process where it is started rather than at the first cell it is handed.

    A worker missing what it serves would otherwise report for work, take a cell and fail on it.
    Raised as an exit because Celery swallows what a signal handler raises and logs it through
    a logger it has not configured this early, so anything less would be a refusal nobody sees.
    """
    try:
        worker()
    except ValueError as error:
        raise SystemExit(f"this worker cannot serve what it was started for: {error}") from error
