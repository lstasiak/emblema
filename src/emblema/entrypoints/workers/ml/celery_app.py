"""The Celery application the worker of the ``ml`` queue is started with, and the job it answers.

Which queue this process consumes is the command it is started with, and which queue a
job goes to is the job's own pool. Neither is declared on the task: a third statement of
the same fact would be one Celery shares between applications of one process, so the
module imported first would answer for the other.

Module level because that is the shape Celery discovers: ``celery -A`` imports this module and
expects to find the application and the tasks registered on it. Everything a task does is on
``CampaignWorker``; what is here is the wiring the queue insists on, and the worker is built on
first use rather than at import, so that a process which only routes — one that imports this to
send a task — needs neither a backbone nor a database.
"""

from celery.signals import worker_init

from emblema.config.settings import Settings
from emblema.entrypoints.workers.campaign_worker import CampaignWorker
from emblema.entrypoints.workers.ml.composition_root import CompositionRoot
from emblema.evaluation.application.use_cases.advance_campaign import RUN_CAMPAIGN_CELL
from emblema.shared.adapters.queues.celery_application import celery_application
from emblema.shared.adapters.queues.celery_job_queue import CeleryJobQueue
from emblema.shared.jobs.job_argument import JobArgument

app = celery_application(Settings().require_broker().url())

_worker: CampaignWorker | None = None


def worker() -> CampaignWorker:  # pragma: no cover - environment
    """The worker this process runs jobs with, built once on first use.

    It submits through this module's application rather than opening one of its own, so the
    process holds a single Celery application: a second would take over as the current one and
    carry none of the configuration this one was given.
    """
    global _worker
    if _worker is None:
        root = CompositionRoot.from_environment(CeleryJobQueue(app))
        _worker = CampaignWorker(root.services.run_campaign_cell)
    return _worker


@app.task(name=RUN_CAMPAIGN_CELL)
def run_campaign_cell(**arguments: JobArgument) -> None:  # pragma: no cover - environment
    """Adapt a backbone for one cell of a campaign."""
    worker().run_campaign_cell(arguments)


@worker_init.connect
def build_on_startup(**_: object) -> None:  # pragma: no cover - environment
    """Assemble the process where it is started rather than at the first cell it is handed.

    A worker missing what it serves would otherwise report for work, take a cell and fail on it,
    which costs a cell and tells whoever started it nothing. Connected to the signal rather than
    run on import, because a process that imports this module only to submit a job needs none of
    what a consumer needs.

    Raised as an exit rather than let out as it comes: Celery keeps what a signal handler raises
    to itself and reports it through a logger it has not configured this early, so anything
    short of leaving the process would be a refusal nobody sees.
    """
    try:
        worker()
    except ValueError as error:
        raise SystemExit(f"this worker cannot serve what it was started for: {error}") from error
