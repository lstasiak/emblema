"""The Celery application the worker is started with, and the jobs it answers.

Module level because that is the shape Celery discovers: ``celery -A`` imports this module and
expects to find the application and the tasks registered on it. Everything a task does is on
``CampaignWorker``; what is here is the wiring the queue insists on, and the worker is built on
first use rather than at import, so that a process which only routes — one that imports this to
send a task — needs neither a backbone nor a database.
"""

from emblema.config.settings import Settings
from emblema.entrypoints.workers.campaign_worker import CampaignWorker
from emblema.evaluation.application.use_cases.advance_campaign import RUN_CAMPAIGN_CELL
from emblema.shared.adapters.queues.celery_application import celery_application
from emblema.shared.adapters.queues.celery_job_queue import CeleryJobQueue
from emblema.shared.jobs.job_argument import JobArgument
from emblema.shared.jobs.worker_pool import WorkerPool

app = celery_application(Settings().require_broker().url())

_worker: CampaignWorker | None = None


def worker() -> CampaignWorker:
    """The worker this process runs jobs with, built once on first use.

    It submits through this module's application rather than opening one of its own, so the
    process holds a single Celery application: a second would take over as the current one and
    carry none of the configuration this one was given.
    """
    global _worker
    if _worker is None:
        _worker = CampaignWorker.from_environment(CeleryJobQueue(app))
    return _worker


@app.task(name=RUN_CAMPAIGN_CELL, queue=str(WorkerPool.ML))
def run_campaign_cell(**arguments: JobArgument) -> None:
    """Run one cell of a campaign on this machine."""
    worker().run_campaign_cell(arguments)
