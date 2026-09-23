"""The one Celery application this system builds, however a process came to need one.

A function rather than a class, because there is no state to hold. Stated once because two
applications configured apart would agree until the day one of them gained an option.
"""

from celery import Celery

from emblema.shared.jobs.worker_pool import WorkerPool


def celery_application(broker_url: str, *, name: str = "emblema") -> Celery:
    """A Celery application on that broker, configured as every process of this system needs it.

    Late acknowledgement, so a cell whose worker died is delivered again rather than lost; the
    use case answers a cell it has already recorded from what is stored, so a second delivery
    costs a query. On AMQP that acknowledgement is the broker's own: a job stays unacknowledged
    for exactly as long as the worker holds it, however many hours a cell takes, with no timeout
    to guess at. Publishing waits for the broker to confirm, so a submission that returns is one
    the broker took responsibility for rather than one that was written to a socket.
    """
    app = Celery(name, broker=broker_url)
    app.conf.task_ignore_result = True
    app.conf.task_acks_late = True
    app.conf.task_default_queue = str(WorkerPool.ML)
    app.conf.broker_transport_options = {"confirm_publish": True}
    # Celery's worker-to-worker mailbox declares transient non-exclusive queues, which RabbitMQ 4
    # refuses outright: a worker that opens one never finishes starting. Nothing here inspects or
    # controls workers at runtime, so it is switched off rather than the broker asked to permit
    # what it has deprecated. This covers the mailbox; the startup handshakes that also use it
    # are command-line flags with no setting behind them, so the worker is started with
    # `--without-mingle --without-gossip` (README).
    app.conf.worker_enable_remote_control = False
    return app
