# syntax=docker/dockerfile:1

# The application, as every process of it is deployed. One recipe, one image per set of optional
# dependencies: a worker that fits classical candidates has no use for the training stack and a
# worker that trains has no use for the classical one, so which extras an image carries is a
# build argument rather than one fatter image that carries both. Which process an image runs is
# the command, because that is the only thing that differs between them.
#
# The floor of `requires-python` rather than the version used for development: this is also the
# Python the free GPU platforms ship, so what runs here runs there.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12

# Named so that the copies below can refer to it: a `COPY --from` takes no variable.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv

# The environment sits outside the working directory so that bind-mounting the source over it —
# which the test image does — cannot shadow the packages.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Optional dependencies this image carries, as named in pyproject.toml.
ARG EXTRAS=ml

# The lock alone first: a change to the source then re-installs the project and nothing else.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra "${EXTRAS}"

COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra "${EXTRAS}"


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

# The workspace is made here rather than by the volume that will take it over: a named volume
# inherits the ownership of the path it covers, and a path Docker creates belongs to root, which
# an unprivileged process cannot write to on a host whose user ids are not the image's.
RUN useradd --system --create-home --uid 10001 emblema \
    && mkdir -p /home/emblema/data/workspace \
    && chown -R emblema:emblema /home/emblema

USER emblema
WORKDIR /home/emblema

# The common case, stated so that the image says what it is for; whoever starts a process names
# the queue it serves. The two handshakes a worker performs with its neighbours run over a kind
# of queue RabbitMQ 4 has withdrawn, and this system's workers have nothing to say to each other.
CMD ["celery", "-A", "emblema.entrypoints.workers.celery_app", "worker", \
     "--queues", "ml", "--pool=solo", "--without-mingle", "--without-gossip"]


# The image the suite runs in. Everything installed — every extra and every development group —
# because one run of the tests is the gate, and the source is bind-mounted over the working
# directory so a run needs no rebuild. Git is here because the command line reads the revision
# it is installed at, and the test of that reads it out of a repository it makes.
FROM python:${PYTHON_VERSION}-slim AS tests
COPY --from=uv /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install --no-install-recommends --yes git

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --all-groups

CMD ["pytest"]
