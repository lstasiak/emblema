#!/usr/bin/env bash
# Smoke test of the local stack after `docker compose up -d --wait`: each service does the one
# thing the platform needs from it, and the API answers over them. Stops at the first failure.
# Needs curl and docker compose.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Addressed by IPv4 literal, not `localhost`: compose publishes the port on 127.0.0.1 only, while
# `localhost` reaches ::1 first. On macOS that socket belongs to AirPlay Receiver, which answers 403.
MLFLOW_URL=${MLFLOW_URL:-http://127.0.0.1:${MLFLOW_PORT:-5000}}
API_URL=${API_URL:-http://127.0.0.1:${EMBLEMA_API__PORT:?EMBLEMA_API__PORT is required (copy env.example to .env)}}
BUCKET=${EMBLEMA_ARTIFACT_STORE__BUCKET:-emblema}
PG_USER=${EMBLEMA_DATABASE__USER:?EMBLEMA_DATABASE__USER is required (copy env.example to .env)}
PG_DB=${EMBLEMA_DATABASE__NAME:?EMBLEMA_DATABASE__NAME is required (copy env.example to .env)}
BROKER_USER=${EMBLEMA_BROKER__USER:?EMBLEMA_BROKER__USER is required (copy env.example to .env)}
BROKER_VHOST=${EMBLEMA_BROKER__VHOST:?EMBLEMA_BROKER__VHOST is required (copy env.example to .env)}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

psql_scalar() {
  docker compose exec -T postgres psql -U "$PG_USER" -d "$PG_DB" -tAc "$1"
}

echo "postgres: databases and context schemas"
[[ $(psql_scalar "select count(*) from pg_database where datname = 'mlflow'") == 1 ]] \
  || fail "database mlflow is missing"
schemas=$(psql_scalar "select string_agg(schema_name, ',' order by schema_name) from information_schema.schemata where schema_name in ('catalog', 'evaluation', 'pretraining', 'serving')")
[[ $schemas == "catalog,evaluation,pretraining,serving" ]] || fail "expected four context schemas, got '${schemas}'"

echo "broker: the virtual host and the role the application connects as"
# Read whole, then matched: a healthcheck says the node is up, this says the vhost and the role
# the application's URL names are the ones the broker actually has. The default vhost is "/",
# which only survives the URL percent-encoded, so a wrong one shows up here rather than at the
# first job.
# `tail -n +2` drops the column header, which would otherwise match a vhost called "name".
vhosts=$(docker compose exec -T rabbitmq rabbitmqctl -q list_vhosts | tail -n +2)
grep -qx -- "$BROKER_VHOST" <<<"$vhosts" \
  || fail "the broker has no virtual host '${BROKER_VHOST}' (has: $(echo "$vhosts" | tr '\n' ' '))"
permissions=$(docker compose exec -T rabbitmq rabbitmqctl -q list_permissions --vhost "$BROKER_VHOST" | tail -n +2)
grep -q "^${BROKER_USER}[[:space:]]" <<<"$permissions" \
  || fail "role '${BROKER_USER}' has no permissions on virtual host '${BROKER_VHOST}'"

echo "bucket: lifecycle rules"
# `-T` denies the container a terminal, which aws-cli v2 would take as an invitation to page its
# output. Reading each pipe to the end keeps `pipefail` from seeing a writer killed by SIGPIPE.
docker compose run --rm -T --no-deps --entrypoint aws bootstrap \
  s3api get-bucket-lifecycle-configuration --bucket "$BUCKET" \
  | grep "transient" >/dev/null \
  || fail "no lifecycle rule for transient artifacts on ${BUCKET}"

echo "mlflow: a run with a metric and an artifact"
post() {
  curl -fsS -X POST -H "Content-Type: application/json" "${MLFLOW_URL}/api/2.0/mlflow/$1" -d "$2"
}
now_ms() { echo "$(date +%s)000"; }

experiment=$(post experiments/create "{\"name\": \"smoke-$(date +%s)\"}" \
  | grep -oE '"experiment_id": ?"[0-9]+"' | grep -oE '[0-9]+')
run=$(post runs/create "{\"experiment_id\": \"${experiment}\", \"start_time\": $(now_ms)}" \
  | grep -oE '"run_id": ?"[a-f0-9]{32}"' | head -1 | grep -oE '[a-f0-9]{32}')
post runs/log-metric "{\"run_id\": \"${run}\", \"key\": \"smoke\", \"value\": 1.0, \"timestamp\": $(now_ms), \"step\": 0}" >/dev/null
# Artifacts go through the server into the bucket: this is the MLflow -> S3 wiring under test.
curl -fsS -X PUT --data-binary "smoke" \
  "${MLFLOW_URL}/api/2.0/mlflow-artifacts/artifacts/${experiment}/${run}/artifacts/smoke.txt" >/dev/null
post runs/update "{\"run_id\": \"${run}\", \"status\": \"FINISHED\", \"end_time\": $(now_ms)}" >/dev/null

curl -fsS "${MLFLOW_URL}/api/2.0/mlflow/runs/get?run_id=${run}" \
  | grep -E '"key": ?"smoke"' >/dev/null \
  || fail "metric not recorded on run ${run}"
[[ $(curl -fsS "${MLFLOW_URL}/api/2.0/mlflow-artifacts/artifacts/${experiment}/${run}/artifacts/smoke.txt") == "smoke" ]] \
  || fail "artifact not readable back from run ${run}"

echo "api: alive, ready against the stack, described, and answering a list"
[[ $(curl -fsS "${API_URL}/health") == '{"status":"ok"}' ]] || fail "the API does not report up"
# Readiness reaches the database and the bucket from inside the network: a probe that passes
# here says the process was told where they are, not only that it started.
curl -fsS "${API_URL}/ready" | grep -q '"status":"ready"' || fail "the API is not ready: $(curl -sS "${API_URL}/ready")"
curl -fsS "${API_URL}/openapi.json" | grep -q '/served-models/{served_model_id}/predictions' \
  || fail "the schema does not describe the prediction route"
curl -fsS "${API_URL}/served-models" | grep -q '"items":' || fail "the API does not list served models"
# A refusal answers as a problem document, whatever was asked.
[[ $(curl -sS -o /dev/null -w '%{content_type}' "${API_URL}/served-models/not-a-uuid") == "application/problem+json" ]] \
  || fail "a refusal is not a problem document"

echo "OK: postgres, broker, bucket, mlflow and the api are ready"
